// Single gRPC worker: loads block weights once, runs transformer_block_forward
// per request on a single CUDA stream, measures GPU time with cudaEvents.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <grpcpp/grpcpp.h>

#include "infer.grpc.pb.h"
#include "block.cuh"

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t _e = (call);                                                 \
    if (_e != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e),    \
              __FILE__, __LINE__);                                           \
      std::exit(2);                                                          \
    }                                                                        \
  } while (0)

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

static std::string arg_str(int argc, char** argv, const char* flag,
                           const std::string& def) {
  for (int i = 1; i < argc - 1; ++i)
    if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
  return def;
}
static int arg_int(int argc, char** argv, const char* flag, int def) {
  for (int i = 1; i < argc - 1; ++i)
    if (std::strcmp(argv[i], flag) == 0) return std::atoi(argv[i + 1]);
  return def;
}

static std::vector<float> read_bin(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(3); }
  std::streamsize n = f.tellg();
  f.seekg(0);
  std::vector<float> v(n / sizeof(float));
  f.read(reinterpret_cast<char*>(v.data()), n);
  return v;
}

static float* dev_copy(const float* h, size_t n) {
  float* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice));
  return d;
}

class WorkerService final : public infer::InferService::Service {
 public:
  WorkerService(const std::string& weights_path, const std::string& dims_path,
                int worker_id)
      : worker_id_(worker_id) {
    // model dims from dims.txt: B S D H ffn (B,S used only as default capacity)
    std::ifstream df(dims_path);
    if (!df) { fprintf(stderr, "cannot open %s\n", dims_path.c_str()); std::exit(3); }
    int bdef, sdef;
    df >> bdef >> sdef >> D_ >> H_ >> ffn_;
    dh_ = D_ / H_;

    // load weights once (order: Wqkv,bqkv,Wo,bo,W1,b1,W2,b2)
    std::vector<float> wb = read_bin(weights_path);
    size_t off = 0;
    auto take = [&](size_t n) { float* p = wb.data() + off; off += n; return dev_copy(p, n); };
    w_.Wqkv = take((size_t)D_ * 3 * D_); w_.bqkv = take(3 * D_);
    w_.Wo   = take((size_t)D_ * D_);     w_.bo   = take(D_);
    w_.W1   = take((size_t)D_ * ffn_);   w_.b1   = take(ffn_);
    w_.W2   = take((size_t)ffn_ * D_);   w_.b2   = take(D_);
    if (off != wb.size())
      fprintf(stderr, "[worker %d] WARN weights size mismatch (%zu used, %zu file)\n",
              worker_id_, off, wb.size());

    CUDA_CHECK(cudaStreamCreate(&stream_));
    CUDA_CHECK(cudaEventCreate(&ev_start_));
    CUDA_CHECK(cudaEventCreate(&ev_stop_));
    fprintf(stderr, "[worker %d] loaded weights (D=%d H=%d dh=%d ffn=%d)\n",
            worker_id_, D_, H_, dh_, ffn_);
  }

  Status Infer(ServerContext*, const infer::InferRequest* req,
               infer::InferResponse* resp) override {
    const int B = req->batch_size();
    const int S = req->seq_len();
    const int Dreq = req->hidden_dim();
    const size_t in_floats = (size_t)req->input_size();

    if (Dreq != D_) {
      return Status(grpc::StatusCode::INVALID_ARGUMENT,
                    "hidden_dim != model D");
    }
    if (in_floats != (size_t)B * S * D_) {
      return Status(grpc::StatusCode::INVALID_ARGUMENT,
                    "input size != B*S*D");
    }

    std::lock_guard<std::mutex> lock(gpu_mu_);  // single stream / shared scratch
    BlockConfig cfg{B, S, D_, H_, dh_, ffn_};
    ensure_capacity(cfg);

    const size_t io = (size_t)B * S * D_;
    CUDA_CHECK(cudaMemcpyAsync(dx_, req->input().data(), io * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));

    CUDA_CHECK(cudaEventRecord(ev_start_, stream_));
    transformer_block_forward(dx_, dout_, w_, cfg, scratch_, stream_);
    CUDA_CHECK(cudaEventRecord(ev_stop_, stream_));

    std::vector<float> out(io);
    CUDA_CHECK(cudaMemcpyAsync(out.data(), dout_, io * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start_, ev_stop_));

    auto* o = resp->mutable_output();
    o->Resize(out.size(), 0.f);
    std::memcpy(o->mutable_data(), out.data(), out.size() * sizeof(float));
    resp->set_compute_ms(ms);
    resp->set_request_id(req->request_id());
    resp->set_worker_id(worker_id_);

    fprintf(stderr,
            "[worker %d] req=%d B=%d S=%d  in=%zuB out=%zuB  compute=%.3fms\n",
            worker_id_, req->request_id(), B, S, io * sizeof(float),
            io * sizeof(float), ms);
    return Status::OK;
  }

 private:
  void ensure_capacity(const BlockConfig& cfg) {
    const size_t io = (size_t)cfg.B * cfg.S * cfg.D;
    if (io > io_cap_) {
      if (dx_) cudaFree(dx_);
      if (dout_) cudaFree(dout_);
      CUDA_CHECK(cudaMalloc(&dx_, io * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&dout_, io * sizeof(float)));
      io_cap_ = io;
    }
    const size_t sb = block_scratch_bytes(cfg);
    if (sb > scratch_cap_) {
      if (scratch_base_) cudaFree(scratch_base_);
      CUDA_CHECK(cudaMalloc(&scratch_base_, sb));
      scratch_cap_ = sb;
    }
    block_scratch_partition(scratch_, scratch_base_, cfg);
  }

  int worker_id_, D_, H_, dh_, ffn_;
  BlockWeights w_{};
  BlockScratch scratch_{};
  void* scratch_base_ = nullptr;
  size_t scratch_cap_ = 0;
  float* dx_ = nullptr;
  float* dout_ = nullptr;
  size_t io_cap_ = 0;
  cudaStream_t stream_{};
  cudaEvent_t ev_start_{}, ev_stop_{};
  std::mutex gpu_mu_;
};

int main(int argc, char** argv) {
  const int port = arg_int(argc, argv, "--port", 50051);
  const int id = arg_int(argc, argv, "--id", 0);
  const std::string weights = arg_str(argc, argv, "--weights", "fixtures/weights.bin");
  const std::string dims = arg_str(argc, argv, "--dims", "fixtures/dims.txt");

  WorkerService service(weights, dims, id);

  // larger max message: outputs can be many MB
  grpc::ServerBuilder builder;
  builder.SetMaxReceiveMessageSize(256 * 1024 * 1024);
  builder.SetMaxSendMessageSize(256 * 1024 * 1024);
  const std::string addr = "0.0.0.0:" + std::to_string(port);
  builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<Server> server(builder.BuildAndStart());
  fprintf(stderr, "[worker %d] listening on %s\n", id, addr.c_str());
  server->Wait();
  return 0;
}
