// OpenMP dispatcher: gRPC server to clients, gRPC client to workers.
// Queues incoming requests, batches same-shape requests within a time window,
// uses OpenMP to gather inputs / scatter outputs, and routes micro-batches to
// workers (least-loaded by default). Reports queue_ms per request.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <omp.h>
#include <grpcpp/grpcpp.h>
#include "infer.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using Clock = std::chrono::steady_clock;

static std::string arg_str(int c, char** v, const char* f, const std::string& d) {
  for (int i = 1; i < c - 1; ++i) if (!std::strcmp(v[i], f)) return v[i + 1];
  return d;
}
static int arg_int(int c, char** v, const char* f, int d) {
  for (int i = 1; i < c - 1; ++i) if (!std::strcmp(v[i], f)) return std::atoi(v[i + 1]);
  return d;
}

static std::vector<std::string> split_csv(const std::string& s) {
  std::vector<std::string> out;
  size_t p = 0;
  while (p < s.size()) {
    size_t q = s.find(',', p);
    if (q == std::string::npos) q = s.size();
    out.push_back(s.substr(p, q - p));
    p = q + 1;
  }
  return out;
}

struct Result {
  std::vector<float> output;
  double compute_ms = 0, queue_ms = 0;
  int worker_id = -1;
  bool ok = false;
  std::string err;
};

struct Pending {
  const infer::InferRequest* req;
  std::promise<Result> prom;
  Clock::time_point arrival;
  int S, D, batch;
};

class Dispatcher final : public infer::InferService::Service {
 public:
  Dispatcher(const std::vector<std::string>& workers, int window_ms,
             int max_batch, bool least_loaded)
      : window_(window_ms), max_batch_(max_batch), least_loaded_(least_loaded) {
    grpc::ChannelArguments ca;
    ca.SetMaxReceiveMessageSize(256 * 1024 * 1024);
    ca.SetMaxSendMessageSize(256 * 1024 * 1024);
    for (auto& w : workers) {
      auto ch = grpc::CreateCustomChannel(w, grpc::InsecureChannelCredentials(), ca);
      stubs_.push_back(infer::InferService::NewStub(ch));
    }
    inflight_.assign(workers.size(), 0);
    processed_.assign(workers.size(), 0);
    worker_addr_ = workers;
    batcher_ = std::thread([this] { batch_loop(); });
    fprintf(stderr, "[dispatcher] %zu workers, window=%dms max_batch=%d routing=%s\n",
            workers.size(), window_, max_batch_, least_loaded_ ? "least-loaded" : "round-robin");
  }

  ~Dispatcher() {
    { std::lock_guard<std::mutex> lk(mu_); stop_ = true; }
    cv_.notify_all();
    if (batcher_.joinable()) batcher_.join();
  }

  Status Infer(ServerContext*, const infer::InferRequest* req,
               infer::InferResponse* resp) override {
    Pending p;
    p.req = req;
    p.arrival = Clock::now();
    p.S = req->seq_len();
    p.D = req->hidden_dim();
    p.batch = req->batch_size();
    auto fut = p.prom.get_future();
    { std::lock_guard<std::mutex> lk(mu_); queue_.push_back(std::move(p)); }
    cv_.notify_one();

    Result r = fut.get();
    if (!r.ok) return Status(grpc::StatusCode::INTERNAL, r.err);
    auto* o = resp->mutable_output();
    o->Resize(r.output.size(), 0.f);
    std::memcpy(o->mutable_data(), r.output.data(), r.output.size() * sizeof(float));
    resp->set_compute_ms(r.compute_ms);
    resp->set_queue_ms(r.queue_ms);
    resp->set_request_id(req->request_id());
    resp->set_worker_id(r.worker_id);
    return Status::OK;
  }

  void print_stats() const {
    fprintf(stderr, "[dispatcher] per-worker processed micro-batches:");
    for (size_t i = 0; i < processed_.size(); ++i)
      fprintf(stderr, " w%zu(%s)=%ld", i, worker_addr_[i].c_str(), processed_[i]);
    fprintf(stderr, "\n");
  }

 private:
  struct MicroBatch { int S, D; std::vector<int> idx; };

  void batch_loop() {
    while (true) {
      std::vector<Pending> batch;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this] { return stop_ || !queue_.empty(); });
        if (stop_ && queue_.empty()) break;
        // accumulate for up to `window_` ms (or until max_batch requests)
        cv_.wait_for(lk, std::chrono::milliseconds(window_),
                     [this] { return stop_ || (int)queue_.size() >= max_batch_; });
        batch.swap(queue_);
      }
      process(batch);
    }
  }

  void process(std::vector<Pending>& batch) {
    // group by (S,D)
    std::map<std::pair<int, int>, std::vector<int>> groups;
    for (int i = 0; i < (int)batch.size(); ++i)
      groups[{batch[i].S, batch[i].D}].push_back(i);

    std::vector<MicroBatch> mbs;
    for (auto& g : groups) {
      auto& idxs = g.second;
      for (size_t s = 0; s < idxs.size(); s += max_batch_) {
        MicroBatch mb;
        mb.S = g.first.first;
        mb.D = g.first.second;
        for (size_t j = s; j < std::min(idxs.size(), s + (size_t)max_batch_); ++j)
          mb.idx.push_back(idxs[j]);
        mbs.push_back(std::move(mb));
      }
    }

    // dispatch micro-batches in parallel so multiple workers run concurrently
    std::vector<std::thread> ths;
    ths.reserve(mbs.size());
    for (auto& mb : mbs) ths.emplace_back([this, &batch, &mb] { send_microbatch(batch, mb); });
    for (auto& t : ths) t.join();
  }

  int pick_worker() {
    std::lock_guard<std::mutex> g(route_mu_);
    int best = 0;
    if (least_loaded_) {
      for (size_t w = 1; w < inflight_.size(); ++w)
        if (inflight_[w] < inflight_[best]) best = (int)w;
    } else {
      best = rr_++ % (int)inflight_.size();
    }
    inflight_[best]++;
    return best;
  }

  void send_microbatch(std::vector<Pending>& batch, MicroBatch& mb) {
    const int n = (int)mb.idx.size();
    const int S = mb.S, D = mb.D;
    std::vector<size_t> len(n), off(n);
    size_t total_floats = 0;
    long total_rows = 0;
    for (int i = 0; i < n; ++i) {
      len[i] = (size_t)batch[mb.idx[i]].batch * S * D;
      off[i] = total_floats;
      total_floats += len[i];
      total_rows += batch[mb.idx[i]].batch;
    }

    // --- OpenMP gather: pack each request's input into one batch buffer ---
    std::vector<float> input(total_floats);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
      std::memcpy(input.data() + off[i], batch[mb.idx[i]].req->input().data(),
                  len[i] * sizeof(float));

    const int wk = pick_worker();
    const auto t_dispatch = Clock::now();

    infer::InferRequest greq;
    greq.set_batch_size((int)total_rows);
    greq.set_seq_len(S);
    greq.set_hidden_dim(D);
    greq.set_request_id(-1);
    auto* gin = greq.mutable_input();
    gin->Resize(total_floats, 0.f);
    std::memcpy(gin->mutable_data(), input.data(), total_floats * sizeof(float));

    infer::InferResponse gresp;
    ClientContext cctx;
    Status st = stubs_[wk]->Infer(&cctx, greq, &gresp);

    { std::lock_guard<std::mutex> g(route_mu_); inflight_[wk]--; processed_[wk]++; }

    if (!st.ok()) {
      for (int i = 0; i < n; ++i) {
        Result r; r.ok = false; r.err = st.error_message();
        batch[mb.idx[i]].prom.set_value(std::move(r));
      }
      return;
    }

    // --- OpenMP scatter: split worker output back to per-request results ---
    const float* outv = gresp.output().data();
    const double cms = gresp.compute_ms();
    std::vector<Result> results(n);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
      results[i].output.resize(len[i]);
      std::memcpy(results[i].output.data(), outv + off[i], len[i] * sizeof(float));
      results[i].compute_ms = cms;
      results[i].worker_id = wk;
      results[i].ok = true;
    }
    for (int i = 0; i < n; ++i) {
      results[i].queue_ms =
          std::chrono::duration<double, std::milli>(t_dispatch - batch[mb.idx[i]].arrival).count();
      batch[mb.idx[i]].prom.set_value(std::move(results[i]));
    }

    fprintf(stderr,
            "[dispatcher] worker=%d reqs=%d batch=%ld in=%zuB compute=%.3fms\n",
            wk, n, total_rows, total_floats * sizeof(float), cms);
  }

  std::vector<std::unique_ptr<infer::InferService::Stub>> stubs_;
  std::vector<std::string> worker_addr_;
  std::vector<long> inflight_, processed_;
  std::mutex route_mu_;
  int rr_ = 0;

  std::vector<Pending> queue_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::thread batcher_;
  bool stop_ = false;

  int window_, max_batch_;
  bool least_loaded_;
};

static Dispatcher* g_disp = nullptr;

int main(int argc, char** argv) {
  const int port = arg_int(argc, argv, "--port", 50050);
  const std::string workers_csv =
      arg_str(argc, argv, "--workers", "localhost:50051");
  const int window_ms = arg_int(argc, argv, "--window-ms", 5);
  const int max_batch = arg_int(argc, argv, "--max-batch", 8);
  const bool least_loaded =
      arg_str(argc, argv, "--routing", "ll") != "rr";

  auto workers = split_csv(workers_csv);
  Dispatcher service(workers, window_ms, max_batch, least_loaded);
  g_disp = &service;

  ServerBuilder builder;
  builder.SetMaxReceiveMessageSize(256 * 1024 * 1024);
  builder.SetMaxSendMessageSize(256 * 1024 * 1024);
  const std::string addr = "0.0.0.0:" + std::to_string(port);
  builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<Server> server(builder.BuildAndStart());
  fprintf(stderr, "[dispatcher] listening on %s -> workers [%s]\n",
          addr.c_str(), workers_csv.c_str());
  server->Wait();
  return 0;
}
