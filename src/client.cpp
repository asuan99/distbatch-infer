// Client: Phase 3 verify mode (single request vs PyTorch reference).
// Load-generation / CSV measurement is added in Phase 5.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include <grpcpp/grpcpp.h>
#include "infer.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

static std::string arg_str(int argc, char** argv, const char* flag,
                           const std::string& def) {
  for (int i = 1; i < argc - 1; ++i)
    if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
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

int main(int argc, char** argv) {
  const std::string target = arg_str(argc, argv, "--target", "localhost:50051");
  const std::string dir = arg_str(argc, argv, "--fixtures", "fixtures");

  int B, S, D, H, ffn;
  {
    std::ifstream df(dir + "/dims.txt");
    if (!df) { fprintf(stderr, "cannot open %s/dims.txt\n", dir.c_str()); return 3; }
    df >> B >> S >> D >> H >> ffn;
  }

  std::vector<float> input = read_bin(dir + "/input.bin");
  std::vector<float> ref = read_bin(dir + "/output_ref.bin");

  grpc::ChannelArguments ch_args;
  ch_args.SetMaxReceiveMessageSize(256 * 1024 * 1024);
  ch_args.SetMaxSendMessageSize(256 * 1024 * 1024);
  auto channel = grpc::CreateCustomChannel(
      target, grpc::InsecureChannelCredentials(), ch_args);
  auto stub = infer::InferService::NewStub(channel);

  infer::InferRequest req;
  req.set_batch_size(B);
  req.set_seq_len(S);
  req.set_hidden_dim(D);
  req.set_request_id(1);
  auto* in = req.mutable_input();
  in->Resize(input.size(), 0.f);
  std::memcpy(in->mutable_data(), input.data(), input.size() * sizeof(float));

  infer::InferResponse resp;
  ClientContext ctx;
  Status status = stub->Infer(&ctx, req, &resp);
  if (!status.ok()) {
    fprintf(stderr, "RPC failed: %d %s\n", status.error_code(),
            status.error_message().c_str());
    return 1;
  }

  if ((size_t)resp.output_size() != ref.size()) {
    fprintf(stderr, "output size %d != ref %zu\n", resp.output_size(), ref.size());
    return 1;
  }

  const float ATOL = 1e-2f, RTOL = 1e-2f;
  float max_abs = 0.f, max_rel = 0.f;
  bool pass = true;
  for (size_t i = 0; i < ref.size(); ++i) {
    float a = std::fabs(resp.output(i) - ref[i]);
    float r = a / (std::fabs(ref[i]) + 1e-12f);
    max_abs = std::fmax(max_abs, a);
    max_rel = std::fmax(max_rel, r);
    if (a > ATOL + RTOL * std::fabs(ref[i])) pass = false;
  }

  printf("[client] target=%s  B=%d S=%d D=%d  worker_id=%d  compute_ms=%.3f\n",
         target.c_str(), B, S, D, resp.worker_id(), resp.compute_ms());
  printf("[client] vs PyTorch ref: max_abs=%.3e max_rel=%.3e  %s\n",
         max_abs, max_rel, pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
