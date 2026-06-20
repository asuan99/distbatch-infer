// Client: verify mode (vs PyTorch reference) and load mode (throughput /
// latency measurement -> CSV). Phases 3 + 5.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <grpcpp/grpcpp.h>
#include "infer.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
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
static bool has_flag(int c, char** v, const char* f) {
  for (int i = 1; i < c; ++i) if (!std::strcmp(v[i], f)) return true;
  return false;
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

static std::shared_ptr<Channel> make_channel(const std::string& target) {
  grpc::ChannelArguments ca;
  ca.SetMaxReceiveMessageSize(256 * 1024 * 1024);
  ca.SetMaxSendMessageSize(256 * 1024 * 1024);
  return grpc::CreateCustomChannel(target, grpc::InsecureChannelCredentials(), ca);
}

// --------------------------------------------------------------------------
static int verify_mode(int argc, char** argv) {
  const std::string target = arg_str(argc, argv, "--target", "localhost:50051");
  const std::string dir = arg_str(argc, argv, "--fixtures", "fixtures");
  int B, S, D, H, ffn;
  { std::ifstream df(dir + "/dims.txt");
    if (!df) { fprintf(stderr, "cannot open %s/dims.txt\n", dir.c_str()); return 3; }
    df >> B >> S >> D >> H >> ffn; }

  std::vector<float> input = read_bin(dir + "/input.bin");
  std::vector<float> ref = read_bin(dir + "/output_ref.bin");

  auto stub = infer::InferService::NewStub(make_channel(target));
  infer::InferRequest req;
  req.set_batch_size(B); req.set_seq_len(S); req.set_hidden_dim(D); req.set_request_id(1);
  auto* in = req.mutable_input();
  in->Resize(input.size(), 0.f);
  std::memcpy(in->mutable_data(), input.data(), input.size() * sizeof(float));

  infer::InferResponse resp;
  ClientContext ctx;
  Status status = stub->Infer(&ctx, req, &resp);
  if (!status.ok()) {
    fprintf(stderr, "RPC failed: %d %s\n", status.error_code(), status.error_message().c_str());
    return 1;
  }
  if ((size_t)resp.output_size() != ref.size()) {
    fprintf(stderr, "output size %d != ref %zu\n", resp.output_size(), ref.size());
    return 1;
  }
  const float ATOL = 1e-2f, RTOL = 1e-2f;
  float max_abs = 0.f, max_rel = 0.f; bool pass = true;
  for (size_t i = 0; i < ref.size(); ++i) {
    float a = std::fabs(resp.output(i) - ref[i]);
    max_abs = std::fmax(max_abs, a);
    max_rel = std::fmax(max_rel, a / (std::fabs(ref[i]) + 1e-12f));
    if (a > ATOL + RTOL * std::fabs(ref[i])) pass = false;
  }
  printf("[client] target=%s B=%d S=%d D=%d worker_id=%d compute_ms=%.3f\n",
         target.c_str(), B, S, D, resp.worker_id(), resp.compute_ms());
  printf("[client] vs PyTorch ref: max_abs=%.3e max_rel=%.3e  %s\n",
         max_abs, max_rel, pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}

// --------------------------------------------------------------------------
struct Sample { double latency_ms, queue_ms, compute_ms; bool ok; };

static int load_mode(int argc, char** argv) {
  const std::string target = arg_str(argc, argv, "--target", "localhost:50050");
  const int requests = arg_int(argc, argv, "--requests", 100);
  const int concurrency = arg_int(argc, argv, "--concurrency", 8);
  const int batch = arg_int(argc, argv, "--batch", 1);
  const int seq_len = arg_int(argc, argv, "--seq_len", 32);
  const int hidden = arg_int(argc, argv, "--hidden_dim", 128);
  const int warmup = arg_int(argc, argv, "--warmup", std::min(concurrency * 2, requests));
  const std::string csv = arg_str(argc, argv, "--csv", "");
  const std::string tag = arg_str(argc, argv, "--tag", "run");

  const size_t io = (size_t)batch * seq_len * hidden;
  // shared input payload (content irrelevant for perf)
  std::vector<float> payload(io);
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> d(-1.f, 1.f);
  for (auto& x : payload) x = d(rng);

  auto channel = make_channel(target);
  auto stub = infer::InferService::NewStub(channel);

  auto one_request = [&](int rid) -> Sample {
    infer::InferRequest req;
    req.set_batch_size(batch); req.set_seq_len(seq_len);
    req.set_hidden_dim(hidden); req.set_request_id(rid);
    auto* in = req.mutable_input();
    in->Resize(io, 0.f);
    std::memcpy(in->mutable_data(), payload.data(), io * sizeof(float));

    infer::InferResponse resp;
    ClientContext ctx;
    auto t0 = Clock::now();
    Status st = stub->Infer(&ctx, req, &resp);
    auto t1 = Clock::now();
    Sample s;
    s.ok = st.ok();
    s.latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    s.queue_ms = st.ok() ? resp.queue_ms() : 0;
    s.compute_ms = st.ok() ? resp.compute_ms() : 0;
    return s;
  };

  // warmup (not measured)
  for (int i = 0; i < warmup; ++i) one_request(-1);

  std::vector<Sample> samples(requests);
  std::atomic<int> next{0};
  auto worker = [&]() {
    int i;
    while ((i = next.fetch_add(1)) < requests) samples[i] = one_request(i);
  };

  auto wall0 = Clock::now();
  std::vector<std::thread> th;
  for (int c = 0; c < concurrency; ++c) th.emplace_back(worker);
  for (auto& t : th) t.join();
  auto wall1 = Clock::now();
  double wall_ms = std::chrono::duration<double, std::milli>(wall1 - wall0).count();

  // aggregate
  std::vector<double> lat;
  double sum_lat = 0, sum_q = 0, sum_c = 0;
  int ok = 0;
  for (auto& s : samples) {
    if (!s.ok) continue;
    ok++;
    lat.push_back(s.latency_ms);
    sum_lat += s.latency_ms; sum_q += s.queue_ms; sum_c += s.compute_ms;
  }
  std::sort(lat.begin(), lat.end());
  auto pct = [&](double p) {
    if (lat.empty()) return 0.0;
    size_t idx = (size_t)(p * (lat.size() - 1));
    return lat[idx];
  };
  double lat_mean = ok ? sum_lat / ok : 0;
  double q_mean = ok ? sum_q / ok : 0;
  double c_mean = ok ? sum_c / ok : 0;
  double other_mean = lat_mean - q_mean - c_mean;  // serialize + transport + H2D/D2H
  double throughput = ok / (wall_ms / 1000.0);
  size_t req_bytes = io * sizeof(float);

  printf("[load] tag=%s target=%s req=%d/%d conc=%d B=%d S=%d D=%d\n",
         tag.c_str(), target.c_str(), ok, requests, concurrency, batch, seq_len, hidden);
  printf("[load] throughput=%.1f req/s  wall=%.1fms  lat mean=%.3f p50=%.3f p99=%.3f ms\n",
         throughput, wall_ms, lat_mean, pct(0.50), pct(0.99));
  printf("[load] breakdown(ms): queue=%.3f compute=%.3f other=%.3f  req_bytes=%zu\n",
         q_mean, c_mean, other_mean, req_bytes);

  if (!csv.empty()) {
    bool exists = std::ifstream(csv).good();
    std::ofstream f(csv, std::ios::app);
    if (!exists)
      f << "tag,target,requests_ok,requests,concurrency,batch,seq_len,hidden_dim,"
           "wall_ms,throughput_rps,lat_mean_ms,lat_p50_ms,lat_p99_ms,"
           "queue_mean_ms,compute_mean_ms,other_mean_ms,req_bytes\n";
    f << tag << "," << target << "," << ok << "," << requests << "," << concurrency
      << "," << batch << "," << seq_len << "," << hidden << "," << wall_ms << ","
      << throughput << "," << lat_mean << "," << pct(0.50) << "," << pct(0.99) << ","
      << q_mean << "," << c_mean << "," << other_mean << "," << req_bytes << "\n";
    fprintf(stderr, "[load] appended to %s\n", csv.c_str());
  }
  return ok == requests ? 0 : 1;
}

int main(int argc, char** argv) {
  // load mode if --requests given, else verify mode
  if (has_flag(argc, argv, "--requests")) return load_mode(argc, argv);
  return verify_mode(argc, argv);
}
