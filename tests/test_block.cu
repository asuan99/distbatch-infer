#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>

#include "block.cuh"

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t _e = (call);                                                 \
    if (_e != cudaSuccess) {                                                 \
      printf("CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__,   \
             __LINE__);                                                      \
      std::exit(2);                                                          \
    }                                                                        \
  } while (0)

static const float ATOL = 1e-2f;
static const float RTOL = 1e-2f;

static std::vector<float> read_bin(const std::string& path, size_t count) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { printf("cannot open %s\n", path.c_str()); std::exit(3); }
  std::vector<float> v(count);
  f.read(reinterpret_cast<char*>(v.data()), count * sizeof(float));
  if ((size_t)f.gcount() != count * sizeof(float)) {
    printf("short read on %s (got %zd, want %zu)\n", path.c_str(),
           (ssize_t)f.gcount(), count * sizeof(float));
    std::exit(3);
  }
  return v;
}

static float* to_dev(const std::vector<float>& h) {
  float* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  return d;
}

static float* dev_get(float* d, std::vector<float>& host) {
  CUDA_CHECK(cudaMemcpy(host.data(), d, host.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  return d;
}

// returns true if within tolerance; prints max abs/rel error
static bool compare(const char* tag, const std::vector<float>& got,
                    const std::vector<float>& ref) {
  float max_abs = 0.f, max_rel = 0.f;
  bool pass = true;
  for (size_t i = 0; i < got.size(); ++i) {
    float a = std::fabs(got[i] - ref[i]);
    float r = a / (std::fabs(ref[i]) + 1e-12f);
    max_abs = std::fmax(max_abs, a);
    max_rel = std::fmax(max_rel, r);
    if (a > ATOL + RTOL * std::fabs(ref[i])) pass = false;
  }
  printf("  %-10s max_abs=%.3e  max_rel=%.3e  %s\n", tag, max_abs, max_rel,
         pass ? "OK" : "FAIL");
  return pass;
}

int main(int argc, char** argv) {
  std::string dir = (argc > 1) ? argv[1] : "fixtures";

  // dims.txt: B S D H ffn
  int B, S, D, H, ffn;
  {
    std::ifstream f(dir + "/dims.txt");
    if (!f) { printf("cannot open %s/dims.txt (run tests/ref_block.py first)\n",
                     dir.c_str()); return 3; }
    f >> B >> S >> D >> H >> ffn;
  }
  const int dh = D / H;
  BlockConfig cfg{B, S, D, H, dh, ffn};
  printf("[test_block] dir=%s  B=%d S=%d D=%d H=%d dh=%d ffn=%d  ATOL=%.0e RTOL=%.0e\n",
         dir.c_str(), B, S, D, H, dh, ffn, ATOL, RTOL);

  const size_t BS = (size_t)B * S;
  // weights.bin sizes
  std::vector<float> wbuf = read_bin(dir + "/weights.bin",
      (size_t)D * 3 * D + 3 * D + (size_t)D * D + D +
      (size_t)D * ffn + ffn + (size_t)ffn * D + D);

  size_t off = 0;
  auto slice = [&](size_t n) { float* p = wbuf.data() + off; off += n; return std::vector<float>(p, p + n); };
  std::vector<float> hWqkv = slice((size_t)D * 3 * D);
  std::vector<float> hbqkv = slice(3 * D);
  std::vector<float> hWo   = slice((size_t)D * D);
  std::vector<float> hbo   = slice(D);
  std::vector<float> hW1   = slice((size_t)D * ffn);
  std::vector<float> hb1   = slice(ffn);
  std::vector<float> hW2   = slice((size_t)ffn * D);
  std::vector<float> hb2   = slice(D);

  BlockWeights w;
  w.Wqkv = to_dev(hWqkv); w.bqkv = to_dev(hbqkv);
  w.Wo = to_dev(hWo);     w.bo = to_dev(hbo);
  w.W1 = to_dev(hW1);     w.b1 = to_dev(hb1);
  w.W2 = to_dev(hW2);     w.b2 = to_dev(hb2);

  std::vector<float> hx = read_bin(dir + "/input.bin", BS * D);
  float* dx = to_dev(hx);
  float* dout = nullptr;
  CUDA_CHECK(cudaMalloc(&dout, BS * D * sizeof(float)));

  void* base = nullptr;
  CUDA_CHECK(cudaMalloc(&base, block_scratch_bytes(cfg)));
  BlockScratch s;
  block_scratch_partition(s, base, cfg);

  transformer_block_forward(dx, dout, w, cfg, s, 0);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  bool ok = true;

  // intermediates
  std::vector<float> qkv(BS * 3 * D), ref_qkv = read_bin(dir + "/qkv_ref.bin", BS * 3 * D);
  dev_get(s.qkv, qkv);
  ok &= compare("qkv", qkv, ref_qkv);

  std::vector<float> scores((size_t)B * H * S * S),
      ref_scores = read_bin(dir + "/scores_ref.bin", (size_t)B * H * S * S);
  dev_get(s.scores, scores);
  ok &= compare("scores", scores, ref_scores);

  std::vector<float> context(BS * D), ref_context = read_bin(dir + "/context_ref.bin", BS * D);
  dev_get(s.context, context);
  ok &= compare("context", context, ref_context);

  std::vector<float> out(BS * D), ref_out = read_bin(dir + "/output_ref.bin", BS * D);
  dev_get(dout, out);
  ok &= compare("output", out, ref_out);

  printf("\n[test_block] %s\n", ok ? "PASSED" : "FAILED");
  return ok ? 0 : 1;
}
