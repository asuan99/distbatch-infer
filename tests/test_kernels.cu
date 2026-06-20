#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_runtime.h>

#include "gemm.cuh"
#include "softmax.cuh"
#include "gelu.cuh"

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t _e = (call);                                                  \
    if (_e != cudaSuccess) {                                                  \
      printf("CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__,    \
             __LINE__);                                                       \
      std::exit(2);                                                           \
    }                                                                         \
  } while (0)

static const float ATOL = 1e-3f;

static std::mt19937 rng(1234);
static void fill_uniform(std::vector<float>& v, float lo = -1.f, float hi = 1.f) {
  std::uniform_real_distribution<float> d(lo, hi);
  for (auto& x : v) x = d(rng);
}

// max abs diff between gpu (float) and ref (double)
static float max_abs_diff(const std::vector<float>& gpu,
                          const std::vector<double>& ref) {
  float m = 0.f;
  for (size_t i = 0; i < gpu.size(); ++i)
    m = std::fmax(m, std::fabs(gpu[i] - (float)ref[i]));
  return m;
}

static float* dev(const std::vector<float>& h) {
  float* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  return d;
}

// ---------------------------------------------------------------------------
static bool test_gemm(int M, int N, int K) {
  std::vector<float> A(M * K), B(K * N), C(M * N, 0.f);
  fill_uniform(A);
  fill_uniform(B);

  float *dA = dev(A), *dB = dev(B), *dC = dev(C);
  gemm_tiled(dA, dB, dC, M, N, K);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(C.data(), dC, C.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  std::vector<double> ref(M * N);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k) acc += (double)A[i * K + k] * B[k * N + j];
      ref[i * N + j] = acc;
    }

  float err = max_abs_diff(C, ref);
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  printf("  gemm_tiled   M=%-4d N=%-4d K=%-4d  max_abs_err=%.2e  %s\n", M, N, K,
         err, err < ATOL ? "OK" : "FAIL");
  return err < ATOL;
}

// ---------------------------------------------------------------------------
static bool test_batched_gemm(int batch, int M, int N, int K) {
  std::vector<float> A(batch * M * K), B(batch * K * N), C(batch * M * N, 0.f);
  fill_uniform(A);
  fill_uniform(B);

  float *dA = dev(A), *dB = dev(B), *dC = dev(C);
  batched_gemm(dA, dB, dC, batch, M, N, K);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(C.data(), dC, C.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  std::vector<double> ref(batch * M * N);
  for (int b = 0; b < batch; ++b)
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j) {
        double acc = 0.0;
        const float* Ab = &A[(size_t)b * M * K];
        const float* Bb = &B[(size_t)b * K * N];
        for (int k = 0; k < K; ++k) acc += (double)Ab[i * K + k] * Bb[k * N + j];
        ref[(size_t)b * M * N + i * N + j] = acc;
      }

  float err = max_abs_diff(C, ref);
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  printf("  batched_gemm b=%-3d M=%-4d N=%-4d K=%-4d  max_abs_err=%.2e  %s\n",
         batch, M, N, K, err, err < ATOL ? "OK" : "FAIL");
  return err < ATOL;
}

// ---------------------------------------------------------------------------
static bool test_softmax(int batch, int rows, int cols, float scale) {
  std::vector<float> X(batch * rows * cols);
  fill_uniform(X, -3.f, 3.f);

  float* dX = dev(X);
  softmax_reduction(dX, batch, rows, cols, scale);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> out(X.size());
  CUDA_CHECK(cudaMemcpy(out.data(), dX, out.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  std::vector<double> ref(X.size());
  const int totalRows = batch * rows;
  for (int r = 0; r < totalRows; ++r) {
    const float* xr = &X[(size_t)r * cols];
    double mx = -1e300;
    for (int c = 0; c < cols; ++c) mx = std::fmax(mx, (double)xr[c] * scale);
    double sum = 0.0;
    for (int c = 0; c < cols; ++c) sum += std::exp((double)xr[c] * scale - mx);
    for (int c = 0; c < cols; ++c)
      ref[(size_t)r * cols + c] = std::exp((double)xr[c] * scale - mx) / sum;
  }

  float err = max_abs_diff(out, ref);
  cudaFree(dX);
  printf("  softmax      b=%-3d rows=%-4d cols=%-4d scale=%.3f  max_abs_err=%.2e  %s\n",
         batch, rows, cols, scale, err, err < ATOL ? "OK" : "FAIL");
  return err < ATOL;
}

// ---------------------------------------------------------------------------
static bool test_gelu(int M, int N) {
  std::vector<float> X(M * N), bias(N);
  fill_uniform(X, -4.f, 4.f);
  fill_uniform(bias, -1.f, 1.f);

  float* dX = dev(X);
  float* dB = dev(bias);
  fused_bias_gelu(dX, dB, M, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> out(X.size());
  CUDA_CHECK(cudaMemcpy(out.data(), dX, out.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  std::vector<double> ref(X.size());
  const double k0 = 0.7978845608028654;  // sqrt(2/pi)
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double v = (double)X[i * N + j] + (double)bias[j];
      double inner = k0 * (v + 0.044715 * v * v * v);
      ref[i * N + j] = 0.5 * v * (1.0 + std::tanh(inner));
    }

  float err = max_abs_diff(out, ref);
  cudaFree(dX); cudaFree(dB);
  printf("  fused_bias_gelu M=%-4d N=%-4d  max_abs_err=%.2e  %s\n", M, N, err,
         err < ATOL ? "OK" : "FAIL");
  return err < ATOL;
}

// ---------------------------------------------------------------------------
int main() {
  int dev_id = -1;
  CUDA_CHECK(cudaGetDevice(&dev_id));
  cudaDeviceProp p;
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev_id));
  printf("[test_kernels] %s (cc=%d.%d), ATOL=%.0e\n", p.name, p.major, p.minor,
         ATOL);

  bool ok = true;

  printf("[gemm_tiled]\n");
  ok &= test_gemm(32, 32, 32);       // exact tile
  ok &= test_gemm(64, 128, 256);     // multiples
  ok &= test_gemm(67, 53, 91);       // non-multiples (boundary)
  ok &= test_gemm(768, 768, 768);    // model-sized

  printf("[batched_gemm]\n");
  ok &= test_batched_gemm(4, 32, 32, 32);
  ok &= test_batched_gemm(12, 64, 64, 64);   // B*H heads, S=64, d_head=64
  ok &= test_batched_gemm(8, 50, 70, 33);    // non-multiples

  printf("[softmax_reduction]\n");
  ok &= test_softmax(1, 8, 32, 1.0f);
  ok &= test_softmax(2, 16, 128, 0.125f);    // scale = 1/sqrt(64)
  ok &= test_softmax(4, 64, 1000, 0.125f);   // wide rows, non-mult of 32

  printf("[fused_bias_gelu]\n");
  ok &= test_gelu(32, 64);
  ok &= test_gelu(128, 3072);                // FFN-sized (4*768)
  ok &= test_gelu(100, 50);                  // non-mult

  printf("\n[test_kernels] %s\n", ok ? "ALL PASSED" : "FAILURES PRESENT");
  return ok ? 0 : 1;
}
