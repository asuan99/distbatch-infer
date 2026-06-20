// Micro-benchmark: launches each kernel ONCE at a representative large size so
// ncu profiles exactly three kernels (gemm = compute-bound, gelu = memory-bound,
// softmax = memory/latency-bound) for the roofline.
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "gemm.cuh"
#include "softmax.cuh"
#include "gelu.cuh"

#define CK(c) do{ cudaError_t e=(c); if(e){printf("cuda %s\n",cudaGetErrorString(e));return 1;} }while(0)

int main() {
  // GEMM: 1024^3 (compute-bound)
  {
    int M = 1024, N = 1024, K = 1024;
    float *a, *b, *c;
    CK(cudaMalloc(&a, (size_t)M * K * 4));
    CK(cudaMalloc(&b, (size_t)K * N * 4));
    CK(cudaMalloc(&c, (size_t)M * N * 4));
    gemm_tiled(a, b, c, M, N, K);
    CK(cudaDeviceSynchronize());
    cudaFree(a); cudaFree(b); cudaFree(c);
  }
  // GELU: 4096 x 4096 (memory-bound elementwise)
  {
    int M = 4096, N = 4096;
    float *x, *bias;
    CK(cudaMalloc(&x, (size_t)M * N * 4));
    CK(cudaMalloc(&bias, (size_t)N * 4));
    fused_bias_gelu(x, bias, M, N);
    CK(cudaDeviceSynchronize());
    cudaFree(x); cudaFree(bias);
  }
  // SOFTMAX: 4096 rows x 1024 cols
  {
    int rows = 4096, cols = 1024;
    float* x;
    CK(cudaMalloc(&x, (size_t)rows * cols * 4));
    softmax_reduction(x, 1, rows, cols, 0.125f);
    CK(cudaDeviceSynchronize());
    cudaFree(x);
  }
  printf("[bench] done\n");
  return 0;
}
