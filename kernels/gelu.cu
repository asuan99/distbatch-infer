#include <cuda_runtime.h>
#include <math.h>
#include "gelu.cuh"

// ---------------------------------------------------------------------------
// Fused bias-add + GELU (tanh approximation), FP32, in-place. bias is
// broadcast over rows (length N). Single elementwise kernel (no separate add).
// This is a memory-bound kernel (contrast with compute-bound GEMM in the
// roofline). Hand-written; no library calls.
// ---------------------------------------------------------------------------
__global__ void fused_bias_gelu_kernel(float* x, const float* bias,
                                       int M, int N) {
  const long total = (long)M * N;
  for (long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
       idx < total; idx += (long)blockDim.x * gridDim.x) {
    const int col = idx % N;
    const float v = x[idx] + (bias ? bias[col] : 0.0f);
    const float k0 = 0.7978845608028654f;       // sqrt(2/pi)
    const float inner = k0 * (v + 0.044715f * v * v * v);
    x[idx] = 0.5f * v * (1.0f + tanhf(inner));
  }
}

void fused_bias_gelu(float* dx, const float* dbias, int M, int N,
                     cudaStream_t stream) {
  const int threads = 256;
  const long total = (long)M * N;
  int blocks = (int)((total + threads - 1) / threads);
  if (blocks > 65535) blocks = 65535;           // grid-stride caps the grid
  fused_bias_gelu_kernel<<<blocks, threads, 0, stream>>>(dx, dbias, M, N);
}
