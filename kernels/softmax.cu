#include <cuda_runtime.h>
#include <math.h>
#include "softmax.cuh"

// ---------------------------------------------------------------------------
// Row-wise softmax over the last dim (cols), one warp per row. Applies `scale`
// before softmax and subtracts the row max for numerical stability. Uses
// warp-level butterfly reductions (__shfl_xor_sync) so every lane gets the
// reduced value. In-place. Hand-written; no library calls.
// ---------------------------------------------------------------------------
__inline__ __device__ float warp_reduce_max(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
  }
  return v;
}

__inline__ __device__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, offset);
  }
  return v;
}

__global__ void softmax_kernel(float* x, int totalRows, int cols, float scale) {
  const int warpId = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warpId >= totalRows) return;

  float* row = x + (size_t)warpId * cols;

  // 1) row max of scaled logits
  float m = -INFINITY;
  for (int c = lane; c < cols; c += 32) {
    m = fmaxf(m, row[c] * scale);
  }
  m = warp_reduce_max(m);

  // 2) exp(scaled - max), write back, accumulate sum
  float s = 0.0f;
  for (int c = lane; c < cols; c += 32) {
    float e = __expf(row[c] * scale - m);
    row[c] = e;
    s += e;
  }
  s = warp_reduce_sum(s);

  // 3) normalize
  const float inv = 1.0f / s;
  for (int c = lane; c < cols; c += 32) {
    row[c] *= inv;
  }
}

void softmax_reduction(float* dx, int batch, int rows, int cols,
                       float scale, cudaStream_t stream) {
  const int totalRows = batch * rows;
  const int threads = 128;            // 4 warps per block
  const int warpsPerBlock = threads / 32;
  const int blocks = (totalRows + warpsPerBlock - 1) / warpsPerBlock;
  softmax_kernel<<<blocks, threads, 0, stream>>>(dx, totalRows, cols, scale);
}
