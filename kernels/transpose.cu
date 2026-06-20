#include <cuda_runtime.h>
#include "transpose.cuh"

// One thread per element of the (B*H, S, d_head) layout. Decode the destination
// index into (b,h,s,e) and copy from the source layout.
__global__ void split_heads_kernel(const float* qkv, float* q, float* k,
                                   float* v, int B, int S, int H, int dh) {
  const long total = (long)B * H * S * dh;
  const int D = H * dh;
  for (long idx = blockIdx.x * (long)blockDim.x + threadIdx.x; idx < total;
       idx += (long)blockDim.x * gridDim.x) {
    const int e = idx % dh;
    long r = idx / dh;          // = (b*H+h)*S + s
    const int s = r % S;
    r /= S;                     // = b*H + h
    const int h = r % H;
    const int b = r / H;

    const long base = (long)(b * S + s) * (3 * D) + h * dh + e;
    q[idx] = qkv[base + 0 * D];
    k[idx] = qkv[base + 1 * D];
    v[idx] = qkv[base + 2 * D];
  }
}

__global__ void merge_heads_kernel(const float* context, float* out,
                                   int B, int S, int H, int dh) {
  const long total = (long)B * H * S * dh;
  const int D = H * dh;
  for (long idx = blockIdx.x * (long)blockDim.x + threadIdx.x; idx < total;
       idx += (long)blockDim.x * gridDim.x) {
    const int e = idx % dh;
    long r = idx / dh;          // = (b*H+h)*S + s
    const int s = r % S;
    r /= S;                     // = b*H + h
    const int h = r % H;
    const int b = r / H;

    const long dst = (long)(b * S + s) * D + h * dh + e;
    out[dst] = context[idx];
  }
}

static inline int grid_for(long total, int threads) {
  long b = (total + threads - 1) / threads;
  return (int)(b > 65535 ? 65535 : b);
}

void split_heads(const float* qkv, float* q, float* k, float* v,
                 int B, int S, int H, int d_head, cudaStream_t stream) {
  const int threads = 256;
  const long total = (long)B * H * S * d_head;
  split_heads_kernel<<<grid_for(total, threads), threads, 0, stream>>>(
      qkv, q, k, v, B, S, H, d_head);
}

void merge_heads(const float* context, float* out,
                 int B, int S, int H, int d_head, cudaStream_t stream) {
  const int threads = 256;
  const long total = (long)B * H * S * d_head;
  merge_heads_kernel<<<grid_for(total, threads), threads, 0, stream>>>(
      context, out, B, S, H, d_head);
}
