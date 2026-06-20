#include <cuda_runtime.h>
#include <math.h>
#include "block.cuh"
#include "gemm.cuh"
#include "softmax.cuh"
#include "gelu.cuh"
#include "transpose.cuh"

// ---------------------------------------------------------------------------
// Small hand-written bias-add (broadcast over rows). Used for qkv/out-proj/
// ffn-down biases (FFN-up bias is folded into fused_bias_gelu).
// ---------------------------------------------------------------------------
__global__ void add_bias_kernel(float* x, const float* bias, int M, int N) {
  const long total = (long)M * N;
  for (long idx = blockIdx.x * (long)blockDim.x + threadIdx.x; idx < total;
       idx += (long)blockDim.x * gridDim.x) {
    x[idx] += bias[idx % N];
  }
}

static void add_bias(float* dx, const float* dbias, int M, int N,
                     cudaStream_t stream) {
  if (!dbias) return;
  const int threads = 256;
  long blocks = ((long)M * N + threads - 1) / threads;
  if (blocks > 65535) blocks = 65535;
  add_bias_kernel<<<(int)blocks, threads, 0, stream>>>(dx, dbias, M, N);
}

// ---------------------------------------------------------------------------
// Scratch layout (single allocation, in this order).
// ---------------------------------------------------------------------------
namespace {
struct Sizes {
  size_t qkv, q, k, v, scores, context, attn_out, proj, ffn1, total;
};
Sizes scratch_sizes(const BlockConfig& c) {
  const size_t BS = (size_t)c.B * c.S;
  const size_t BH = (size_t)c.B * c.H;
  Sizes z;
  z.qkv      = BS * (3 * c.D);
  z.q        = BS * c.D;          // == BH * S * d_head
  z.k        = BS * c.D;
  z.v        = BS * c.D;
  z.scores   = BH * (size_t)c.S * c.S;
  z.context  = BS * c.D;
  z.attn_out = BS * c.D;
  z.proj     = BS * c.D;
  z.ffn1     = BS * (size_t)c.ffn;
  z.total = z.qkv + z.q + z.k + z.v + z.scores + z.context + z.attn_out +
            z.proj + z.ffn1;
  return z;
}
}  // namespace

size_t block_scratch_bytes(const BlockConfig& cfg) {
  return scratch_sizes(cfg).total * sizeof(float);
}

void block_scratch_partition(BlockScratch& s, void* base,
                             const BlockConfig& cfg) {
  Sizes z = scratch_sizes(cfg);
  float* p = reinterpret_cast<float*>(base);
  s.qkv = p;      p += z.qkv;
  s.q = p;        p += z.q;
  s.k = p;        p += z.k;
  s.v = p;        p += z.v;
  s.scores = p;   p += z.scores;
  s.context = p;  p += z.context;
  s.attn_out = p; p += z.attn_out;
  s.proj = p;     p += z.proj;
  s.ffn1 = p;     p += z.ffn1;
}

// ---------------------------------------------------------------------------
// Forward (no LayerNorm / residual / dropout). All ops are hand-written
// kernels; no library calls on this path.
// ---------------------------------------------------------------------------
void transformer_block_forward(const float* dx, float* dout,
                               const BlockWeights& w, const BlockConfig& cfg,
                               BlockScratch& s, cudaStream_t stream) {
  const int B = cfg.B, S = cfg.S, D = cfg.D, H = cfg.H;
  const int dh = cfg.d_head, ffn = cfg.ffn;
  const int BS = B * S;
  const int BH = B * H;
  const float scale = 1.0f / sqrtf((float)dh);

  // 1) QKV projection: (B*S,D) @ Wqkv(D,3D) -> qkv (+bqkv)
  gemm_tiled(dx, w.Wqkv, s.qkv, BS, 3 * D, D, stream);
  add_bias(s.qkv, w.bqkv, BS, 3 * D, stream);

  // 2) split heads -> q,k,v (B*H, S, d_head)
  split_heads(s.qkv, s.q, s.k, s.v, B, S, H, dh, stream);

  // 3) scores = Q @ K^T  (B*H, S, S)
  batched_gemm_nt(s.q, s.k, s.scores, BH, S, S, dh, stream);

  // 4) softmax(scale) in-place over last dim
  softmax_reduction(s.scores, BH, S, S, scale, stream);

  // 5) context = scores @ V  (B*H, S, d_head)
  batched_gemm(s.scores, s.v, s.context, BH, S, dh, S, stream);

  // 6) merge heads -> attn_out (B*S, D)
  merge_heads(s.context, s.attn_out, B, S, H, dh, stream);

  // 7) output projection: (B*S,D) @ Wo(D,D) -> proj (+bo)
  gemm_tiled(s.attn_out, w.Wo, s.proj, BS, D, D, stream);
  add_bias(s.proj, w.bo, BS, D, stream);

  // 8) FFN up: (B*S,D) @ W1(D,4D) -> ffn1
  gemm_tiled(s.proj, w.W1, s.ffn1, BS, ffn, D, stream);

  // 9) fused bias + GELU
  fused_bias_gelu(s.ffn1, w.b1, BS, ffn, stream);

  // 10) FFN down: (B*S,4D) @ W2(4D,D) -> dout (+b2)
  gemm_tiled(s.ffn1, w.W2, dout, BS, D, ffn, stream);
  add_bias(dout, w.b2, BS, D, stream);
}
