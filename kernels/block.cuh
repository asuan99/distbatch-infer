#pragma once
// Transformer block forward, assembled from hand-written kernels.

// Weights for one transformer block (row-major FP32, device pointers).
struct BlockWeights {
  const float* Wqkv;   // (D, 3D)
  const float* bqkv;   // (3D)   -- optional bias on qkv (may be null)
  const float* Wo;     // (D, D)
  const float* bo;     // (D)    -- optional (may be null)
  const float* W1;     // (D, 4D) FFN up
  const float* b1;     // (4D)
  const float* W2;     // (4D, D) FFN down
  const float* b2;     // (D)     -- optional (may be null)
};

struct BlockConfig {
  int B;        // batch
  int S;        // seq len
  int D;        // hidden dim
  int H;        // num heads
  int d_head;   // D / H
  int ffn;      // FFN hidden (default 4*D)
};

// Scratch device buffers needed for the forward pass. Caller allocates.
struct BlockScratch {
  float* qkv;      // (B*S, 3D)
  float* q;        // (B*H, S, d_head)
  float* k;        // (B*H, S, d_head)
  float* v;        // (B*H, S, d_head)
  float* scores;   // (B*H, S, S)
  float* context;  // (B*H, S, d_head)
  float* attn_out; // (B*S, D)  context merged back to (B,S,D)
  float* proj;     // (B*S, D)  output projection result
  float* ffn1;     // (B*S, 4D)
};

size_t block_scratch_bytes(const BlockConfig& cfg);

// Partition a single device allocation of `block_scratch_bytes(cfg)` bytes into
// the individual scratch buffers. `base` must point to that allocation.
void block_scratch_partition(BlockScratch& s, void* base, const BlockConfig& cfg);

// x: (B, S, D) device pointer (input). out: (B, S, D) device pointer.
void transformer_block_forward(const float* dx, float* dout,
                               const BlockWeights& w, const BlockConfig& cfg,
                               BlockScratch& scratch, cudaStream_t stream = 0);
