#pragma once
// Hand-written head split/merge transposes for multi-head attention.

// SPLIT: qkv (B, S, 3D) row-major where each (b,s) row is [Q(D)|K(D)|V(D)] and
// each D-block is laid out (H, d_head). Produces q,k,v each in (B*H, S, d_head)
// layout: index ((b*H+h)*S + s)*d_head + e.
void split_heads(const float* qkv, float* q, float* k, float* v,
                 int B, int S, int H, int d_head, cudaStream_t stream = 0);

// MERGE: inverse of split for a single tensor. context (B*H, S, d_head) ->
// out (B, S, D=H*d_head) row-major, out[(b*S+s)*D + h*d_head + e].
void merge_heads(const float* context, float* out,
                 int B, int S, int H, int d_head, cudaStream_t stream = 0);
