#include <cuda_runtime.h>
#include "block.cuh"
#include "gemm.cuh"
#include "softmax.cuh"
#include "gelu.cuh"

// Phase 0 stub. Real block assembly lands in Phase 2.
size_t block_scratch_bytes(const BlockConfig&) { return 0; }

void transformer_block_forward(const float*, float*,
                               const BlockWeights&, const BlockConfig&,
                               BlockScratch&, cudaStream_t) {}
