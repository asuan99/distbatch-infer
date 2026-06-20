#include <cuda_runtime.h>
#include "gemm.cuh"

// Phase 0 stub. Real tiled implementation lands in Phase 1.
void gemm_tiled(const float*, const float*, float*,
                int, int, int, cudaStream_t) {}

void batched_gemm(const float*, const float*, float*,
                  int, int, int, int, cudaStream_t) {}
