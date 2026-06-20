#include <cuda_runtime.h>
#include "gelu.cuh"

// Phase 0 stub. Real fused bias+GELU lands in Phase 1.
void fused_bias_gelu(float*, const float*, int, int, cudaStream_t) {}
