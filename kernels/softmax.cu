#include <cuda_runtime.h>
#include "softmax.cuh"

// Phase 0 stub. Real warp-reduction softmax lands in Phase 1.
void softmax_reduction(float*, int, int, int, float, cudaStream_t) {}
