#include <cstdio>
#include "gemm.cuh"
#include "softmax.cuh"
#include "gelu.cuh"

// Phase 0 stub: just confirms the kernels library links and CUDA runs.
// Phase 1 fills in CPU-reference comparisons (atol=1e-3) for all 4 kernels.
int main() {
  int dev = -1;
  cudaError_t e = cudaGetDevice(&dev);
  if (e != cudaSuccess) {
    printf("[test_kernels] cudaGetDevice failed: %s\n", cudaGetErrorString(e));
    return 1;
  }
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, dev);
  printf("[test_kernels] scaffold OK on %s (cc=%d.%d)\n", p.name, p.major, p.minor);
  printf("[test_kernels] Phase 0: kernels are stubs; correctness tests added in Phase 1.\n");
  return 0;
}
