#include <cuda_runtime.h>
#include "gemm.cuh"

// ---------------------------------------------------------------------------
// Tiled FP32 GEMM, shared-memory, TILE x TILE per block. Hand-written; no
// cuBLAS/CUTLASS. Handles dimensions that are not multiples of TILE via
// boundary-guarded loads (out-of-range -> 0).
// ---------------------------------------------------------------------------
#define TILE GEMM_TILE

__device__ __forceinline__ void tiled_gemm_device(
    const float* A, const float* B, float* C, int M, int N, int K) {
  // +1 padding on the inner dim avoids shared-memory bank conflicts.
  __shared__ float As[TILE][TILE + 1];
  __shared__ float Bs[TILE][TILE + 1];

  const int ty = threadIdx.y;
  const int tx = threadIdx.x;
  const int row = blockIdx.y * TILE + ty;
  const int col = blockIdx.x * TILE + tx;

  float acc = 0.0f;
  const int numTiles = (K + TILE - 1) / TILE;

  for (int t = 0; t < numTiles; ++t) {
    const int aCol = t * TILE + tx;
    const int bRow = t * TILE + ty;
    As[ty][tx] = (row < M && aCol < K) ? A[(size_t)row * K + aCol] : 0.0f;
    Bs[ty][tx] = (bRow < K && col < N) ? B[(size_t)bRow * N + col] : 0.0f;
    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE; ++k) {
      acc += As[ty][k] * Bs[k][tx];
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    C[(size_t)row * N + col] = acc;
  }
}

__global__ void gemm_tiled_kernel(const float* A, const float* B, float* C,
                                  int M, int N, int K) {
  tiled_gemm_device(A, B, C, M, N, K);
}

__global__ void batched_gemm_kernel(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
  const int b = blockIdx.z;
  tiled_gemm_device(A + (size_t)b * M * K,
                    B + (size_t)b * K * N,
                    C + (size_t)b * M * N, M, N, K);
}

// C[M,N] = A[M,K] @ B[N,K]^T (B stored row-major as N x K). Tiled; same shared
// loads as the NN variant except the B tile is gathered transposed.
__device__ __forceinline__ void tiled_gemm_nt_device(
    const float* A, const float* B, float* C, int M, int N, int K) {
  __shared__ float As[TILE][TILE + 1];
  __shared__ float Bs[TILE][TILE + 1];

  const int ty = threadIdx.y;
  const int tx = threadIdx.x;
  const int row = blockIdx.y * TILE + ty;   // M index
  const int col = blockIdx.x * TILE + tx;   // N index

  float acc = 0.0f;
  const int numTiles = (K + TILE - 1) / TILE;

  for (int t = 0; t < numTiles; ++t) {
    const int aK = t * TILE + tx;           // k for A[row, k]
    const int bN = blockIdx.x * TILE + tx;  // N index for B[bN, bK]
    const int bK = t * TILE + ty;           // k for B[bN, k]
    As[ty][tx] = (row < M && aK < K) ? A[(size_t)row * K + aK] : 0.0f;
    Bs[ty][tx] = (bN < N && bK < K) ? B[(size_t)bN * K + bK] : 0.0f;
    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE; ++k) {
      acc += As[ty][k] * Bs[k][tx];
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    C[(size_t)row * N + col] = acc;
  }
}

__global__ void batched_gemm_nt_kernel(const float* A, const float* B, float* C,
                                       int M, int N, int K) {
  const int b = blockIdx.z;
  tiled_gemm_nt_device(A + (size_t)b * M * K,
                       B + (size_t)b * N * K,
                       C + (size_t)b * M * N, M, N, K);
}

// --- host launchers ---------------------------------------------------------
void gemm_tiled(const float* dA, const float* dB, float* dC,
                int M, int N, int K, cudaStream_t stream) {
  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  gemm_tiled_kernel<<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K);
}

void batched_gemm(const float* dA, const float* dB, float* dC,
                  int batch, int M, int N, int K, cudaStream_t stream) {
  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, batch);
  batched_gemm_kernel<<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K);
}

void batched_gemm_nt(const float* dA, const float* dB, float* dC,
                     int batch, int M, int N, int K, cudaStream_t stream) {
  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, batch);
  batched_gemm_nt_kernel<<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K);
}
