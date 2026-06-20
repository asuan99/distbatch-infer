#pragma once
// Hand-written tiled GEMM kernels (FP32). No cuBLAS/CUTLASS.

#ifndef GEMM_TILE
#define GEMM_TILE 32
#endif

// C[M,N] = A[M,K] @ B[K,N], row-major, FP32.
void gemm_tiled(const float* dA, const float* dB, float* dC,
                int M, int N, int K, cudaStream_t stream = 0);

// Batched: C[b] = A[b] @ B[b] for b in [0,batch). Each matrix row-major,
// contiguous: A is (batch,M,K), B is (batch,K,N), C is (batch,M,N).
void batched_gemm(const float* dA, const float* dB, float* dC,
                  int batch, int M, int N, int K, cudaStream_t stream = 0);
