#pragma once
// Hand-written fused bias+GELU (tanh approximation), FP32.

// y[M,N] = GELU(x[M,N] + bias[N]); bias broadcast over rows. In-place on x.
void fused_bias_gelu(float* dx, const float* dbias, int M, int N,
                     cudaStream_t stream = 0);
