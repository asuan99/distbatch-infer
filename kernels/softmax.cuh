#pragma once
// Hand-written row-wise softmax with warp-level reduction (FP32).

// x: (batch*rows, cols) viewed row-major; softmax over last dim (cols).
// Applies `scale` (e.g. 1/sqrt(d_head)) before softmax. In-place on x.
void softmax_reduction(float* dx, int batch, int rows, int cols,
                       float scale, cudaStream_t stream = 0);
