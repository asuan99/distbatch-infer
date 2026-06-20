#!/usr/bin/env python3
"""PyTorch reference for the transformer block (reference/verification only).

Generates a single weights.bin + input.bin (seeded), runs an explicit-matmul
forward that mirrors the CUDA dataflow 1:1, and dumps output_ref.bin plus
intermediates (qkv/scores/context) for stage-by-stage debugging.

IMPORTANT: uses explicit `x @ W` (row-major), NOT nn.Linear (which is x @ W^T),
so the weight layout matches the hand-written CUDA GEMM exactly.
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F


def get_arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main():
    out = get_arg("--out", "fixtures")
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)

    B, S, D, H = 2, 32, 128, 8
    dh = D // H
    ffn = 4 * D
    scale = 1.0 / (dh ** 0.5)

    def rnd(*shape):
        return torch.rand(*shape, dtype=torch.float64) * 2 - 1  # uniform [-1,1]

    # generate in float64 then cast to float32 for storage; forward in float32.
    x = rnd(B, S, D)
    Wqkv, bqkv = rnd(D, 3 * D), rnd(3 * D)
    Wo, bo = rnd(D, D), rnd(D)
    W1, b1 = rnd(D, ffn), rnd(ffn)
    W2, b2 = rnd(ffn, D), rnd(D)

    def f32(t):
        return t.to(torch.float32)

    xf = f32(x).reshape(B * S, D)
    qkv = xf @ f32(Wqkv) + f32(bqkv)                       # (B*S, 3D)
    qkv3 = qkv.reshape(B, S, 3 * D)
    Q = qkv3[..., 0:D].reshape(B, S, H, dh).permute(0, 2, 1, 3).reshape(B * H, S, dh)
    K = qkv3[..., D:2 * D].reshape(B, S, H, dh).permute(0, 2, 1, 3).reshape(B * H, S, dh)
    V = qkv3[..., 2 * D:3 * D].reshape(B, S, H, dh).permute(0, 2, 1, 3).reshape(B * H, S, dh)

    scores = torch.bmm(Q, K.transpose(1, 2))              # raw QK^T (B*H,S,S)
    attn = torch.softmax(scores * scale, dim=-1)          # post-softmax probs
    context = torch.bmm(attn, V)                          # (B*H,S,dh)
    attn_out = context.reshape(B, H, S, dh).permute(0, 2, 1, 3).reshape(B * S, D)

    proj = attn_out @ f32(Wo) + f32(bo)
    ffn1 = F.gelu(proj @ f32(W1) + f32(b1), approximate="tanh")
    out_t = ffn1 @ f32(W2) + f32(b2)
    output = out_t.reshape(B, S, D)

    def save(name, t):
        t.contiguous().to(torch.float32).numpy().astype(np.float32).tofile(
            os.path.join(out, name))

    # weights.bin in fixed order: Wqkv,bqkv,Wo,bo,W1,b1,W2,b2
    with open(os.path.join(out, "weights.bin"), "wb") as fh:
        for t in [Wqkv, bqkv, Wo, bo, W1, b1, W2, b2]:
            fh.write(t.contiguous().to(torch.float32).numpy().astype(np.float32).tobytes())

    save("input.bin", x)
    save("output_ref.bin", output)
    save("qkv_ref.bin", qkv)          # matches scratch.qkv
    save("scores_ref.bin", attn)      # matches scratch.scores (post-softmax)
    save("context_ref.bin", context)  # matches scratch.context

    with open(os.path.join(out, "dims.txt"), "w") as fh:
        fh.write(f"{B} {S} {D} {H} {ffn}\n")

    print(f"[ref_block] wrote fixtures to {out}/ "
          f"(B={B} S={S} D={D} H={H} dh={dh} ffn={ffn})")


if __name__ == "__main__":
    main()
