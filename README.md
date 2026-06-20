# distbatch-infer — Distributed Batch Inference Engine (CUDA + OpenMP + gRPC)

A Transformer block forward pass implemented with **hand-written CUDA kernels**
(no cuBLAS/cuDNN/CUTLASS/Thrust on the serving path), served through a
**gRPC worker pool** with an **OpenMP dispatcher** that assembles micro-batches.

```
 [Client] --gRPC(protobuf)--> [Dispatcher (OpenMP batch assembly/routing)]
                                       |  round-robin / least-loaded
                                       v
                              [Worker pool] (each worker = process/port, 1 CUDA stream)
                                       |  Transformer block forward (3 hand-written kernels)
                                       v
                              results --> Client aggregation (latency/throughput CSV)
```

## Hard constraints
- All GEMM / softmax / GELU are hand-written `__global__` kernels in `kernels/*.cu`.
- PyTorch / NumPy are used **only** for reference correctness and plotting — never on the serving path.

## Build
```bash
cmake -B build -S .            # add -DCUDA_ARCH=native if sm_120 is rejected
cmake --build build -j
ctest --test-dir build         # kernel correctness tests
```

### Requirements
- CUDA 12.x toolkit (`nvcc`), CMake >= 3.24, a C++17 compiler, OpenMP.
- gRPC C++ + Protobuf for the serving binaries (worker/dispatcher/client):
  `sudo apt install -y libgrpc++-dev protobuf-compiler-grpc libprotobuf-dev`
  If gRPC is not found, CMake still builds the kernels + tests and skips the
  serving binaries (warning printed).

## Environment (this machine)
- GPU: **NVIDIA GeForce RTX 5060 Ti** (Blackwell, GB206, **cc 12.0, 36 SMs**).
- CUDA **12.9**, Driver 580.
- **`-arch=sm_120` confirmed working** (compile + run verified). No fallback needed.
- gRPC/Protobuf **not yet installed** as of Phase 0 — resolved in Phase 3.

## Project layout
```
kernels/   hand-written CUDA kernels (gemm, softmax, gelu) + block assembly
src/       worker / dispatcher / client (gRPC)
proto/     infer.proto
tests/     test_kernels.cu (CPU reference), ref_block.py (PyTorch reference)
scripts/   launch_workers.sh, run_experiments.sh, plot.py
docker/    Dockerfile + docker-compose.yml
results/   CSV output + generated graphs
```

## Status / phases
- [x] **Phase 0** — scaffold, CMake build passes, `sm_120` verified.
- [ ] Phase 1 — CUDA kernels + CPU-reference correctness (atol=1e-3).
- [ ] Phase 2 — Transformer block assembly + PyTorch reference (atol/rtol=1e-2).
- [ ] Phase 3 — single gRPC worker, end-to-end client request.
- [ ] Phase 4 — OpenMP dispatcher + multi-worker routing.
- [ ] Phase 5 — experiment harness, plots, `ncu` profiling / roofline.

## Correctness
Recorded per phase as gates are passed (see Phase 1/2).

## Model config (defaults)
`D=768, H=12, d_head=64, FFN=4D`. LayerNorm / residual / dropout omitted for
simplicity (future work).
