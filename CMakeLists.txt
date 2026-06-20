cmake_minimum_required(VERSION 3.24)
project(distbatch_infer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------------------
# CUDA architecture: target sm_120 (Blackwell, RTX 5060 Ti). Fall back to
# "native" if the toolkit rejects 120. Override with -DCUDA_ARCH=...
# ---------------------------------------------------------------------------
set(CUDA_ARCH "120" CACHE STRING "CUDA architecture (e.g. 120, native)")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
message(STATUS "CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -lineinfo")

# CUDA runtime for host-compiled targets (worker uses cudaMalloc/events and
# includes block.cuh which references cudaStream_t).
find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------------------------
# Kernels library (the core deliverable: hand-written CUDA kernels)
# ---------------------------------------------------------------------------
add_library(kernels STATIC
  kernels/gemm.cu
  kernels/softmax.cu
  kernels/gelu.cu
  kernels/transpose.cu
  kernels/block.cu
)
target_include_directories(kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/kernels)
set_target_properties(kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ---------------------------------------------------------------------------
# Kernel correctness tests (CPU reference, no gRPC needed)
# ---------------------------------------------------------------------------
add_executable(test_kernels tests/test_kernels.cu)
target_link_libraries(test_kernels PRIVATE kernels)

add_executable(test_block tests/test_block.cu)
target_link_libraries(test_block PRIVATE kernels)

enable_testing()
add_test(NAME kernels COMMAND test_kernels)

# Block test: generate PyTorch fixtures (setup fixture), then compare.
find_program(PYTHON3_EXE python3)
if(PYTHON3_EXE)
  add_test(NAME gen_fixtures
           COMMAND ${PYTHON3_EXE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/ref_block.py
                   --out ${CMAKE_CURRENT_BINARY_DIR}/fixtures)
  set_tests_properties(gen_fixtures PROPERTIES FIXTURES_SETUP BLOCKFIX)
  add_test(NAME block COMMAND test_block ${CMAKE_CURRENT_BINARY_DIR}/fixtures)
  set_tests_properties(block PROPERTIES FIXTURES_REQUIRED BLOCKFIX)
endif()

# ---------------------------------------------------------------------------
# OpenMP (dispatcher)
# ---------------------------------------------------------------------------
find_package(OpenMP)

# ---------------------------------------------------------------------------
# gRPC + Protobuf (worker / dispatcher / client).
# Ubuntu's apt packages (libgrpc++-dev, libprotobuf-dev) ship ONLY pkg-config
# files, no CMake config packages. So we use the FindProtobuf module for
# protobuf and pkg-config for grpc++, plus the protoc / grpc_cpp_plugin
# executables directly. Still optional so the scaffold builds without gRPC.
# ---------------------------------------------------------------------------
find_package(Protobuf MODULE QUIET)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(GRPCPP QUIET IMPORTED_TARGET grpc++)
endif()
find_program(PROTOC_EXECUTABLE protoc)
find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin)

if(Protobuf_FOUND AND GRPCPP_FOUND AND PROTOC_EXECUTABLE AND GRPC_CPP_PLUGIN)
  message(STATUS "gRPC ${GRPCPP_VERSION} + Protobuf ${Protobuf_VERSION} found: "
                 "building worker/dispatcher/client")
  set(BUILD_GRPC ON)
else()
  message(WARNING "gRPC/Protobuf NOT fully found: skipping worker/dispatcher/client (Phase 3+). "
                  "Install: sudo apt install libgrpc++-dev protobuf-compiler-grpc libprotobuf-dev")
  set(BUILD_GRPC OFF)
endif()

if(BUILD_GRPC)
  # Generate protobuf + grpc sources from proto/infer.proto
  set(PROTO_FILE ${CMAKE_CURRENT_SOURCE_DIR}/proto/infer.proto)
  set(PROTO_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/gen)
  file(MAKE_DIRECTORY ${PROTO_GEN_DIR})

  get_filename_component(PROTO_PATH ${PROTO_FILE} DIRECTORY)
  set(PROTO_SRCS ${PROTO_GEN_DIR}/infer.pb.cc)
  set(PROTO_HDRS ${PROTO_GEN_DIR}/infer.pb.h)
  set(GRPC_SRCS  ${PROTO_GEN_DIR}/infer.grpc.pb.cc)
  set(GRPC_HDRS  ${PROTO_GEN_DIR}/infer.grpc.pb.h)

  add_custom_command(
    OUTPUT ${PROTO_SRCS} ${PROTO_HDRS} ${GRPC_SRCS} ${GRPC_HDRS}
    COMMAND ${PROTOC_EXECUTABLE}
    ARGS --cpp_out=${PROTO_GEN_DIR}
         --grpc_out=${PROTO_GEN_DIR}
         --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
         -I ${PROTO_PATH}
         ${PROTO_FILE}
    DEPENDS ${PROTO_FILE}
    COMMENT "Generating gRPC/protobuf sources from infer.proto")

  add_library(proto_gen STATIC ${PROTO_SRCS} ${GRPC_SRCS})
  target_include_directories(proto_gen PUBLIC ${PROTO_GEN_DIR})
  target_link_libraries(proto_gen PUBLIC PkgConfig::GRPCPP protobuf::libprotobuf)

  add_executable(worker src/worker.cpp)
  target_link_libraries(worker PRIVATE kernels proto_gen CUDA::cudart)

  add_executable(dispatcher src/dispatcher.cpp)
  target_link_libraries(dispatcher PRIVATE proto_gen)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(dispatcher PRIVATE OpenMP::OpenMP_CXX)
  endif()

  add_executable(client src/client.cpp)
  target_link_libraries(client PRIVATE proto_gen)
endif()
