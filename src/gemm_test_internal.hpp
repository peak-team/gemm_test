#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <mpi.h>

#include <array>
#include <cstddef>

namespace gemm_test::detail {

enum class Operation {
    Dgemm = 0,
    Zgemm = 1,
    GemmExInt8 = 2,
    LtMatmulInt8 = 3,
    Count = 4,
};

constexpr int kOperationCount = static_cast<int>(Operation::Count);

struct Options {
    int m = 4096;
    int n = 4096;
    int k = 4096;
    int iterations = 100;
    int warmup = 10;
    int sm_count = 0;
    int stream_count = 1;
    int size_threshold = 0;
    int batch_count = 1;
    int lt_algorithm_count = 32;
    std::size_t workspace_mib = 64;
    bool transpose_a = true;
    bool transpose_b = false;
    bool verbose = false;
    std::array<bool, kOperationCount> run = {true, false, true, true};
};

struct DeviceContext {
    int device_count = 0;
    int device_id = 0;
    int total_sm_count = 0;
    int effective_sm_count = 0;
    CUcontext context = nullptr;
};

struct OperationResult {
    float total_milliseconds = 0.0F;
    double tops = 0.0;
    int returned_algorithms = 0;
    int selected_algorithm = -1;
    bool skipped = false;
};

[[noreturn]] void fail(const char* api,
                       int code,
                       const char* message,
                       const char* file,
                       int line);

Options parse_options(int argc, char** argv, int rank);
DeviceContext initialize_device(int rank,
                                int requested_sm_count,
                                bool use_context_affinity);
void destroy_device_context(DeviceContext& device);
void print_individual_result(int rank,
                             const char* operation_name,
                             const OperationResult& result,
                             int iterations);

bool should_use_batched_mode(const Options& options);

OperationResult run_dgemm(const Options& options, int effective_sm_count);
OperationResult run_zgemm(const Options& options, int effective_sm_count);
OperationResult run_gemmex_int8(const Options& options, int effective_sm_count);
OperationResult run_ltmatmul_int8(const Options& options,
                                  int effective_sm_count);

}  // namespace gemm_test::detail

#define GEMM_TEST_CUDA_CHECK(call)                                              \
    do {                                                                        \
        const cudaError_t status_ = (call);                                     \
        if (status_ != cudaSuccess) {                                           \
            ::gemm_test::detail::fail("CUDA",                                  \
                                      static_cast<int>(status_),                 \
                                      cudaGetErrorString(status_),               \
                                      __FILE__,                                  \
                                      __LINE__);                                 \
        }                                                                       \
    } while (0)

#define GEMM_TEST_CUDA_DRIVER_CHECK(call)                                       \
    do {                                                                        \
        const CUresult status_ = (call);                                        \
        if (status_ != CUDA_SUCCESS) {                                          \
            const char* message_ = nullptr;                                     \
            cuGetErrorString(status_, &message_);                               \
            ::gemm_test::detail::fail("CUDA Driver",                           \
                                      static_cast<int>(status_),                 \
                                      message_,                                  \
                                      __FILE__,                                  \
                                      __LINE__);                                 \
        }                                                                       \
    } while (0)

#define GEMM_TEST_CUBLAS_CHECK(call)                                            \
    do {                                                                        \
        const cublasStatus_t status_ = (call);                                  \
        if (status_ != CUBLAS_STATUS_SUCCESS) {                                 \
            ::gemm_test::detail::fail("cuBLAS",                                \
                                      static_cast<int>(status_),                 \
                                      "call failed",                            \
                                      __FILE__,                                  \
                                      __LINE__);                                 \
        }                                                                       \
    } while (0)
