#include "gemm_test_internal.hpp"

#include <cuComplex.h>

#include <cstdint>
#include <type_traits>
#include <vector>

namespace gemm_test::detail {
namespace {

template <typename T>
struct CudaDataType;

template <>
struct CudaDataType<double> {
    static constexpr cudaDataType_t value = CUDA_R_64F;
};

template <>
struct CudaDataType<cuDoubleComplex> {
    static constexpr cudaDataType_t value = CUDA_C_64F;
};

template <>
struct CudaDataType<std::int8_t> {
    static constexpr cudaDataType_t value = CUDA_R_8I;
};

template <>
struct CudaDataType<std::int32_t> {
    static constexpr cudaDataType_t value = CUDA_R_32I;
};

template <typename T>
T scalar_one();

template <>
double scalar_one<double>() {
    return 1.0;
}

template <>
std::int32_t scalar_one<std::int32_t>() {
    return 1;
}

template <>
cuDoubleComplex scalar_one<cuDoubleComplex>() {
    return make_cuDoubleComplex(1.0, 0.0);
}

template <typename T>
T scalar_zero();

template <>
double scalar_zero<double>() {
    return 0.0;
}

template <>
std::int32_t scalar_zero<std::int32_t>() {
    return 0;
}

template <>
cuDoubleComplex scalar_zero<cuDoubleComplex>() {
    return make_cuDoubleComplex(0.0, 0.0);
}

template <typename T>
__global__ void fill_kernel(T* data, std::size_t count, T value) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        data[index] = value;
    }
}

template <typename T>
void initialize_buffer(T* data, std::size_t count) {
    constexpr int kThreads = 256;
    const unsigned int blocks =
        static_cast<unsigned int>((count + kThreads - 1) / kThreads);
    fill_kernel<<<blocks, kThreads>>>(data, count, scalar_one<T>());
    GEMM_TEST_CUDA_CHECK(cudaGetLastError());
}

template <>
void initialize_buffer<std::int8_t>(std::int8_t* data, std::size_t count) {
    GEMM_TEST_CUDA_CHECK(
        cudaMemset(data, 1, count * sizeof(std::int8_t)));
}

template <Operation operation,
          typename TypeA,
          typename TypeB,
          typename TypeC,
          typename ScalarType>
OperationResult run_operation(const Options& options,
                              int effective_sm_count,
                              double operations_per_element) {
    constexpr bool kIsLt = operation == Operation::LtMatmulInt8;
    constexpr bool kIsInt8 = operation == Operation::LtMatmulInt8 ||
                             operation == Operation::GemmExInt8;

    const bool batched = kIsInt8 && should_use_batched_mode(options);
    const int batch_count = batched ? options.batch_count : 1;
    if (batched && options.iterations % batch_count != 0) {
        fail("arguments",
             1,
             "iterations must be divisible by batch-count",
             __FILE__,
             __LINE__);
    }

    const int lda = options.transpose_a ? options.k : options.m;
    const int ldb = options.transpose_b ? options.n : options.k;
    const int ldc = options.m;

    if (kIsInt8 && (lda % 4 != 0 || ldb % 4 != 0 || ldc % 4 != 0)) {
        OperationResult skipped;
        skipped.skipped = true;
        return skipped;
    }

    const std::size_t elements_a =
        static_cast<std::size_t>(lda) *
        (options.transpose_a ? options.m : options.k);
    const std::size_t elements_b =
        static_cast<std::size_t>(ldb) *
        (options.transpose_b ? options.k : options.n);
    const std::size_t elements_c =
        static_cast<std::size_t>(ldc) * options.n;

    TypeA* device_a = nullptr;
    TypeB* device_b = nullptr;
    TypeC* device_c = nullptr;
    GEMM_TEST_CUDA_CHECK(cudaMalloc(
        &device_a, elements_a * sizeof(TypeA) * batch_count));
    GEMM_TEST_CUDA_CHECK(cudaMalloc(
        &device_b, elements_b * sizeof(TypeB) * batch_count));
    GEMM_TEST_CUDA_CHECK(cudaMalloc(
        &device_c,
        elements_c * sizeof(TypeC) * batch_count * options.stream_count));

    initialize_buffer(device_a, elements_a * batch_count);
    initialize_buffer(device_b, elements_b * batch_count);
    GEMM_TEST_CUDA_CHECK(cudaMemset(
        device_c,
        0,
        elements_c * sizeof(TypeC) * batch_count * options.stream_count));
    GEMM_TEST_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<cudaStream_t> streams(options.stream_count);
    for (cudaStream_t& stream : streams) {
        GEMM_TEST_CUDA_CHECK(cudaStreamCreate(&stream));
    }

    std::vector<cublasHandle_t> handles;
    cublasLtHandle_t lt_handle = nullptr;
    if constexpr (kIsLt) {
        GEMM_TEST_CUBLAS_CHECK(cublasLtCreate(&lt_handle));
    } else {
        handles.resize(options.stream_count);
        for (int index = 0; index < options.stream_count; ++index) {
            GEMM_TEST_CUBLAS_CHECK(cublasCreate(&handles[index]));
            GEMM_TEST_CUBLAS_CHECK(
                cublasSetStream(handles[index], streams[index]));
            GEMM_TEST_CUBLAS_CHECK(
                cublasSetSmCountTarget(handles[index], effective_sm_count));
        }
    }

    const cublasOperation_t op_a =
        options.transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t op_b =
        options.transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    ScalarType alpha = scalar_one<ScalarType>();
    ScalarType beta = scalar_zero<ScalarType>();

    cublasLtMatrixLayout_t a_layout = nullptr;
    cublasLtMatrixLayout_t b_layout = nullptr;
    cublasLtMatrixLayout_t c_layout = nullptr;
    cublasLtMatmulDesc_t matmul_desc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    std::vector<cublasLtMatmulHeuristicResult_t> heuristic_results;
    const cublasLtMatmulAlgo_t* selected_algorithm = nullptr;
    int returned_algorithms = 0;
    int selected_algorithm_index = -1;

    void* workspace = nullptr;
    const std::size_t workspace_bytes =
        options.workspace_mib * 1024ULL * 1024ULL;

    const long long stride_a = static_cast<long long>(elements_a);
    const long long stride_b = static_cast<long long>(elements_b);
    const long long stride_c = static_cast<long long>(elements_c);

    if constexpr (kIsLt) {
        GEMM_TEST_CUDA_CHECK(cudaMalloc(
            &workspace,
            workspace_bytes * static_cast<std::size_t>(options.stream_count)));

        GEMM_TEST_CUBLAS_CHECK(cublasLtMatmulDescCreate(
            &matmul_desc, CUBLAS_COMPUTE_32I, CUDA_R_32I));
        GEMM_TEST_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            matmul_desc,
            CUBLASLT_MATMUL_DESC_TRANSA,
            &op_a,
            sizeof(op_a)));
        GEMM_TEST_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            matmul_desc,
            CUBLASLT_MATMUL_DESC_TRANSB,
            &op_b,
            sizeof(op_b)));
        GEMM_TEST_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            matmul_desc,
            CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
            &effective_sm_count,
            sizeof(effective_sm_count)));

        const int a_rows = options.transpose_a ? options.k : options.m;
        const int a_columns = options.transpose_a ? options.m : options.k;
        const int b_rows = options.transpose_b ? options.n : options.k;
        const int b_columns = options.transpose_b ? options.k : options.n;

        GEMM_TEST_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
            &a_layout,
            CudaDataType<TypeA>::value,
            a_rows,
            a_columns,
            lda));
        GEMM_TEST_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
            &b_layout,
            CudaDataType<TypeB>::value,
            b_rows,
            b_columns,
            ldb));
        GEMM_TEST_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
            &c_layout,
            CudaDataType<TypeC>::value,
            options.m,
            options.n,
            ldc));

        if (batched) {
            GEMM_TEST_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
                a_layout,
                CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                &batch_count,
                sizeof(batch_count)));
            GEMM_TEST_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
                a_layout,
                CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                &stride_a,
                sizeof(stride_a)));
            GEMM_TEST_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
                b_layout,
                CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                &batch_count,
                sizeof(batch_count)));
            GEMM_TEST_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
                b_layout,
                CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                &stride_b,
                sizeof(stride_b)));
            GEMM_TEST_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
                c_layout,
                CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                &batch_count,
                sizeof(batch_count)));
            GEMM_TEST_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
                c_layout,
                CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                &stride_c,
                sizeof(stride_c)));
        }

        GEMM_TEST_CUBLAS_CHECK(
            cublasLtMatmulPreferenceCreate(&preference));
        GEMM_TEST_CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
            preference,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_bytes,
            sizeof(workspace_bytes)));

        heuristic_results.resize(options.lt_algorithm_count);
        const cublasStatus_t heuristic_status =
            cublasLtMatmulAlgoGetHeuristic(lt_handle,
                                           matmul_desc,
                                           a_layout,
                                           b_layout,
                                           c_layout,
                                           c_layout,
                                           preference,
                                           options.lt_algorithm_count,
                                           heuristic_results.data(),
                                           &returned_algorithms);
        if (heuristic_status != CUBLAS_STATUS_SUCCESS &&
            heuristic_status != CUBLAS_STATUS_NOT_SUPPORTED) {
            fail("cuBLASLt heuristic",
                 static_cast<int>(heuristic_status),
                 "algorithm query failed",
                 __FILE__,
                 __LINE__);
        }
        if (heuristic_status == CUBLAS_STATUS_NOT_SUPPORTED) {
            returned_algorithms = 0;
        }

        for (int index = 0; index < returned_algorithms; ++index) {
            if (heuristic_results[index].state == CUBLAS_STATUS_SUCCESS) {
                selected_algorithm = &heuristic_results[index].algo;
                selected_algorithm_index = index;
                break;
            }
        }

        // A null algorithm is a valid cuBLASLt fallback. It preserves support
        // for shapes where the heuristic API returns no usable candidate.
    }

    auto launch = [&](int stream_index) {
        TypeC* output =
            device_c + static_cast<std::size_t>(stream_index) * elements_c *
                           batch_count;

        if constexpr (operation == Operation::Dgemm) {
            GEMM_TEST_CUBLAS_CHECK(cublasDgemm(
                handles[stream_index],
                op_a,
                op_b,
                options.m,
                options.n,
                options.k,
                reinterpret_cast<const double*>(&alpha),
                device_a,
                lda,
                device_b,
                ldb,
                reinterpret_cast<const double*>(&beta),
                output,
                ldc));
        } else if constexpr (operation == Operation::Zgemm) {
            GEMM_TEST_CUBLAS_CHECK(cublasZgemm(
                handles[stream_index],
                op_a,
                op_b,
                options.m,
                options.n,
                options.k,
                reinterpret_cast<const cuDoubleComplex*>(&alpha),
                device_a,
                lda,
                device_b,
                ldb,
                reinterpret_cast<const cuDoubleComplex*>(&beta),
                output,
                ldc));
        } else if constexpr (operation == Operation::GemmExInt8) {
            if (batched) {
                GEMM_TEST_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                    handles[stream_index],
                    op_a,
                    op_b,
                    options.m,
                    options.n,
                    options.k,
                    &alpha,
                    device_a,
                    CudaDataType<TypeA>::value,
                    lda,
                    stride_a,
                    device_b,
                    CudaDataType<TypeB>::value,
                    ldb,
                    stride_b,
                    &beta,
                    output,
                    CudaDataType<TypeC>::value,
                    ldc,
                    stride_c,
                    batch_count,
                    CUBLAS_COMPUTE_32I,
                    CUBLAS_GEMM_DEFAULT));
            } else {
                GEMM_TEST_CUBLAS_CHECK(cublasGemmEx(
                    handles[stream_index],
                    op_a,
                    op_b,
                    options.m,
                    options.n,
                    options.k,
                    &alpha,
                    device_a,
                    CudaDataType<TypeA>::value,
                    lda,
                    device_b,
                    CudaDataType<TypeB>::value,
                    ldb,
                    &beta,
                    output,
                    CudaDataType<TypeC>::value,
                    ldc,
                    CUBLAS_COMPUTE_32I,
                    CUBLAS_GEMM_DEFAULT));
            }
        } else {
            void* stream_workspace =
                static_cast<char*>(workspace) +
                static_cast<std::size_t>(stream_index) * workspace_bytes;
            GEMM_TEST_CUBLAS_CHECK(cublasLtMatmul(
                lt_handle,
                matmul_desc,
                &alpha,
                device_a,
                a_layout,
                device_b,
                b_layout,
                &beta,
                output,
                c_layout,
                output,
                c_layout,
                selected_algorithm,
                stream_workspace,
                workspace_bytes,
                streams[stream_index]));
        }
    };

    for (int iteration = 0; iteration < options.warmup; ++iteration) {
        launch(iteration % options.stream_count);
    }
    GEMM_TEST_CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    GEMM_TEST_CUDA_CHECK(cudaEventCreate(&start));
    GEMM_TEST_CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<cudaEvent_t> stream_done(options.stream_count);
    for (cudaEvent_t& event : stream_done) {
        GEMM_TEST_CUDA_CHECK(
            cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }

    GEMM_TEST_CUDA_CHECK(cudaEventRecord(start, streams.front()));
    for (int index = 1; index < options.stream_count; ++index) {
        GEMM_TEST_CUDA_CHECK(
            cudaStreamWaitEvent(streams[index], start, 0));
    }

    const int launch_count =
        batched ? options.iterations / batch_count : options.iterations;
    for (int iteration = 0; iteration < launch_count; ++iteration) {
        launch(iteration % options.stream_count);
    }

    for (int index = 0; index < options.stream_count; ++index) {
        GEMM_TEST_CUDA_CHECK(
            cudaEventRecord(stream_done[index], streams[index]));
    }
    for (int index = 1; index < options.stream_count; ++index) {
        GEMM_TEST_CUDA_CHECK(
            cudaStreamWaitEvent(streams.front(), stream_done[index], 0));
    }
    GEMM_TEST_CUDA_CHECK(cudaEventRecord(stop, streams.front()));
    GEMM_TEST_CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_milliseconds = 0.0F;
    GEMM_TEST_CUDA_CHECK(
        cudaEventElapsedTime(&elapsed_milliseconds, start, stop));
    MPI_Barrier(MPI_COMM_WORLD);

    OperationResult result;
    result.total_milliseconds = elapsed_milliseconds;
    if (elapsed_milliseconds > 0.0F) {
        result.tops = operations_per_element * static_cast<double>(options.m) *
                      static_cast<double>(options.n) *
                      static_cast<double>(options.k) *
                      static_cast<double>(options.iterations) /
                      (static_cast<double>(elapsed_milliseconds) * 1.0e9);
    }
    result.returned_algorithms = returned_algorithms;
    result.selected_algorithm = selected_algorithm_index;

    for (cudaEvent_t event : stream_done) {
        GEMM_TEST_CUDA_CHECK(cudaEventDestroy(event));
    }
    GEMM_TEST_CUDA_CHECK(cudaEventDestroy(start));
    GEMM_TEST_CUDA_CHECK(cudaEventDestroy(stop));

    if constexpr (kIsLt) {
        GEMM_TEST_CUBLAS_CHECK(
            cublasLtMatmulPreferenceDestroy(preference));
        GEMM_TEST_CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(a_layout));
        GEMM_TEST_CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(b_layout));
        GEMM_TEST_CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(c_layout));
        GEMM_TEST_CUBLAS_CHECK(cublasLtMatmulDescDestroy(matmul_desc));
        GEMM_TEST_CUBLAS_CHECK(cublasLtDestroy(lt_handle));
        GEMM_TEST_CUDA_CHECK(cudaFree(workspace));
    } else {
        for (cublasHandle_t handle : handles) {
            GEMM_TEST_CUBLAS_CHECK(cublasDestroy(handle));
        }
    }

    for (cudaStream_t stream : streams) {
        GEMM_TEST_CUDA_CHECK(cudaStreamDestroy(stream));
    }
    GEMM_TEST_CUDA_CHECK(cudaFree(device_c));
    GEMM_TEST_CUDA_CHECK(cudaFree(device_b));
    GEMM_TEST_CUDA_CHECK(cudaFree(device_a));

    return result;
}

}  // namespace

bool should_use_batched_mode(const Options& options) {
    return options.size_threshold > 0 && options.m < options.size_threshold &&
           options.n < options.size_threshold && options.k < options.size_threshold;
}

OperationResult run_dgemm(const Options& options, int effective_sm_count) {
    return run_operation<Operation::Dgemm, double, double, double, double>(
        options, effective_sm_count, 2.0);
}

OperationResult run_zgemm(const Options& options, int effective_sm_count) {
    return run_operation<Operation::Zgemm,
                         cuDoubleComplex,
                         cuDoubleComplex,
                         cuDoubleComplex,
                         cuDoubleComplex>(options, effective_sm_count, 8.0);
}

OperationResult run_gemmex_int8(const Options& options,
                                int effective_sm_count) {
    return run_operation<Operation::GemmExInt8,
                         std::int8_t,
                         std::int8_t,
                         std::int32_t,
                         std::int32_t>(options, effective_sm_count, 2.0);
}

OperationResult run_ltmatmul_int8(const Options& options,
                                  int effective_sm_count) {
    return run_operation<Operation::LtMatmulInt8,
                         std::int8_t,
                         std::int8_t,
                         std::int32_t,
                         std::int32_t>(options, effective_sm_count, 2.0);
}

}  // namespace gemm_test::detail
