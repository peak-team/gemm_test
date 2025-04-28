#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuComplex.h> // For cuDoubleComplex
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <getopt.h>
#include <mpi.h>        // Include MPI
#include <complex>      // For std::complex
#include <cstdint>      // For int8_t, int32_t
#include <type_traits> // For std::is_same_v, std::is_floating_point_v
#include <string>       // For std::string
#include <vector>
#include <stdexcept>
#include <iostream>     // For cerr
#include <utility>      // For std::pair
#include <limits>       // For numeric_limits
#include <array>        // For storing results

// Error checking macro for CUDA
#define CUDA_CHECK(err)                                                        \
    do {                                                                       \
        cudaError_t err_ = (err);                                              \
        if (err_ != cudaSuccess) {                                             \
            int rank_ = -1;                                                    \
            int initialized_ = 0;                                              \
            MPI_Initialized(&initialized_);                                    \
            if (initialized_) MPI_Comm_rank(MPI_COMM_WORLD, &rank_);           \
            fprintf(stderr, "[MPI Rank %d] CUDA Error: %s at %s:%d\n", rank_,  \
                    cudaGetErrorString(err_), __FILE__, __LINE__);             \
            if (initialized_) MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);         \
            else exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// Error checking macro for cuBLAS (covers both v2 and Lt)
#define CUBLAS_CHECK(err)                                                      \
    do {                                                                       \
        cublasStatus_t err_ = (err);                                           \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                   \
            int rank_ = -1;                                                    \
            int initialized_ = 0;                                              \
            MPI_Initialized(&initialized_);                                    \
            if (initialized_) MPI_Comm_rank(MPI_COMM_WORLD, &rank_);           \
            const char* errorString = "Unknown cuBLAS Error";                  \
            switch(err_) {                                                     \
                case CUBLAS_STATUS_NOT_INITIALIZED: errorString = "CUBLAS_STATUS_NOT_INITIALIZED"; break; \
                case CUBLAS_STATUS_ALLOC_FAILED:    errorString = "CUBLAS_STATUS_ALLOC_FAILED"; break;    \
                case CUBLAS_STATUS_INVALID_VALUE:   errorString = "CUBLAS_STATUS_INVALID_VALUE"; break;  \
                case CUBLAS_STATUS_ARCH_MISMATCH:   errorString = "CUBLAS_STATUS_ARCH_MISMATCH"; break;  \
                case CUBLAS_STATUS_MAPPING_ERROR:   errorString = "CUBLAS_STATUS_MAPPING_ERROR"; break;  \
                case CUBLAS_STATUS_EXECUTION_FAILED: errorString = "CUBLAS_STATUS_EXECUTION_FAILED"; break;\
                case CUBLAS_STATUS_INTERNAL_ERROR:  errorString = "CUBLAS_STATUS_INTERNAL_ERROR"; break; \
                case CUBLAS_STATUS_NOT_SUPPORTED:   errorString = "CUBLAS_STATUS_NOT_SUPPORTED"; break;  \
                case CUBLAS_STATUS_LICENSE_ERROR:   errorString = "CUBLAS_STATUS_LICENSE_ERROR"; break;  \
                default: /* errorString already set */ break;                  \
            }                                                                  \
            fprintf(stderr, "[MPI Rank %d] cuBLAS Error: %s (%d) at %s:%d\n",  \
                    rank_, errorString, static_cast<int>(err_), __FILE__, __LINE__);        \
            if (initialized_) MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);         \
            else exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)


// Helper to get CUDA data type enum from C++ type
template <typename T> struct CudaDataType;
template <> struct CudaDataType<double>          { static constexpr cudaDataType_t value = CUDA_R_64F; };
template <> struct CudaDataType<float>           { static constexpr cudaDataType_t value = CUDA_R_32F; };
template <> struct CudaDataType<cuDoubleComplex> { static constexpr cudaDataType_t value = CUDA_C_64F; };
template <> struct CudaDataType<cuComplex>       { static constexpr cudaDataType_t value = CUDA_C_32F; };
template <> struct CudaDataType<int8_t>          { static constexpr cudaDataType_t value = CUDA_R_8I;  };
template <> struct CudaDataType<int32_t>         { static constexpr cudaDataType_t value = CUDA_R_32I; };
// Add half, etc. if needed

// Enum to identify the GEMM operation type for dispatch and array indexing
enum class GemmOpIndex {
    DGEMM = 0,
    ZGEMM = 1,
    GEMM_EX_INT8 = 2,
    LT_MATMUL_INT8 = 3,
    COUNT // Number of test types
};
const int NUM_TEST_TYPES = static_cast<int>(GemmOpIndex::COUNT);

// --- Generic GEMM Test Function --- (Implementation remains the same as previous version)
// Returns: { total_milliseconds, t_ops } for this rank
template <GemmOpIndex OperationTypeIndex, // Use enum for dispatch (changed name slightly for clarity)
          typename TypeA, typename TypeB, typename TypeC, typename TypeCompute>
std::pair<float, double> runGemmTest(int m, int n, int k, bool transposeA, bool transposeB, int iterations,
                                     double flopsPerOp, // Pass FLOPs as a regular argument
                                     const char* opName, // Name for printing results (used for skip message)
                                     int rank)
{
    // --- Handle Creation (Dispatch based on type) ---
    cublasHandle_t handle = nullptr;
    cublasLtHandle_t ltHandle = nullptr;

    if constexpr (OperationTypeIndex == GemmOpIndex::LT_MATMUL_INT8) {
        CUBLAS_CHECK(cublasLtCreate(&ltHandle));
    } else {
        CUBLAS_CHECK(cublasCreate(&handle));
    }
    // --- End Handle Creation ---

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Set stream (only for v2 handle)
    if constexpr (OperationTypeIndex != GemmOpIndex::LT_MATMUL_INT8) {
        CUBLAS_CHECK(cublasSetStream(handle, stream));
    }

    int lda = transposeA ? k : m;
    int ldb = transposeB ? n : k;
    int ldc = m;

    // --- Special handling for LtMatmul constraints ---
    if constexpr (OperationTypeIndex == GemmOpIndex::LT_MATMUL_INT8) {
         if (lda % 4 != 0 || ldb % 4 != 0 || ldc % 4 != 0) {
             if (ltHandle) CUBLAS_CHECK(cublasLtDestroy(ltHandle));
             CUDA_CHECK(cudaStreamDestroy(stream));
             return {0.0f, 0.0}; // Return 0s to indicate skip
         }
    }
    // --- End LtMatmul specific constraints ---

    size_t sizeA = static_cast<size_t>(lda) * (transposeA ? m : k) * sizeof(TypeA);
    size_t sizeB = static_cast<size_t>(ldb) * (transposeB ? k : n) * sizeof(TypeB);
    size_t sizeC = static_cast<size_t>(ldc) * n * sizeof(TypeC);

    // Allocate device memory
    TypeA *d_A = nullptr;
    TypeB *d_B = nullptr;
    TypeC *d_C = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_A, sizeA, stream));
    CUDA_CHECK(cudaMallocAsync(&d_B, sizeB, stream));
    CUDA_CHECK(cudaMallocAsync(&d_C, sizeC, stream));

    // Allocate and Initialize host matrices
    TypeA *h_A = (TypeA *)malloc(sizeA);
    TypeB *h_B = (TypeB *)malloc(sizeB);
    if (!h_A || !h_B) {
        fprintf(stderr, "[MPI Rank %d] Error: Failed to allocate host memory\n", rank);
        if (h_A) free(h_A);
        if (h_B) free(h_B);
        if(d_A) cudaFreeAsync(d_A, stream); if(d_B) cudaFreeAsync(d_B, stream); if(d_C) cudaFreeAsync(d_C, stream);
        if (handle) CUBLAS_CHECK(cublasDestroy(handle)); if (ltHandle) CUBLAS_CHECK(cublasLtDestroy(ltHandle));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Initialize with random data
    for (size_t i = 0; i < sizeA / sizeof(TypeA); i++) {
        if constexpr (std::is_same_v<TypeA, cuDoubleComplex>) { h_A[i].x = (double)(rand() % 100)/100.0; h_A[i].y = (double)(rand() % 100)/100.0; }
        else if constexpr (std::is_same_v<TypeA, int8_t>) { h_A[i] = static_cast<int8_t>(rand() % 200 - 100); }
        else if constexpr (std::is_floating_point_v<TypeA>) { h_A[i] = static_cast<TypeA>((rand() % 1000 - 500) / 500.0); }
        else { h_A[i] = static_cast<TypeA>(rand() % 100); }
    }
     for (size_t i = 0; i < sizeB / sizeof(TypeB); i++) {
        if constexpr (std::is_same_v<TypeB, cuDoubleComplex>) { h_B[i].x = (double)(rand() % 100)/100.0; h_B[i].y = (double)(rand() % 100)/100.0; }
        else if constexpr (std::is_same_v<TypeB, int8_t>) { h_B[i] = static_cast<int8_t>(rand() % 200 - 100); }
        else if constexpr (std::is_floating_point_v<TypeB>) { h_B[i] = static_cast<TypeB>((rand() % 1000 - 500) / 500.0); }
        else { h_B[i] = static_cast<TypeB>(rand() % 100); }
    }

    // Wait for allocations before copying
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, sizeA, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, sizeB, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(d_C, 0, sizeC, stream)); // Initialize C

    // Set transpose operations
    cublasOperation_t opA = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;

    // Define scalars (alpha, beta) based on TypeCompute
    TypeCompute alpha_compute, beta_compute;
    if constexpr (std::is_same_v<TypeCompute, cuDoubleComplex>) { alpha_compute = make_cuDoubleComplex(1.0, 0.0); beta_compute = make_cuDoubleComplex(0.0, 0.0); }
    else if constexpr (std::is_same_v<TypeCompute, int32_t>) { alpha_compute = 1; beta_compute = 0; }
    else if constexpr (std::is_same_v<TypeCompute, double>) { alpha_compute = 1.0; beta_compute = 0.0; }
    else if constexpr (std::is_same_v<TypeCompute, float>) { alpha_compute = 1.0f; beta_compute = 0.0f; }
    else { static_assert(sizeof(TypeCompute) == 0, "Unsupported TypeCompute in runGemmTest"); }

    // Timing setup
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- LtMatmul specific setup ---
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
    cublasLtMatmulDesc_t matmulDesc = nullptr;
    void *workspace = nullptr;
    size_t workspaceSize = 0;
    if constexpr (OperationTypeIndex == GemmOpIndex::LT_MATMUL_INT8) {
        cudaDataType_t scaleType = CUDA_R_32I; cublasComputeType_t computeType = CUBLAS_COMPUTE_32I;
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CudaDataType<TypeA>::value, transposeA ? m : k, transposeA ? k : m, lda));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CudaDataType<TypeB>::value, transposeB ? k : n, transposeB ? n : k, ldb));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, CudaDataType<TypeC>::value, m, n, ldc));
        CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmulDesc, computeType, scaleType));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));
        workspaceSize = 1024 * 1024 * 4; CUDA_CHECK(cudaMallocAsync(&workspace, workspaceSize, stream));
    }
    // --- End LtMatmul specific setup ---

    CUDA_CHECK(cudaStreamSynchronize(stream)); // Wait for copies and Lt setup

    // Warm-up call
    if constexpr (OperationTypeIndex == GemmOpIndex::DGEMM) {
        CUBLAS_CHECK(cublasDgemm(handle, opA, opB, m, n, k, (const double*)&alpha_compute, d_A, lda, d_B, ldb, (const double*)&beta_compute, d_C, ldc));
    } else if constexpr (OperationTypeIndex == GemmOpIndex::ZGEMM) {
        CUBLAS_CHECK(cublasZgemm(handle, opA, opB, m, n, k, (const cuDoubleComplex*)&alpha_compute, d_A, lda, d_B, ldb, (const cuDoubleComplex*)&beta_compute, d_C, ldc));
    } else if constexpr (OperationTypeIndex == GemmOpIndex::GEMM_EX_INT8) {
        cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
        CUBLAS_CHECK(cublasGemmEx(handle, opA, opB, m, n, k, &alpha_compute, d_A, CudaDataType<TypeA>::value, lda, d_B, CudaDataType<TypeB>::value, ldb, &beta_compute, d_C, CudaDataType<TypeC>::value, ldc, CUBLAS_COMPUTE_32I, algo));
    } else if constexpr (OperationTypeIndex == GemmOpIndex::LT_MATMUL_INT8) {
        CUBLAS_CHECK(cublasLtMatmul(ltHandle, matmulDesc, &alpha_compute, d_A, Adesc, d_B, Bdesc, &beta_compute, d_C, Cdesc, d_C, Cdesc, NULL, workspace, workspaceSize, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream)); // Wait for warm-up

    // --- Timed Execution Loop ---
    MPI_Barrier(MPI_COMM_WORLD);
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; i++) {
        if constexpr (OperationTypeIndex == GemmOpIndex::DGEMM) {
            CUBLAS_CHECK(cublasDgemm(handle, opA, opB, m, n, k, (const double*)&alpha_compute, d_A, lda, d_B, ldb, (const double*)&beta_compute, d_C, ldc));
        } else if constexpr (OperationTypeIndex == GemmOpIndex::ZGEMM) {
            CUBLAS_CHECK(cublasZgemm(handle, opA, opB, m, n, k, (const cuDoubleComplex*)&alpha_compute, d_A, lda, d_B, ldb, (const cuDoubleComplex*)&beta_compute, d_C, ldc));
        } else if constexpr (OperationTypeIndex == GemmOpIndex::GEMM_EX_INT8) {
            cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
            CUBLAS_CHECK(cublasGemmEx(handle, opA, opB, m, n, k, &alpha_compute, d_A, CudaDataType<TypeA>::value, lda, d_B, CudaDataType<TypeB>::value, ldb, &beta_compute, d_C, CudaDataType<TypeC>::value, ldc, CUBLAS_COMPUTE_32I, algo));
        } else if constexpr (OperationTypeIndex == GemmOpIndex::LT_MATMUL_INT8) {
            CUBLAS_CHECK(cublasLtMatmul(ltHandle, matmulDesc, &alpha_compute, d_A, Adesc, d_B, Bdesc, &beta_compute, d_C, Cdesc, d_C, Cdesc, NULL, workspace, workspaceSize, stream));
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    MPI_Barrier(MPI_COMM_WORLD);
    // --- End Timed Execution Loop ---

    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Compute TFLOPS/TOPS for this rank
    double total_ops = flopsPerOp * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k) * iterations;
    double avg_time_s = (milliseconds / 1000.0);
    double t_ops = (avg_time_s > 1e-9 && iterations > 0) ? (total_ops / avg_time_s) / 1e12 : 0.0;

    // Cleanup host memory
    free(h_A); free(h_B);
    // Cleanup device memory (async)
    CUDA_CHECK(cudaFreeAsync(d_A, stream)); CUDA_CHECK(cudaFreeAsync(d_B, stream)); CUDA_CHECK(cudaFreeAsync(d_C, stream));
    // LtMatmul specific cleanup (async)
    if constexpr (OperationTypeIndex == GemmOpIndex::LT_MATMUL_INT8) {
        if (Adesc) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Adesc));
        if (Bdesc) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Bdesc));
        if (Cdesc) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Cdesc));
        if (matmulDesc) CUBLAS_CHECK(cublasLtMatmulDescDestroy(matmulDesc));
        if (workspace) CUDA_CHECK(cudaFreeAsync(workspace, stream));
    }
    // Wait for async frees
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // Cleanup CUDA Events and Stream
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop)); CUDA_CHECK(cudaStreamDestroy(stream));
    // Cleanup cuBLAS Handles
    if (handle) CUBLAS_CHECK(cublasDestroy(handle)); if (ltHandle) CUBLAS_CHECK(cublasLtDestroy(ltHandle));

    return {milliseconds, t_ops}; // Return total time and T*OPS
}


int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Default values
    int m = 4096, n = 4096, k = 4096, iterations = 10;
    bool transposeA = true, transposeB = false, verbose = false;
    bool runDgemm = true, runZgemm = false, runGemmEx = true, runLtMatmul = true;
    std::array<bool, NUM_TEST_TYPES> run_flags = {runDgemm, runZgemm, runGemmEx, runLtMatmul}; // Initial flags

    // Option parsing (using array for flags)
    static struct option long_options[] = {
        {"m", required_argument, 0, 'm'}, {"n", required_argument, 0, 'n'},
        {"k", required_argument, 0, 'k'}, {"transposeA", required_argument, 0, 'a'},
        {"transposeB", required_argument, 0, 'b'}, {"iterations", required_argument, 0, 'i'},
        {"verbose", no_argument, 0, 'v'}, {"mn", required_argument, 0, '1'},
        {"mk", required_argument, 0, '2'}, {"nk", required_argument, 0, '3'},
        {"mnk", required_argument, 0, '4'},
        {"dgemm", required_argument, 0, 'd'}, {"zgemm", required_argument, 0, 'z'},
        {"gemmex", required_argument, 0, 'g'}, {"ltmatmul", required_argument, 0, 'l'},
        {0, 0, 0, 0}
    };
    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "m:n:k:a:b:i:v1:2:3:4:d:z:g:l:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'm': m = atoi(optarg); break; case 'n': n = atoi(optarg); break;
            case 'k': k = atoi(optarg); break; case 'a': transposeA = atoi(optarg) != 0; break;
            case 'b': transposeB = atoi(optarg) != 0; break; case 'i': iterations = atoi(optarg); break;
            case 'v': verbose = true; break; case '1': m = n = atoi(optarg); break;
            case '2': m = k = atoi(optarg); break; case '3': n = k = atoi(optarg); break;
            case '4': m = n = k = atoi(optarg); break;
            // Update run flags directly
            case 'd': run_flags[static_cast<int>(GemmOpIndex::DGEMM)] = (atoi(optarg) != 0); break;
            case 'z': run_flags[static_cast<int>(GemmOpIndex::ZGEMM)] = (atoi(optarg) != 0); break;
            case 'g': run_flags[static_cast<int>(GemmOpIndex::GEMM_EX_INT8)] = (atoi(optarg) != 0); break;
            case 'l': run_flags[static_cast<int>(GemmOpIndex::LT_MATMUL_INT8)] = (atoi(optarg) != 0); break;
            case '?': if (rank == 0) { /* Print usage */ } MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); break;
            default: if (rank == 0) fprintf(stderr, "Unexpected option parsing error.\n"); MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    // GPU Selection
    int deviceCount; CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) { if (rank == 0) fprintf(stderr, "Error: No CUDA devices found.\n"); MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); }
    int deviceId = rank % deviceCount; CUDA_CHECK(cudaSetDevice(deviceId));
    if (verbose && rank == 0) {
        printf("MPI Size: %d, GPUs: %d\n", size, deviceCount);
        for(int r=0; r<size; ++r) printf("  Rank %d -> GPU %d\n", r, r % deviceCount);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Seed RNG
    srand(time(NULL) + rank);

    // Parameter validation
    if (m <= 0 || n <= 0 || k <= 0 || iterations <= 0) {
        if (rank == 0) fprintf(stderr, "Error: m, n, k, iterations must be positive\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Print header (Rank 0)
    if (rank == 0) {
        if (verbose) {
            printf("---------------------------------------------------------------------\n");
            printf("MPI GEMM Test: m=%d, n=%d, k=%d, iterations=%d, ranks=%d\n", m, n, k, iterations, size);
            printf("Transpose A: %s, Transpose B: %s\n", transposeA ? "Yes" : "No", transposeB ? "Yes" : "No");
            printf("Tests: Dgemm:%s Zgemm:%s GemmEx(I8):%s LtMatmul(I8):%s Verbose:%s\n",
                   run_flags[0]?"Y":"N", run_flags[1]?"Y":"N", run_flags[2]?"Y":"N", run_flags[3]?"Y":"N", verbose ? "Y":"N");
            printf("---------------------------------------------------------------------\n");
            printf("--- Individual Rank Performance ---\n");
            printf("| Rank | Operation          | Time/Iter (ms) | T*OPS    |\n");
            printf("|------|--------------------|----------------|----------|\n");
        }
        // Aggregated header printed later, after all tests run
    }

    // Define operation names and flops
    const char* opNames[NUM_TEST_TYPES] = {"Dgemm", "Zgemm", "GemmEx(int8)", "LtMatmul(int8)"};
    const double opFlops[NUM_TEST_TYPES] = {2.0, 8.0, 2.0, 2.0};

    // --- Result Storage ---
    // Local results for this rank
    std::array<float, NUM_TEST_TYPES> my_times_ms = {0.0f};
    std::array<double, NUM_TEST_TYPES> my_tops = {0.0};
    // Aggregated results on Rank 0
    std::array<float, NUM_TEST_TYPES> reduced_times_sum = {0.0f};
    std::array<double, NUM_TEST_TYPES> reduced_tops_sum = {0.0};

    // --- Run Selected Tests ---
    if (run_flags[static_cast<int>(GemmOpIndex::DGEMM)]) {
        int idx = static_cast<int>(GemmOpIndex::DGEMM);
        auto result = runGemmTest<GemmOpIndex::DGEMM, double, double, double, double>
                      (m, n, k, transposeA, transposeB, iterations, opFlops[idx], opNames[idx], rank);
        my_times_ms[idx] = result.first; my_tops[idx] = result.second;
        if (verbose) printf("| %4d | %-18s | %14.3f | %8.3f |\n", rank, opNames[idx], (iterations > 0 ? my_times_ms[idx]/iterations : 0.0), my_tops[idx]);
        MPI_Reduce(&my_times_ms[idx], &reduced_times_sum[idx], 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&my_tops[idx], &reduced_tops_sum[idx], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    if (run_flags[static_cast<int>(GemmOpIndex::ZGEMM)]) {
        int idx = static_cast<int>(GemmOpIndex::ZGEMM);
        auto result = runGemmTest<GemmOpIndex::ZGEMM, cuDoubleComplex, cuDoubleComplex, cuDoubleComplex, cuDoubleComplex>
                      (m, n, k, transposeA, transposeB, iterations, opFlops[idx], opNames[idx], rank);
        my_times_ms[idx] = result.first; my_tops[idx] = result.second;
        if (verbose) printf("| %4d | %-18s | %14.3f | %8.3f |\n", rank, opNames[idx], (iterations > 0 ? my_times_ms[idx]/iterations : 0.0), my_tops[idx]);
        MPI_Reduce(&my_times_ms[idx], &reduced_times_sum[idx], 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&my_tops[idx], &reduced_tops_sum[idx], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    if (run_flags[static_cast<int>(GemmOpIndex::GEMM_EX_INT8)]) {
        int idx = static_cast<int>(GemmOpIndex::GEMM_EX_INT8);
        auto result = runGemmTest<GemmOpIndex::GEMM_EX_INT8, int8_t, int8_t, int32_t, int32_t>
                      (m, n, k, transposeA, transposeB, iterations, opFlops[idx], opNames[idx], rank);
        my_times_ms[idx] = result.first; my_tops[idx] = result.second;
        if (verbose) printf("| %4d | %-18s | %14.3f | %8.3f |\n", rank, opNames[idx], (iterations > 0 ? my_times_ms[idx]/iterations : 0.0), my_tops[idx]);
        MPI_Reduce(&my_times_ms[idx], &reduced_times_sum[idx], 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&my_tops[idx], &reduced_tops_sum[idx], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    if (run_flags[static_cast<int>(GemmOpIndex::LT_MATMUL_INT8)]) {
        int idx = static_cast<int>(GemmOpIndex::LT_MATMUL_INT8);
        auto result = runGemmTest<GemmOpIndex::LT_MATMUL_INT8, int8_t, int8_t, int32_t, int32_t>
                     (m, n, k, transposeA, transposeB, iterations, opFlops[idx], opNames[idx], rank);
        my_times_ms[idx] = result.first; my_tops[idx] = result.second;
        bool skipped = (fabs(my_times_ms[idx]) < std::numeric_limits<float>::epsilon() && fabs(my_tops[idx]) < std::numeric_limits<double>::epsilon());
        if (verbose && !skipped) printf("| %4d | %-18s | %14.3f | %8.3f |\n", rank, opNames[idx], (iterations > 0 ? my_times_ms[idx]/iterations : 0.0), my_tops[idx]);
        MPI_Reduce(&my_times_ms[idx], &reduced_times_sum[idx], 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&my_tops[idx], &reduced_tops_sum[idx], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    // --- Print Final Aggregated Table (Rank 0) ---
    MPI_Barrier(MPI_COMM_WORLD); // Ensure all ranks finish tests and reductions
    if (rank == 0) {
        if (verbose) {
            printf("|------|--------------------|----------------|----------|\n"); // Footer for verbose table
        }
        // Always print aggregated table header
        printf("--- Aggregated Performance ---\n");
        printf("| Operation          | Avg Time/Iter (ms) | Total T*OPS |\n");
        printf("|--------------------|--------------------|-------------|\n");

        // Iterate through tests and print results from stored arrays
        for (int i = 0; i < NUM_TEST_TYPES; ++i) {
            if (run_flags[i]) { // Only print if test was enabled
                // Check for skip condition (LtMatmul specific, but check generally)
                bool skipped_agg = (fabs(reduced_times_sum[i]) < std::numeric_limits<float>::epsilon() * size &&
                                    fabs(reduced_tops_sum[i]) < std::numeric_limits<double>::epsilon() * size);

                if (skipped_agg && static_cast<GemmOpIndex>(i) == GemmOpIndex::LT_MATMUL_INT8) {
                    printf("| %-18s | %18s | %11s |\n", opNames[i], "Skipped", "N/A");
                } else {
                    double avg_time_all_ranks = (size > 0 && iterations > 0) ? reduced_times_sum[i] / size / iterations : 0.0;
                    printf("| %-18s | %18.3f | %11.3f |\n", opNames[i], avg_time_all_ranks, reduced_tops_sum[i]);
                }
            }
        }
        printf("|--------------------|--------------------|-------------|\n");
    }

    MPI_Finalize();
    return 0;
}

