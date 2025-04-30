#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda.h>       // Include CUDA Driver API header
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

// --- Error Checking Macros ---

// Error checking macro for CUDA Runtime API
#define CUDA_CHECK(err)                                                        \
    do {                                                                       \
        cudaError_t err_ = (err);                                              \
        if (err_ != cudaSuccess) {                                             \
            int rank_ = -1; int initialized_ = 0;                              \
            MPI_Initialized(&initialized_);                                    \
            if (initialized_) MPI_Comm_rank(MPI_COMM_WORLD, &rank_);           \
            fprintf(stderr, "[MPI Rank %d] CUDA Runtime Error: %s (%d) at %s:%d\n", \
                    rank_, cudaGetErrorString(err_), (int)err_, __FILE__, __LINE__); \
            if (initialized_) MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);         \
            else exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// Error checking macro for CUDA Driver API
#define CUDA_DRIVER_CHECK(err)                                                 \
    do {                                                                       \
        CUresult err_ = (err);                                                 \
        if (err_ != CUDA_SUCCESS) {                                            \
            int rank_ = -1; int initialized_ = 0;                              \
            const char *err_str;                                               \
            cuGetErrorString(err_, &err_str);                                  \
            MPI_Initialized(&initialized_);                                    \
            if (initialized_) MPI_Comm_rank(MPI_COMM_WORLD, &rank_);           \
            fprintf(stderr, "[MPI Rank %d] CUDA Driver Error: %s (%d) at %s:%d\n", \
                    rank_, err_str ? err_str : "Unknown", (int)err_, __FILE__, __LINE__); \
            if (initialized_) MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);         \
            else exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)


// Error checking macro for cuBLAS (covers both v2 and Lt)
#define CUBLAS_CHECK(err)                                                      \
    do {                                                                       \
        cublasStatus_t err_ = (err);                                           \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                   \
            int rank_ = -1; int initialized_ = 0;                              \
            MPI_Initialized(&initialized_);                                    \
            if (initialized_) MPI_Comm_rank(MPI_COMM_WORLD, &rank_);           \
            const char* errorString = "Unknown cuBLAS Error";                  \
            switch(err_) {                                                     \
                case CUBLAS_STATUS_NOT_INITIALIZED: errorString = "NOT_INITIALIZED"; break; \
                case CUBLAS_STATUS_ALLOC_FAILED:    errorString = "ALLOC_FAILED"; break;    \
                case CUBLAS_STATUS_INVALID_VALUE:   errorString = "INVALID_VALUE"; break;  \
                case CUBLAS_STATUS_ARCH_MISMATCH:   errorString = "ARCH_MISMATCH"; break;  \
                case CUBLAS_STATUS_MAPPING_ERROR:   errorString = "MAPPING_ERROR"; break;  \
                case CUBLAS_STATUS_EXECUTION_FAILED: errorString = "EXECUTION_FAILED"; break;\
                case CUBLAS_STATUS_INTERNAL_ERROR:  errorString = "INTERNAL_ERROR"; break; \
                case CUBLAS_STATUS_NOT_SUPPORTED:   errorString = "NOT_SUPPORTED"; break;  \
                case CUBLAS_STATUS_LICENSE_ERROR:   errorString = "LICENSE_ERROR"; break;  \
                default: /* errorString already set */ break;                  \
            }                                                                  \
            fprintf(stderr, "[MPI Rank %d] cuBLAS Error: %s (%d) at %s:%d\n",  \
                    rank_, errorString, static_cast<int>(err_), __FILE__, __LINE__);        \
            if (initialized_) MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);         \
            else exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)


// --- Type Helpers and Enums ---

// Helper to get CUDA data type enum from C++ type
template <typename T> struct CudaDataType;
template <> struct CudaDataType<double>          { static constexpr cudaDataType_t value = CUDA_R_64F; };
template <> struct CudaDataType<float>           { static constexpr cudaDataType_t value = CUDA_R_32F; };
template <> struct CudaDataType<cuDoubleComplex> { static constexpr cudaDataType_t value = CUDA_C_64F; };
template <> struct CudaDataType<cuComplex>       { static constexpr cudaDataType_t value = CUDA_C_32F; };
template <> struct CudaDataType<int8_t>          { static constexpr cudaDataType_t value = CUDA_R_8I;  };
template <> struct CudaDataType<int32_t>         { static constexpr cudaDataType_t value = CUDA_R_32I; };

// Enum to identify the GEMM operation type for dispatch and array indexing
enum class GemmOpIndex {
    DGEMM = 0,
    ZGEMM = 1,
    GEMM_EX_INT8 = 2,
    LT_MATMUL_INT8 = 3,
    COUNT // Number of test types
};
const int NUM_TEST_TYPES = static_cast<int>(GemmOpIndex::COUNT);

// --- Generic GEMM Test Function ---
// Returns: { total_milliseconds, t_ops } for this rank
template <GemmOpIndex OperationTypeIndex,
          typename TypeA, typename TypeB, typename TypeC, typename TypeCompute>
std::pair<float, double> runGemmTest(int m, int n, int k, bool transposeA, bool transposeB, int iterations,
                                     double flopsPerOp,
                                     const char* opName,
                                     int rank, int deviceId,
                                     int requested_sm_count,
                                     int num_streams)
{
    // --- Rest of the setup (LDA, LDB, LDC, LtMatmul constraints check) ---
    int lda = transposeA ? k : m;
    int ldb = transposeB ? n : k;
    int ldc = m;
    if constexpr (OperationTypeIndex == GemmOpIndex::LT_MATMUL_INT8 || OperationTypeIndex == GemmOpIndex::GEMM_EX_INT8) {
        if (lda % 4 != 0 || ldb % 4 != 0 || ldc % 4 != 0) {
            return {0.0f, 0.0};
        }
    }

    // --- Handle Creation (No change here) ---
    cublasHandle_t handle = nullptr;
    cublasLtHandle_t ltHandle = nullptr;
    if constexpr (OperationTypeIndex == GemmOpIndex::LT_MATMUL_INT8) {
        CUBLAS_CHECK(cublasLtCreate(&ltHandle));
    } else {
        CUBLAS_CHECK(cublasCreate(&handle));
    }

    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }
    // Note: cublasSetStream will be called *inside* the loop for v2 API calls.

    // --- Memory Allocation (MODIFIED) ---
    // Use streams[0] for async memory operations for simplicity
    size_t sizeA = static_cast<size_t>(lda) * (transposeA ? m : k) * sizeof(TypeA);
    size_t sizeB = static_cast<size_t>(ldb) * (transposeB ? k : n) * sizeof(TypeB);
    size_t sizeC = static_cast<size_t>(ldc) * n * sizeof(TypeC);
    TypeA *d_A = nullptr; TypeB *d_B = nullptr; TypeC *d_C = nullptr;
    TypeA *h_A = nullptr; TypeB *h_B = nullptr;
    bool alloc_success = true;
    // Use streams[0] for memory ops
    CUDA_CHECK(cudaMallocAsync(&d_A, sizeA, streams[0]));
    CUDA_CHECK(cudaMallocAsync(&d_B, sizeB, streams[0]));
    CUDA_CHECK(cudaMallocAsync(&d_C, sizeC, streams[0]));
    h_A = (TypeA *)malloc(sizeA);
    h_B = (TypeB *)malloc(sizeB);
    if (!h_A || !h_B) {
        fprintf(stderr, "[MPI Rank %d] Error: Failed to allocate host memory\n", rank);
        alloc_success = false;
    }
    if (!alloc_success) { // Cleanup and abort
        if (h_A) free(h_A); if (h_B) free(h_B);
        // Use streams[0] for freeing
        if(d_A) cudaFreeAsync(d_A, streams[0]); if(d_B) cudaFreeAsync(d_B, streams[0]); if(d_C) cudaFreeAsync(d_C, streams[0]);
        if (handle) CUBLAS_CHECK(cublasDestroy(handle)); if (ltHandle) CUBLAS_CHECK(cublasLtDestroy(ltHandle));
        // Destroy streams before abort
        for (int i = 0; i < num_streams; ++i) { cudaStreamDestroy(streams[i]); }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // --- Initialization & Copy ---
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
    // ... initialization loops ...
    // Use streams[0] for sync and copies
    CUDA_CHECK(cudaStreamSynchronize(streams[0])); // Sync allocs on streams[0]
    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, sizeA, cudaMemcpyHostToDevice, streams[0]));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, sizeB, cudaMemcpyHostToDevice, streams[0]));
    CUDA_CHECK(cudaMemsetAsync(d_C, 0, sizeC, streams[0])); // Initialize C on streams[0]

    // --- Setup (Transpose, Scalars, Events, LtMatmul Descriptors) ---
    cublasOperation_t opA = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;
    TypeCompute alpha_compute, beta_compute; // Scalars
    if constexpr (std::is_same_v<TypeCompute, cuDoubleComplex>) { alpha_compute = make_cuDoubleComplex(1.0, 0.0); beta_compute = make_cuDoubleComplex(0.0, 0.0); }
    else if constexpr (std::is_same_v<TypeCompute, int32_t>) { alpha_compute = 1; beta_compute = 0; }
    else if constexpr (std::is_same_v<TypeCompute, double>) { alpha_compute = 1.0; beta_compute = 0.0; }
    else if constexpr (std::is_same_v<TypeCompute, float>) { alpha_compute = 1.0f; beta_compute = 0.0f; }
    else { static_assert(sizeof(TypeCompute) == 0, "Unsupported TypeCompute"); }
    cudaEvent_t start, stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
    cublasLtMatmulDesc_t matmulDesc = nullptr;
    void *workspace = nullptr; size_t workspaceSize = 0;
    if constexpr (OperationTypeIndex == GemmOpIndex::LT_MATMUL_INT8) {
        cudaDataType_t scaleType = CUDA_R_32I; cublasComputeType_t computeType = CUBLAS_COMPUTE_32I;
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CudaDataType<TypeA>::value, lda, transposeA ? m : k, lda));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CudaDataType<TypeB>::value, ldb, transposeB ? k : n, ldb));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, CudaDataType<TypeC>::value, m, n, ldc));
        CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmulDesc, computeType, scaleType));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));
        // Use streams[0] for workspace allocation
        workspaceSize = 1024 * 1024 * 4; CUDA_CHECK(cudaMallocAsync(&workspace, workspaceSize, streams[0]));
    }
    // Use streams[0] to sync memory operations before warm-up
    CUDA_CHECK(cudaStreamSynchronize(streams[0]));

    // --- Warm-up ---
    // Perform warm-up on streams[0]
    if constexpr (OperationTypeIndex == GemmOpIndex::DGEMM) {
        CUBLAS_CHECK(cublasSetStream(handle, streams[0])); // Set stream for v2 API
        CUBLAS_CHECK(cublasDgemm(handle, opA, opB, m, n, k, (const double*)&alpha_compute, d_A, lda, d_B, ldb, (const double*)&beta_compute, d_C, ldc));
    } else if constexpr (OperationTypeIndex == GemmOpIndex::ZGEMM) {
        CUBLAS_CHECK(cublasSetStream(handle, streams[0])); // Set stream for v2 API
        CUBLAS_CHECK(cublasZgemm(handle, opA, opB, m, n, k, (const cuDoubleComplex*)&alpha_compute, d_A, lda, d_B, ldb, (const cuDoubleComplex*)&beta_compute, d_C, ldc));
    } else if constexpr (OperationTypeIndex == GemmOpIndex::GEMM_EX_INT8) {
        CUBLAS_CHECK(cublasSetStream(handle, streams[0])); // Set stream for v2 API
        CUBLAS_CHECK(cublasGemmEx(handle, opA, opB, m, n, k, &alpha_compute, d_A, CudaDataType<TypeA>::value, lda, d_B, CudaDataType<TypeB>::value, ldb, &beta_compute, d_C, CudaDataType<TypeC>::value, ldc, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
    } else if constexpr (OperationTypeIndex == GemmOpIndex::LT_MATMUL_INT8) {
        // Pass streams[0] as argument for Lt API
        CUBLAS_CHECK(cublasLtMatmul(ltHandle, matmulDesc, &alpha_compute, d_A, Adesc, d_B, Bdesc, &beta_compute, d_C, Cdesc, d_C, Cdesc, NULL, workspace, workspaceSize, streams[0]));
    }
    // Sync warm-up on streams[0]
    CUDA_CHECK(cudaStreamSynchronize(streams[0]));

    // --- Timed Execution (MODIFIED) ---
    MPI_Barrier(MPI_COMM_WORLD); // Ensure all ranks start together

    // Record start event on streams[0] *before* launching any timed work
    CUDA_CHECK(cudaEventRecord(start, streams[0]));

    for (int i = 0; i < iterations; i++) {
        // Select stream for this iteration
        cudaStream_t current_stream = streams[i % num_streams];

        // (Conditional GEMM calls remain the same)
        if constexpr (OperationTypeIndex == GemmOpIndex::DGEMM) {
            CUBLAS_CHECK(cublasSetStream(handle, current_stream)); // Set stream for this call
            CUBLAS_CHECK(cublasDgemm(handle, opA, opB, m, n, k, (const double *)&alpha_compute, d_A, lda, d_B, ldb, (const double *)&beta_compute, d_C, ldc));
        } else if constexpr (OperationTypeIndex == GemmOpIndex::ZGEMM) {
            CUBLAS_CHECK(cublasSetStream(handle, current_stream)); // Set stream for this call
            CUBLAS_CHECK(cublasZgemm(handle, opA, opB, m, n, k, (const cuDoubleComplex *)&alpha_compute, d_A, lda, d_B, ldb, (const cuDoubleComplex *)&beta_compute, d_C, ldc));
        } else if constexpr (OperationTypeIndex == GemmOpIndex::GEMM_EX_INT8) {
            CUBLAS_CHECK(cublasSetStream(handle, current_stream)); // Set stream for this call
            CUBLAS_CHECK(cublasGemmEx(handle, opA, opB, m, n, k, &alpha_compute, d_A, CudaDataType<TypeA>::value, lda, d_B, CudaDataType<TypeB>::value, ldb, &beta_compute, d_C, CudaDataType<TypeC>::value, ldc, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
        } else if constexpr (OperationTypeIndex == GemmOpIndex::LT_MATMUL_INT8) {
            // Pass current stream as argument for Lt API
            CUBLAS_CHECK(cublasLtMatmul(ltHandle, matmulDesc, &alpha_compute, d_A, Adesc, d_B, Bdesc, &beta_compute, d_C, Cdesc, d_C, Cdesc, NULL, workspace, workspaceSize, current_stream));
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // --- Results (CORRECTED Timing Logic) ---
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop, streams[0]));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    double total_ops = flopsPerOp * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k) * iterations;
    double avg_time_s = (milliseconds / 1000.0);
    double t_ops = (avg_time_s > 1e-9 && iterations > 0) ? (total_ops / avg_time_s) / 1e12 : 0.0;

    // --- Cleanup (MODIFIED) ---
    free(h_A); free(h_B); // Host memory

    // Use streams[0] for async frees
    CUDA_CHECK(cudaFreeAsync(d_A, streams[0]));
    CUDA_CHECK(cudaFreeAsync(d_B, streams[0]));
    CUDA_CHECK(cudaFreeAsync(d_C, streams[0]));

    if constexpr (OperationTypeIndex == GemmOpIndex::LT_MATMUL_INT8) { // Lt specific device cleanup
        if (Adesc) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Adesc));
        if (Bdesc) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Bdesc));
        if (Cdesc) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Cdesc));
        if (matmulDesc) CUBLAS_CHECK(cublasLtMatmulDescDestroy(matmulDesc));
        // Use streams[0] for workspace free
        if (workspace) CUDA_CHECK(cudaFreeAsync(workspace, streams[0]));
    }

    // Wait for async frees on streams[0] to complete before destroying streams/handles
    CUDA_CHECK(cudaStreamSynchronize(streams[0]));

    // Destroy cuBLAS handles
    if (handle) CUBLAS_CHECK(cublasDestroy(handle));
    if (ltHandle) CUBLAS_CHECK(cublasLtDestroy(ltHandle));

    // Destroy CUDA Events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Destroy all CUDA Streams
    // Replace: CUDA_CHECK(cudaStreamDestroy(stream));
    for (int i = 0; i < num_streams; ++i) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }

    return {milliseconds, t_ops};
}

int main(int argc, char *argv[]) {
    // --- MPI Initialization ---
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // --- CUDA Driver API Initialization ---
    // Must be called before any other Driver API calls
    CUDA_DRIVER_CHECK(cuInit(0));

    // Default values
    int m = 4096, n = 4096, k = 4096, iterations = 100;
    int sm_count = 0; // Default: use all SMs
    int num_streams = 1;
    bool transposeA = true, transposeB = false, verbose = false;
    std::array<bool, NUM_TEST_TYPES> run_flags = {true, false, true, true}; // D, Z, Ex, Lt

    // Option parsing
    static struct option long_options[] = {
        {"m", required_argument, 0, 'm'},
        {"n", required_argument, 0, 'n'},
        {"k", required_argument, 0, 'k'},
        {"transposeA", required_argument, 0, 'a'},
        {"transposeB", required_argument, 0, 'b'},
        {"iterations", required_argument, 0, 'i'},
        {"verbose", no_argument, 0, 'v'},
        {"mn", required_argument, 0, '1'},
        {"mk", required_argument, 0, '2'},
        {"nk", required_argument, 0, '3'},
        {"mnk", required_argument, 0, '4'},
	    {"dgemm", required_argument, 0, 'd'},      
        {"zgemm", required_argument, 0, 'z'},
	    {"gemmex", required_argument, 0, 'g'},
	    {"ltmatmul", required_argument, 0, 'l'},
        {"sm-count", required_argument, 0, 's'},
        {"stream-count", required_argument, 0, 't'},
        {0, 0, 0, 0}
    };
    int opt; int option_index = 0;
    while ((opt = getopt_long(argc, argv, "m:n:k:a:b:i:s:t:v1:2:3:4:d:z:g:l:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'm': m = atoi(optarg); break; 
            case 'n': n = atoi(optarg); break;
            case 'k': k = atoi(optarg); break; 
            case 'a': transposeA = atoi(optarg) != 0; break;
            case 'b': transposeB = atoi(optarg) != 0; break; 
            case 'i': iterations = atoi(optarg); break;
            case 's': sm_count = atoi(optarg); break; // Parse SM count
            case 't': num_streams = atoi(optarg); break; 
            case 'v': verbose = true; break; 
            case '1': m = n = atoi(optarg); break;
            case '2': m = k = atoi(optarg); break; 
            case '3': n = k = atoi(optarg); break;
            case '4': m = n = k = atoi(optarg); break;
            case 'd': run_flags[static_cast<int>(GemmOpIndex::DGEMM)] = (atoi(optarg) != 0); break;
            case 'z': run_flags[static_cast<int>(GemmOpIndex::ZGEMM)] = (atoi(optarg) != 0); break;
            case 'g': run_flags[static_cast<int>(GemmOpIndex::GEMM_EX_INT8)] = (atoi(optarg) != 0); break;
            case 'l': run_flags[static_cast<int>(GemmOpIndex::LT_MATMUL_INT8)] = (atoi(optarg) != 0); break;
            case '?': if (rank == 0) { fprintf(stderr, "Usage: %s [--m|-m] <m> [--n|-n] <n> [--k|-k] <k> [--mn] <m=n> [--mk] <m=k> [--nk] <n=k> [--mnk] <m=n=k> [--transposeA|-a] <0/1> [--transposeB|-b] <0/1> [--iterations|-i] <iterations> [--verbose|-v] [--dgemm|-d] <0/1> [--gemmex|-g] <0/1> [--ltmatmul|-l] <0/1> [--zgemm|-z] <0/1> [--sm-count|-s] <sm_count> [--stream-count|-t] <stream_count>\n", argv[0]); } MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); break;
            default: if (rank == 0) fprintf(stderr, "Unexpected option parsing error.\n"); MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }
    // --- GPU Selection (Runtime API is fine here) ---
    int deviceCount; CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) { if (rank == 0) fprintf(stderr, "Error: No CUDA devices found.\n"); MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); }
    int deviceId = rank % deviceCount;
    // Set device for Runtime API calls *before* runGemmTest (e.g., cudaGetDeviceProperties used inside)
    CUDA_CHECK(cudaSetDevice(deviceId));
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
            printf("Transpose A: %s, Transpose B: %s, SM Count: %d, Stream: %d \n",
                   transposeA ? "Yes" : "No", transposeB ? "Yes" : "No", sm_count, num_streams);
            printf("Tests: Dgemm:%s Zgemm:%s GemmEx(I8):%s LtMatmul(I8):%s Verbose:%s\n",
                   run_flags[0]?"Y":"N", run_flags[1]?"Y":"N", run_flags[2]?"Y":"N", run_flags[3]?"Y":"N", verbose ? "Y":"N");
            printf("---------------------------------------------------------------------\n");
            printf("--- Individual Rank Performance ---\n");
            printf("| Rank | Operation          | Time/Iter (ms) | T*OPS    |\n");
            printf("|------|--------------------|----------------|----------|\n");
        }
    }

    // Define operation names and flops
    const char* opNames[NUM_TEST_TYPES] = {"Dgemm", "Zgemm", "GemmEx(int8)", "LtMatmul(int8)"};
    const double opFlops[NUM_TEST_TYPES] = {2.0, 8.0, 2.0, 2.0};

    // --- Result Storage ---
    std::array<float, NUM_TEST_TYPES> my_times_ms = {0.0f};
    std::array<double, NUM_TEST_TYPES> my_tops = {0.0};
    std::array<float, NUM_TEST_TYPES> reduced_times_sum = {0.0f}; // Only used on Rank 0
    std::array<double, NUM_TEST_TYPES> reduced_tops_sum = {0.0}; // Only used on Rank 0

    // --- Context Creation with Affinity ---
    CUdevice cuDevice;
    CUDA_DRIVER_CHECK(cuDeviceGet(&cuDevice, deviceId)); // Get Driver API device handle

    // Get device properties to find total SM count
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    int total_sm_count = prop.multiProcessorCount;
    int sm_count_to_use = sm_count;

    // Validate requested SM count
    if (sm_count_to_use <= 0 || sm_count_to_use > total_sm_count) {
        if (rank == 0 && sm_count > 0) { // Print warning only once if user specified invalid count
             printf("Warning: Requested SM count (%d) is invalid for GPU %d (Total SMs: %d). Using default (all SMs).\n",
                    sm_count, deviceId, total_sm_count);
        }
        sm_count_to_use = total_sm_count; // Default to all SMs
    }

    if (num_streams <= 0) {
        if (rank == 0) { // Print warning only once if user specified invalid count
             printf("Warning: Requested Stream count (%d) is invalid. Using default: 1.\n",
                    num_streams);
        }
        num_streams = 1; // Default to all SMs
    }

    // --- Run Selected Tests & Reduce ---
    // Pass deviceId and sm_count to runGemmTest
    if (run_flags[static_cast<int>(GemmOpIndex::DGEMM)]) {
        int idx = static_cast<int>(GemmOpIndex::DGEMM);
        auto result = runGemmTest<GemmOpIndex::DGEMM, double, double, double, double>
                      (m, n, k, transposeA, transposeB, iterations, opFlops[idx], opNames[idx], rank, deviceId, sm_count, num_streams); // Pass args
        my_times_ms[idx] = result.first; my_tops[idx] = result.second;
        if (verbose) printf("| %4d | %-18s | %14.3f | %8.3f |\n", rank, opNames[idx], (iterations > 0 ? my_times_ms[idx]/iterations : 0.0), my_tops[idx]);
        if (rank == 0) { reduced_times_sum[idx] = my_times_ms[idx]; reduced_tops_sum[idx] = my_tops[idx]; } // Init rank 0 data for IN_PLACE
        MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &my_times_ms[idx], &reduced_times_sum[idx], 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &my_tops[idx], &reduced_tops_sum[idx], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    if (run_flags[static_cast<int>(GemmOpIndex::ZGEMM)]) {
        int idx = static_cast<int>(GemmOpIndex::ZGEMM);
        auto result = runGemmTest<GemmOpIndex::ZGEMM, cuDoubleComplex, cuDoubleComplex, cuDoubleComplex, cuDoubleComplex>
                      (m, n, k, transposeA, transposeB, iterations, opFlops[idx], opNames[idx], rank, deviceId, sm_count, num_streams); // Pass args
        my_times_ms[idx] = result.first; my_tops[idx] = result.second;
        if (verbose) printf("| %4d | %-18s | %14.3f | %8.3f |\n", rank, opNames[idx], (iterations > 0 ? my_times_ms[idx]/iterations : 0.0), my_tops[idx]);
        if (rank == 0) { reduced_times_sum[idx] = my_times_ms[idx]; reduced_tops_sum[idx] = my_tops[idx]; } // Init rank 0 data for IN_PLACE
        MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &my_times_ms[idx], &reduced_times_sum[idx], 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &my_tops[idx], &reduced_tops_sum[idx], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    if (run_flags[static_cast<int>(GemmOpIndex::GEMM_EX_INT8)]) {
        int idx = static_cast<int>(GemmOpIndex::GEMM_EX_INT8);
        auto result = runGemmTest<GemmOpIndex::GEMM_EX_INT8, int8_t, int8_t, int32_t, int32_t>
                      (m, n, k, transposeA, transposeB, iterations, opFlops[idx], opNames[idx], rank, deviceId, sm_count, num_streams); // Pass args
        my_times_ms[idx] = result.first; my_tops[idx] = result.second;
        if (verbose) printf("| %4d | %-18s | %14.3f | %8.3f |\n", rank, opNames[idx], (iterations > 0 ? my_times_ms[idx]/iterations : 0.0), my_tops[idx]);
        if (rank == 0) { reduced_times_sum[idx] = my_times_ms[idx]; reduced_tops_sum[idx] = my_tops[idx]; } // Init rank 0 data for IN_PLACE
        MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &my_times_ms[idx], &reduced_times_sum[idx], 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &my_tops[idx], &reduced_tops_sum[idx], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    if (run_flags[static_cast<int>(GemmOpIndex::LT_MATMUL_INT8)]) {
        int idx = static_cast<int>(GemmOpIndex::LT_MATMUL_INT8);
        auto result = runGemmTest<GemmOpIndex::LT_MATMUL_INT8, int8_t, int8_t, int32_t, int32_t>
                     (m, n, k, transposeA, transposeB, iterations, opFlops[idx], opNames[idx], rank, deviceId, sm_count, num_streams); // Pass args
        my_times_ms[idx] = result.first; my_tops[idx] = result.second;
        bool skipped = (fabs(my_times_ms[idx]) < std::numeric_limits<float>::epsilon() && fabs(my_tops[idx]) < std::numeric_limits<double>::epsilon());
        if (verbose && !skipped) printf("| %4d | %-18s | %14.3f | %8.3f |\n", rank, opNames[idx], (iterations > 0 ? my_times_ms[idx]/iterations : 0.0), my_tops[idx]);
        if (rank == 0) { reduced_times_sum[idx] = my_times_ms[idx]; reduced_tops_sum[idx] = my_tops[idx]; } // Init rank 0 data for IN_PLACE
        MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &my_times_ms[idx], &reduced_times_sum[idx], 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &my_tops[idx], &reduced_tops_sum[idx], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }


    // --- Print Final Aggregated Table (Rank 0) ---
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        if (verbose) {
            printf("|------|--------------------|----------------|----------|\n"); // Footer for verbose table
        }
        printf("--- Aggregated Performance ---\n");
        printf("| Operation          | Avg Time/Iter (ms) | Total T*OPS |\n");
        printf("|--------------------|--------------------|-------------|\n");

        for (int i = 0; i < NUM_TEST_TYPES; ++i) {
            if (run_flags[i]) {
                bool skipped_agg = (fabs(reduced_times_sum[i]) < std::numeric_limits<float>::epsilon() * size &&
                                    fabs(reduced_tops_sum[i]) < std::numeric_limits<double>::epsilon() * size);

                if (skipped_agg && ((static_cast<GemmOpIndex>(i) == GemmOpIndex::LT_MATMUL_INT8) || (static_cast<GemmOpIndex>(i) == GemmOpIndex::GEMM_EX_INT8))) {
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
