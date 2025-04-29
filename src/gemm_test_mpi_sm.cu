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
                                     int rank, int deviceId, // Pass device ID
                                     int requested_sm_count) // Pass requested SM count
{
    // --- Handle Creation (must happen *after* context is set) ---
    cublasHandle_t handle = nullptr;
    cublasLtHandle_t ltHandle = nullptr;
    if constexpr (OperationTypeIndex == GemmOpIndex::LT_MATMUL_INT8) {
        CUBLAS_CHECK(cublasLtCreate(&ltHandle));
    } else {
        CUBLAS_CHECK(cublasCreate(&handle));
    }

    // --- Stream Creation (operates within the current context) ---
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    if constexpr (OperationTypeIndex != GemmOpIndex::LT_MATMUL_INT8) {
        CUBLAS_CHECK(cublasSetStream(handle, stream)); // Associate v2 handle with stream
    }

    // --- Rest of the setup (LDA, LDB, LDC, LtMatmul constraints check) ---
    int lda = transposeA ? k : m;
    int ldb = transposeB ? n : k;
    int ldc = m;
    if constexpr (OperationTypeIndex == GemmOpIndex::LT_MATMUL_INT8) {
         if (lda % 4 != 0 || ldb % 4 != 0 || ldc % 4 != 0) {
             // Cleanup before returning skip
             if (ltHandle) CUBLAS_CHECK(cublasLtDestroy(ltHandle));
             CUDA_CHECK(cudaStreamDestroy(stream));
             return {0.0f, 0.0};
         }
    }

    // --- Memory Allocation ---
    size_t sizeA = static_cast<size_t>(lda) * (transposeA ? m : k) * sizeof(TypeA);
    size_t sizeB = static_cast<size_t>(ldb) * (transposeB ? k : n) * sizeof(TypeB);
    size_t sizeC = static_cast<size_t>(ldc) * n * sizeof(TypeC);
    TypeA *d_A = nullptr; TypeB *d_B = nullptr; TypeC *d_C = nullptr;
    TypeA *h_A = nullptr; TypeB *h_B = nullptr;
    bool alloc_success = true;
    CUDA_CHECK(cudaMallocAsync(&d_A, sizeA, stream));
    CUDA_CHECK(cudaMallocAsync(&d_B, sizeB, stream));
    CUDA_CHECK(cudaMallocAsync(&d_C, sizeC, stream));
    h_A = (TypeA *)malloc(sizeA);
    h_B = (TypeB *)malloc(sizeB);
    if (!h_A || !h_B) {
        fprintf(stderr, "[MPI Rank %d] Error: Failed to allocate host memory\n", rank);
        alloc_success = false;
    }
    if (!alloc_success) { // Cleanup and abort
        if (h_A) free(h_A); if (h_B) free(h_B);
        if(d_A) cudaFreeAsync(d_A, stream); if(d_B) cudaFreeAsync(d_B, stream); if(d_C) cudaFreeAsync(d_C, stream);
        if (handle) CUBLAS_CHECK(cublasDestroy(handle)); if (ltHandle) CUBLAS_CHECK(cublasLtDestroy(ltHandle));
        CUDA_CHECK(cudaStreamDestroy(stream));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // --- Initialization & Copy ---
    // (Random initialization code remains the same)
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
    CUDA_CHECK(cudaStreamSynchronize(stream)); // Sync before copy
    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, sizeA, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, sizeB, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(d_C, 0, sizeC, stream)); // Initialize C

    // --- Setup (Transpose, Scalars, Events, LtMatmul Descriptors) ---
    // (This part remains the same as the previous version)
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
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CudaDataType<TypeA>::value, transposeA ? m : k, transposeA ? k : m, lda));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CudaDataType<TypeB>::value, transposeB ? k : n, transposeB ? n : k, ldb));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, CudaDataType<TypeC>::value, m, n, ldc));
        CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmulDesc, computeType, scaleType));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));
        workspaceSize = 1024 * 1024 * 4; CUDA_CHECK(cudaMallocAsync(&workspace, workspaceSize, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream)); // Sync before warm-up

    // --- Warm-up ---
    // (Calls remain the same, using handle or ltHandle as appropriate)
    if constexpr (OperationTypeIndex == GemmOpIndex::DGEMM) { CUBLAS_CHECK(cublasDgemm(handle, opA, opB, m, n, k, (const double*)&alpha_compute, d_A, lda, d_B, ldb, (const double*)&beta_compute, d_C, ldc)); }
    else if constexpr (OperationTypeIndex == GemmOpIndex::ZGEMM) { CUBLAS_CHECK(cublasZgemm(handle, opA, opB, m, n, k, (const cuDoubleComplex*)&alpha_compute, d_A, lda, d_B, ldb, (const cuDoubleComplex*)&beta_compute, d_C, ldc)); }
    else if constexpr (OperationTypeIndex == GemmOpIndex::GEMM_EX_INT8) { cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP; CUBLAS_CHECK(cublasGemmEx(handle, opA, opB, m, n, k, &alpha_compute, d_A, CudaDataType<TypeA>::value, lda, d_B, CudaDataType<TypeB>::value, ldb, &beta_compute, d_C, CudaDataType<TypeC>::value, ldc, CUBLAS_COMPUTE_32I, algo)); }
    else if constexpr (OperationTypeIndex == GemmOpIndex::LT_MATMUL_INT8) { CUBLAS_CHECK(cublasLtMatmul(ltHandle, matmulDesc, &alpha_compute, d_A, Adesc, d_B, Bdesc, &beta_compute, d_C, Cdesc, d_C, Cdesc, NULL, workspace, workspaceSize, stream)); }
    CUDA_CHECK(cudaStreamSynchronize(stream)); // Sync after warm-up

    // --- Timed Execution ---
    MPI_Barrier(MPI_COMM_WORLD);
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; i++) {
        // (Calls remain the same, using handle or ltHandle as appropriate)
        if constexpr (OperationTypeIndex == GemmOpIndex::DGEMM) { CUBLAS_CHECK(cublasDgemm(handle, opA, opB, m, n, k, (const double*)&alpha_compute, d_A, lda, d_B, ldb, (const double*)&beta_compute, d_C, ldc)); }
        else if constexpr (OperationTypeIndex == GemmOpIndex::ZGEMM) { CUBLAS_CHECK(cublasZgemm(handle, opA, opB, m, n, k, (const cuDoubleComplex*)&alpha_compute, d_A, lda, d_B, ldb, (const cuDoubleComplex*)&beta_compute, d_C, ldc)); }
        else if constexpr (OperationTypeIndex == GemmOpIndex::GEMM_EX_INT8) { cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP; CUBLAS_CHECK(cublasGemmEx(handle, opA, opB, m, n, k, &alpha_compute, d_A, CudaDataType<TypeA>::value, lda, d_B, CudaDataType<TypeB>::value, ldb, &beta_compute, d_C, CudaDataType<TypeC>::value, ldc, CUBLAS_COMPUTE_32I, algo)); }
        else if constexpr (OperationTypeIndex == GemmOpIndex::LT_MATMUL_INT8) { CUBLAS_CHECK(cublasLtMatmul(ltHandle, matmulDesc, &alpha_compute, d_A, Adesc, d_B, Bdesc, &beta_compute, d_C, Cdesc, d_C, Cdesc, NULL, workspace, workspaceSize, stream)); }
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    MPI_Barrier(MPI_COMM_WORLD);

    // --- Results ---
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    double total_ops = flopsPerOp * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k) * iterations;
    double avg_time_s = (milliseconds / 1000.0);
    double t_ops = (avg_time_s > 1e-9 && iterations > 0) ? (total_ops / avg_time_s) / 1e12 : 0.0;

    // --- Cleanup ---
    free(h_A); free(h_B); // Host memory
    CUDA_CHECK(cudaFreeAsync(d_A, stream)); CUDA_CHECK(cudaFreeAsync(d_B, stream)); CUDA_CHECK(cudaFreeAsync(d_C, stream)); // Device matrices
    if constexpr (OperationTypeIndex == GemmOpIndex::LT_MATMUL_INT8) { // Lt specific device cleanup
        if (Adesc) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Adesc));
        if (Bdesc) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Bdesc));
        if (Cdesc) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Cdesc));
        if (matmulDesc) CUBLAS_CHECK(cublasLtMatmulDescDestroy(matmulDesc));
        if (workspace) CUDA_CHECK(cudaFreeAsync(workspace, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream)); // Wait for async frees

    // Destroy cuBLAS handles *before* destroying the context they used
    if (handle) CUBLAS_CHECK(cublasDestroy(handle));
    if (ltHandle) CUBLAS_CHECK(cublasLtDestroy(ltHandle));

    // Destroy CUDA Events and Stream
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));

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
    bool transposeA = true, transposeB = false, verbose = false;
    std::array<bool, NUM_TEST_TYPES> run_flags = {true, false, true, true}; // D, Z, Ex, Lt

    // Option parsing
    static struct option long_options[] = {
        {"m", required_argument, 0, 'm'}, {"n", required_argument, 0, 'n'},
        {"k", required_argument, 0, 'k'}, {"transposeA", required_argument, 0, 'a'},
        {"transposeB", required_argument, 0, 'b'}, {"iterations", required_argument, 0, 'i'},
        {"sm-count", required_argument, 0, 's'}, // New option for SM count
        {"verbose", no_argument, 0, 'v'}, {"mn", required_argument, 0, '1'},
        {"mk", required_argument, 0, '2'}, {"nk", required_argument, 0, '3'},
        {"mnk", required_argument, 0, '4'},
        {"dgemm", required_argument, 0, 'd'}, {"zgemm", required_argument, 0, 'z'},
        {"gemmex", required_argument, 0, 'g'}, {"ltmatmul", required_argument, 0, 'l'},
        {0, 0, 0, 0}
    };
    int opt; int option_index = 0;
    // Add 's:' to getopt short options string
    while ((opt = getopt_long(argc, argv, "m:n:k:a:b:i:s:v1:2:3:4:d:z:g:l:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'm': m = atoi(optarg); break; case 'n': n = atoi(optarg); break;
            case 'k': k = atoi(optarg); break; case 'a': transposeA = atoi(optarg) != 0; break;
            case 'b': transposeB = atoi(optarg) != 0; break; case 'i': iterations = atoi(optarg); break;
            case 's': sm_count = atoi(optarg); break; // Parse SM count
            case 'v': verbose = true; break; case '1': m = n = atoi(optarg); break;
            case '2': m = k = atoi(optarg); break; case '3': n = k = atoi(optarg); break;
            case '4': m = n = k = atoi(optarg); break;
            case 'd': run_flags[static_cast<int>(GemmOpIndex::DGEMM)] = (atoi(optarg) != 0); break;
            case 'z': run_flags[static_cast<int>(GemmOpIndex::ZGEMM)] = (atoi(optarg) != 0); break;
            case 'g': run_flags[static_cast<int>(GemmOpIndex::GEMM_EX_INT8)] = (atoi(optarg) != 0); break;
            case 'l': run_flags[static_cast<int>(GemmOpIndex::LT_MATMUL_INT8)] = (atoi(optarg) != 0); break;
            case '?': if (rank == 0) { fprintf(stderr, "Usage: %s [-m M] [-n N] [-k K] [-i ITERS] [-s SM_COUNT] [-a 0|1] [-b 0|1] [-v] [--dgemm 0|1] ...\n", argv[0]); } MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); break;
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
            printf("Transpose A: %s, Transpose B: %s, SM Count: %d (0=default)\n",
                   transposeA ? "Yes" : "No", transposeB ? "Yes" : "No", sm_count); // Show SM count
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
    CUcontext context = nullptr;
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

    // Set execution affinity parameter
    CUexecAffinityParam affinityParam[1]; 
    affinityParam[0].type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
    affinityParam[0].param.smCount.val = sm_count_to_use;

    // Create context with the specified affinity
    // Using cuCtxCreate (v3 is not strictly necessary here unless using specific flags)
    CUDA_DRIVER_CHECK(cuCtxCreate_v3(&context, affinityParam, 1, 0, cuDevice)); // Create context first
    CUDA_DRIVER_CHECK(cuCtxSetCurrent(context)); // Set context as current for this thread

    // --- Run Selected Tests & Reduce ---
    // Pass deviceId and sm_count to runGemmTest
    if (run_flags[static_cast<int>(GemmOpIndex::DGEMM)]) {
        int idx = static_cast<int>(GemmOpIndex::DGEMM);
        auto result = runGemmTest<GemmOpIndex::DGEMM, double, double, double, double>
                      (m, n, k, transposeA, transposeB, iterations, opFlops[idx], opNames[idx], rank, deviceId, sm_count); // Pass args
        my_times_ms[idx] = result.first; my_tops[idx] = result.second;
        if (verbose) printf("| %4d | %-18s | %14.3f | %8.3f |\n", rank, opNames[idx], (iterations > 0 ? my_times_ms[idx]/iterations : 0.0), my_tops[idx]);
        if (rank == 0) { reduced_times_sum[idx] = my_times_ms[idx]; reduced_tops_sum[idx] = my_tops[idx]; } // Init rank 0 data for IN_PLACE
        MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &my_times_ms[idx], &reduced_times_sum[idx], 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &my_tops[idx], &reduced_tops_sum[idx], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    if (run_flags[static_cast<int>(GemmOpIndex::ZGEMM)]) {
        int idx = static_cast<int>(GemmOpIndex::ZGEMM);
        auto result = runGemmTest<GemmOpIndex::ZGEMM, cuDoubleComplex, cuDoubleComplex, cuDoubleComplex, cuDoubleComplex>
                      (m, n, k, transposeA, transposeB, iterations, opFlops[idx], opNames[idx], rank, deviceId, sm_count); // Pass args
        my_times_ms[idx] = result.first; my_tops[idx] = result.second;
        if (verbose) printf("| %4d | %-18s | %14.3f | %8.3f |\n", rank, opNames[idx], (iterations > 0 ? my_times_ms[idx]/iterations : 0.0), my_tops[idx]);
        if (rank == 0) { reduced_times_sum[idx] = my_times_ms[idx]; reduced_tops_sum[idx] = my_tops[idx]; } // Init rank 0 data for IN_PLACE
        MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &my_times_ms[idx], &reduced_times_sum[idx], 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &my_tops[idx], &reduced_tops_sum[idx], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    if (run_flags[static_cast<int>(GemmOpIndex::GEMM_EX_INT8)]) {
        int idx = static_cast<int>(GemmOpIndex::GEMM_EX_INT8);
        auto result = runGemmTest<GemmOpIndex::GEMM_EX_INT8, int8_t, int8_t, int32_t, int32_t>
                      (m, n, k, transposeA, transposeB, iterations, opFlops[idx], opNames[idx], rank, deviceId, sm_count); // Pass args
        my_times_ms[idx] = result.first; my_tops[idx] = result.second;
        if (verbose) printf("| %4d | %-18s | %14.3f | %8.3f |\n", rank, opNames[idx], (iterations > 0 ? my_times_ms[idx]/iterations : 0.0), my_tops[idx]);
        if (rank == 0) { reduced_times_sum[idx] = my_times_ms[idx]; reduced_tops_sum[idx] = my_tops[idx]; } // Init rank 0 data for IN_PLACE
        MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &my_times_ms[idx], &reduced_times_sum[idx], 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &my_tops[idx], &reduced_tops_sum[idx], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    if (run_flags[static_cast<int>(GemmOpIndex::LT_MATMUL_INT8)]) {
        int idx = static_cast<int>(GemmOpIndex::LT_MATMUL_INT8);
        auto result = runGemmTest<GemmOpIndex::LT_MATMUL_INT8, int8_t, int8_t, int32_t, int32_t>
                     (m, n, k, transposeA, transposeB, iterations, opFlops[idx], opNames[idx], rank, deviceId, sm_count); // Pass args
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

    // Destroy the CUDA context
    CUDA_DRIVER_CHECK(cuCtxDestroy(context));

    MPI_Finalize();
    return 0;
}
