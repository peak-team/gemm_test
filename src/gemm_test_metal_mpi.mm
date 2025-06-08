#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <string>
#include <stdexcept>
#include <getopt.h>
#include <array>
#include <type_traits>

// --- Include MPI Header ---
#include <mpi.h>

// --- Use Objective-C Headers for Apple Frameworks ---
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

// Use __fp16 for float16/half type. It is supported by Clang.
typedef __fp16 half;

// Enum for test types (remains C++)
enum class GemmOpIndex {
    F32_F32_F32 = 0,
    F16_F16_F16 = 1,
    F16_F16_F32 = 2,
    I8_I8_F32   = 3,
    I8_I8_F16   = 4,
    COUNT
};
const int NUM_TEST_TYPES = static_cast<int>(GemmOpIndex::COUNT);

// Helper to get MPSDataType enum from C++ type
template <typename T> struct MpsDataType;
template <> struct MpsDataType<float>   { static constexpr MPSDataType value = MPSDataTypeFloat32; };
template <> struct MpsDataType<half>    { static constexpr MPSDataType value = MPSDataTypeFloat16; };
template <> struct MpsDataType<int8_t>  { static constexpr MPSDataType value = MPSDataTypeInt8; };


// --- Generic GEMM Test Function  ---
template <typename TypeA, typename TypeB, typename TypeC>
std::pair<float, double> runGemmTest(
    id<MTLDevice> device,
    int m, int n, int k,
    bool transposeA, bool transposeB,
    int iterations,
    double flopsPerOp,
    const char* opName,
    int rank)
{
    @autoreleasepool {
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        if (!commandQueue) {
            std::cerr << "[MPI Rank " << rank << "] Failed to create command queue." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        size_t rowsA = transposeA ? k : m;
        size_t colsA = transposeA ? m : k;
        size_t rowsB = transposeB ? n : k;
        size_t colsB = transposeB ? k : n;
        
        MPSDataType mpsTypeA = (std::is_same_v<TypeA, float>) ? MPSDataTypeFloat32 : MPSDataTypeInt8;
        MPSDataType mpsTypeB = (std::is_same_v<TypeB, float>) ? MPSDataTypeFloat32 : MPSDataTypeInt8;
        // This logic now correctly handles float output for the int8 case
        MPSDataType mpsTypeC = (std::is_same_v<TypeC, float>) ? MPSDataTypeFloat32 : MPSDataTypeInt32;

        MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:rowsA columns:colsA rowBytes:colsA * sizeof(TypeA) dataType:MpsDataType<TypeA>::value];
        MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:rowsB columns:colsB rowBytes:colsB * sizeof(TypeB) dataType:MpsDataType<TypeB>::value];
        MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n rowBytes:n * sizeof(TypeC) dataType:MpsDataType<TypeC>::value];

        size_t sizeA = rowsA * colsA * sizeof(TypeA);
        size_t sizeB = rowsB * colsB * sizeof(TypeB);
        size_t sizeC = m * n * sizeof(TypeC);

        id<MTLBuffer> bufferA = [device newBufferWithLength:sizeA options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [device newBufferWithLength:sizeB options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [device newBufferWithLength:sizeC options:MTLResourceStorageModeShared];

        TypeA* h_A = static_cast<TypeA*>([bufferA contents]);
        TypeB* h_B = static_cast<TypeB*>([bufferB contents]);
        for (size_t i = 0; i < sizeA / sizeof(TypeA); ++i) {
            if constexpr (std::is_same_v<TypeA, float>) { h_A[i] = (rand() % 1000 - 500) / 500.0f; }
            else if constexpr (std::is_same_v<TypeA, half>) { h_A[i] = (half)((rand() % 1000 - 500) / 500.0f); }
            else if constexpr (std::is_same_v<TypeA, int8_t>) { h_A[i] = static_cast<int8_t>(rand() % 200 - 100); }
        }
        for (size_t i = 0; i < sizeB / sizeof(TypeB); ++i) {
            if constexpr (std::is_same_v<TypeB, float>) { h_B[i] = (rand() % 1000 - 500) / 500.0f; }
            else if constexpr (std::is_same_v<TypeB, half>) { h_B[i] = (half)((rand() % 1000 - 500) / 500.0f); }
            else if constexpr (std::is_same_v<TypeB, int8_t>) { h_B[i] = static_cast<int8_t>(rand() % 200 - 100); }
        }

        MPSMatrixMultiplication *matmulKernel = [[MPSMatrixMultiplication alloc] initWithDevice:device transposeLeft:transposeA transposeRight:transposeB resultRows:m resultColumns:n interiorColumns:k alpha:1.0 beta:0.0];

        id<MTLCommandBuffer> warmupCommandBuffer = [commandQueue commandBuffer];
        MPSMatrix *matrixA_w = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
        MPSMatrix *matrixB_w = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
        MPSMatrix *matrixC_w = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];
        [matmulKernel encodeToCommandBuffer:warmupCommandBuffer leftMatrix:matrixA_w rightMatrix:matrixB_w resultMatrix:matrixC_w];
        [warmupCommandBuffer commit];
        [warmupCommandBuffer waitUntilCompleted];

        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            MPSMatrix *matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
            MPSMatrix *matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
            MPSMatrix *matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];
            [matmulKernel encodeToCommandBuffer:commandBuffer leftMatrix:matrixA rightMatrix:matrixB resultMatrix:matrixC];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        auto stop = std::chrono::high_resolution_clock::now();
        
        float milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0f;
        double total_ops = flopsPerOp * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k) * iterations;
        double avg_time_s = (milliseconds / 1000.0);
        double t_ops = (avg_time_s > 1e-9 && iterations > 0) ? (total_ops / avg_time_s) / 1e12 : 0.0;
        
        return {milliseconds, t_ops};
    }
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        if (rank == 0) std::cerr << "Error: No Metal devices found." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int m = 4096, n = 4096, k = 4096, iterations = 100;
    bool transposeA = true, transposeB = false, verbose = false;
    std::array<bool, NUM_TEST_TYPES> run_flags;
    run_flags.fill(true); // Run all tests by default
    
    static struct option long_options[] = { {"m", required_argument, 0, 'm'}, {"n", required_argument, 0, 'n'}, {"k", required_argument, 0, 'k'}, {"transposeA", required_argument, 0, 'a'}, {"transposeB", required_argument, 0, 'b'}, {"iterations", required_argument, 0, 'i'}, {"verbose", no_argument, 0, 'v'}, {"mn", required_argument, 0, '1'}, {"mk", required_argument, 0, '2'}, {"nk", required_argument, 0, '3'}, {"mnk", required_argument, 0, '4'}, 
        {"f32f32f32", required_argument, 0, 5},
        {"f16f16f16", required_argument, 0, 6},
        {"f16f16f32", required_argument, 0, 7},
        {"i8i8f32",   required_argument, 0, 8},
        {"i8i8f16",   required_argument, 0, 9},
         {0, 0, 0, 0} };
    int opt; int option_index = 0;
    while ((opt = getopt_long(argc, argv, "m:n:k:a:b:i:v1:2:3:4:5:6:7:8:9:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'm': m = atoi(optarg); break; case 'n': n = atoi(optarg); break; case 'k': k = atoi(optarg); break;
            case 'a': transposeA = atoi(optarg) != 0; break; case 'b': transposeB = atoi(optarg) != 0; break;
            case 'i': iterations = atoi(optarg); break; case 'v': verbose = true; break; case '1': m = n = atoi(optarg); break;
            case '2': m = k = atoi(optarg); break; case '3': n = k = atoi(optarg); break; case '4': m = n = k = atoi(optarg); break;
            case 5: run_flags[0] = (atoi(optarg) != 0); break;
            case 6: run_flags[1] = (atoi(optarg) != 0); break;
            case 7: run_flags[2] = (atoi(optarg) != 0); break;
            case 8: run_flags[3] = (atoi(optarg) != 0); break;
            case 9: run_flags[4] = (atoi(optarg) != 0); break;
            case '?': if (rank == 0) fprintf(stderr, "Usage: %s ...\n", argv[0]); MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); break;
            default: if (rank == 0) fprintf(stderr, "Unexpected option parsing error.\n"); MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    srand(time(NULL) + rank);

    if (m <= 0 || n <= 0 || k <= 0 || iterations <= 0) {
        if (rank == 0) std::cerr << "Error: m, n, k, and iterations must be positive." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    
    if (rank == 0) {
        std::cout << "Using Metal Device: " << [device.name UTF8String] << std::endl;
        if (verbose) {
            printf("---------------------------------------------------------------------\n");
            printf("MPI+Metal GEMM Test: m=%d, n=%d, k=%d, iterations=%d, ranks=%d\n", m, n, k, iterations, size);
            printf("Transpose A: %s, Transpose B: %s\n", transposeA ? "Yes" : "No", transposeB ? "Yes" : "No");
            printf("Tests: F32_F32_F32:%s F16_F16_F16:%s F16_F16_F32:%s I8_I8_F32:%s I8_I8_F16:%s\n", run_flags[0]?"Y":"N", run_flags[1]?"Y":"N", run_flags[2]?"Y":"N", run_flags[3]?"Y":"N", run_flags[4]?"Y":"N");
            printf("---------------------------------------------------------------------\n");
            printf("--- Individual Rank Performance ---\n");
            printf("| Rank | Operation        | Time/Iter (ms) | T*OPS      |\n");
            printf("|------|------------------|----------------|------------|\n");
        }
    }

    const char* opNames[NUM_TEST_TYPES] = {
        "F32_F32_F32", "F16_F16_F16", "F16_F16_F32", "I8_I8_F32", "I8_I8_F16"
    };
    const double opFlops[NUM_TEST_TYPES] = {2.0, 2.0, 2.0, 2.0, 2.0};
    std::array<float, NUM_TEST_TYPES> my_times_ms = {0.0f};
    std::array<double, NUM_TEST_TYPES> my_tops = {0.0};
    std::array<float, NUM_TEST_TYPES> reduced_times_sum = {0.0f};
    std::array<double, NUM_TEST_TYPES> reduced_tops_sum = {0.0};
    
    for (int i = 0; i < NUM_TEST_TYPES; ++i) {
        if (run_flags[i]) {
            std::pair<float, double> result;
            GemmOpIndex op_idx = static_cast<GemmOpIndex>(i);

            if (op_idx == GemmOpIndex::F32_F32_F32) result = runGemmTest<float, float, float>(device, m, n, k, transposeA, transposeB, iterations, opFlops[i], opNames[i], rank);
            else if (op_idx == GemmOpIndex::F16_F16_F16) result = runGemmTest<half, half, half>(device, m, n, k, transposeA, transposeB, iterations, opFlops[i], opNames[i], rank);
            else if (op_idx == GemmOpIndex::F16_F16_F32) result = runGemmTest<half, half, float>(device, m, n, k, transposeA, transposeB, iterations, opFlops[i], opNames[i], rank);
            else if (op_idx == GemmOpIndex::I8_I8_F32)   result = runGemmTest<int8_t, int8_t, float>(device, m, n, k, transposeA, transposeB, iterations, opFlops[i], opNames[i], rank);
            else if (op_idx == GemmOpIndex::I8_I8_F16)   result = runGemmTest<int8_t, int8_t, half>(device, m, n, k, transposeA, transposeB, iterations, opFlops[i], opNames[i], rank);
            
            my_times_ms[i] = result.first;
            my_tops[i] = result.second;
            
            if (verbose) {
                printf("| %4d | %-16s | %14.3f | %10.3f |\n", rank, opNames[i], (iterations > 0 ? my_times_ms[i]/iterations : 0.0), my_tops[i]);
            }
            MPI_Reduce(&my_times_ms[i], &reduced_times_sum[i], 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&my_tops[i], &reduced_tops_sum[i], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        if (verbose) printf("|------|------------------|----------------|------------|\n");
        printf("\n--- Aggregated Performance ---\n");
        printf("| Operation        | Avg Time/Iter (ms) | Total T*OPS  |\n");
        printf("|------------------|--------------------|--------------|\n");
        for (int i = 0; i < NUM_TEST_TYPES; ++i) {
            if (run_flags[i]) {
                double avg_time_all_ranks = (size > 0 && iterations > 0) ? reduced_times_sum[i] / size / iterations : 0.0;
                printf("| %-16s | %18.3f | %12.3f |\n", opNames[i], avg_time_all_ranks, reduced_tops_sum[i]);
            }
        }
        printf("|------------------|--------------------|--------------|\n");
    }

    MPI_Finalize();
    return 0;
}
