#include "gemm_test_common.hpp"
#include "gemm_test_internal.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <getopt.h>
#include <limits>

namespace gemm_test::detail {

[[noreturn]] void fail(const char* api,
                       int code,
                       const char* message,
                       const char* file,
                       int line) {
    int initialized = 0;
    int rank = -1;
    MPI_Initialized(&initialized);
    if (initialized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    std::fprintf(stderr,
                 "[MPI Rank %d] %s error %d at %s:%d: %s\n",
                 rank,
                 api,
                 code,
                 file,
                 line,
                 message != nullptr ? message : "unknown");

    if (initialized) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    std::exit(EXIT_FAILURE);
}

static void print_usage(const char* program, int rank) {
    if (rank != 0) {
        return;
    }

    std::fprintf(
        stderr,
        "Usage: %s [--m N --n N --k N --mn N --mk N --nk N --mnk N] "
        "[--transposeA 0/1 --transposeB 0/1] "
        "[--iterations N --warmup N --verbose] "
        "[--dgemm 0/1 --zgemm 0/1 --gemmex 0/1 --ltmatmul 0/1] "
        "[--sm-count N --stream-count N --threshold N --batch-count N "
        "--workspace-mib N --lt-algorithms N]\n",
        program);
}

Options parse_options(int argc, char** argv, int rank) {
    Options options;

    static option long_options[] = {
        {"m", required_argument, nullptr, 'm'},
        {"n", required_argument, nullptr, 'n'},
        {"k", required_argument, nullptr, 'k'},
        {"transposeA", required_argument, nullptr, 'a'},
        {"transposeB", required_argument, nullptr, 'b'},
        {"iterations", required_argument, nullptr, 'i'},
        {"warmup", required_argument, nullptr, 'w'},
        {"verbose", no_argument, nullptr, 'v'},
        {"mn", required_argument, nullptr, '1'},
        {"mk", required_argument, nullptr, '2'},
        {"nk", required_argument, nullptr, '3'},
        {"mnk", required_argument, nullptr, '4'},
        {"dgemm", required_argument, nullptr, 'd'},
        {"zgemm", required_argument, nullptr, 'z'},
        {"gemmex", required_argument, nullptr, 'g'},
        {"ltmatmul", required_argument, nullptr, 'l'},
        {"sm-count", required_argument, nullptr, 's'},
        {"stream-count", required_argument, nullptr, 't'},
        {"threshold", required_argument, nullptr, 'H'},
        {"batch-count", required_argument, nullptr, 'B'},
        {"workspace-mib", required_argument, nullptr, 'W'},
        {"lt-algorithms", required_argument, nullptr, 'A'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0},
    };

    int option_index = 0;
    int option_value = 0;
    while ((option_value = getopt_long(
                argc,
                argv,
                "m:n:k:a:b:i:w:s:t:v1:2:3:4:d:z:g:l:H:B:W:A:h",
                long_options,
                &option_index)) != -1) {
        switch (option_value) {
            case 'm': options.m = std::atoi(optarg); break;
            case 'n': options.n = std::atoi(optarg); break;
            case 'k': options.k = std::atoi(optarg); break;
            case 'a': options.transpose_a = std::atoi(optarg) != 0; break;
            case 'b': options.transpose_b = std::atoi(optarg) != 0; break;
            case 'i': options.iterations = std::atoi(optarg); break;
            case 'w': options.warmup = std::atoi(optarg); break;
            case 's': options.sm_count = std::atoi(optarg); break;
            case 't': options.stream_count = std::atoi(optarg); break;
            case 'v': options.verbose = true; break;
            case '1': options.m = options.n = std::atoi(optarg); break;
            case '2': options.m = options.k = std::atoi(optarg); break;
            case '3': options.n = options.k = std::atoi(optarg); break;
            case '4': options.m = options.n = options.k = std::atoi(optarg); break;
            case 'd':
                options.run[static_cast<int>(Operation::Dgemm)] =
                    std::atoi(optarg) != 0;
                break;
            case 'z':
                options.run[static_cast<int>(Operation::Zgemm)] =
                    std::atoi(optarg) != 0;
                break;
            case 'g':
                options.run[static_cast<int>(Operation::GemmExInt8)] =
                    std::atoi(optarg) != 0;
                break;
            case 'l':
                options.run[static_cast<int>(Operation::LtMatmulInt8)] =
                    std::atoi(optarg) != 0;
                break;
            case 'H': options.size_threshold = std::atoi(optarg); break;
            case 'B': options.batch_count = std::atoi(optarg); break;
            case 'W':
                options.workspace_mib = std::strtoull(optarg, nullptr, 10);
                break;
            case 'A': options.lt_algorithm_count = std::atoi(optarg); break;
            case 'h':
                print_usage(argv[0], rank);
                MPI_Finalize();
                std::exit(EXIT_SUCCESS);
            default:
                print_usage(argv[0], rank);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    if (options.m <= 0 || options.n <= 0 || options.k <= 0 ||
        options.iterations <= 0 || options.warmup < 0 ||
        options.stream_count <= 0 || options.size_threshold < 0 ||
        options.batch_count <= 0 || options.workspace_mib == 0 ||
        options.lt_algorithm_count <= 0) {
        fail("arguments", 1, "invalid option value", __FILE__, __LINE__);
    }

    if (options.size_threshold == 0) {
        options.batch_count = 1;
    }

    return options;
}

DeviceContext initialize_device(int rank,
                                int requested_sm_count,
                                bool use_context_affinity) {
    DeviceContext device;

    if (use_context_affinity) {
        GEMM_TEST_CUDA_DRIVER_CHECK(cuInit(0));
        GEMM_TEST_CUDA_DRIVER_CHECK(cuDeviceGetCount(&device.device_count));
        if (device.device_count <= 0) {
            fail("CUDA Driver",
                 static_cast<int>(CUDA_ERROR_NO_DEVICE),
                 "no CUDA devices",
                 __FILE__,
                 __LINE__);
        }

        device.device_id = rank % device.device_count;
        CUdevice cu_device = 0;
        GEMM_TEST_CUDA_DRIVER_CHECK(
            cuDeviceGet(&cu_device, device.device_id));
        GEMM_TEST_CUDA_DRIVER_CHECK(cuDeviceGetAttribute(
            &device.total_sm_count,
            CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            cu_device));

        device.effective_sm_count =
            requested_sm_count > 0 && requested_sm_count <= device.total_sm_count
                ? requested_sm_count
                : device.total_sm_count;

        CUexecAffinityParam affinity{};
        affinity.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
        affinity.param.smCount.val = device.effective_sm_count;

#if CUDA_VERSION >= 13000
        CUctxCreateParams create_params{};
        create_params.execAffinityParams = &affinity;
        create_params.numExecAffinityParams = 1;
        create_params.cigParams = nullptr;
        GEMM_TEST_CUDA_DRIVER_CHECK(
            cuCtxCreate(&device.context, &create_params, 0, cu_device));
#else
        GEMM_TEST_CUDA_DRIVER_CHECK(
            cuCtxCreate_v3(&device.context, &affinity, 1, 0, cu_device));
#endif
        GEMM_TEST_CUDA_DRIVER_CHECK(cuCtxSetCurrent(device.context));
    } else {
        GEMM_TEST_CUDA_CHECK(cudaGetDeviceCount(&device.device_count));
        if (device.device_count <= 0) {
            fail("CUDA",
                 static_cast<int>(cudaErrorNoDevice),
                 "no CUDA devices",
                 __FILE__,
                 __LINE__);
        }

        device.device_id = rank % device.device_count;
        GEMM_TEST_CUDA_CHECK(cudaSetDevice(device.device_id));

        cudaDeviceProp properties{};
        GEMM_TEST_CUDA_CHECK(
            cudaGetDeviceProperties(&properties, device.device_id));
        device.total_sm_count = properties.multiProcessorCount;
        device.effective_sm_count =
            requested_sm_count > 0 && requested_sm_count <= device.total_sm_count
                ? requested_sm_count
                : device.total_sm_count;
    }

    return device;
}

void destroy_device_context(DeviceContext& device) {
    if (device.context != nullptr) {
        GEMM_TEST_CUDA_DRIVER_CHECK(cuCtxDestroy(device.context));
        device.context = nullptr;
    }
}

void print_individual_result(int rank,
                             const char* operation_name,
                             const OperationResult& result,
                             int iterations) {
    if (result.skipped) {
        return;
    }

    std::printf("| %4d | %-18s | %14.3f | %8.3f |\n",
                rank,
                operation_name,
                result.total_milliseconds / iterations,
                result.tops);
}

}  // namespace gemm_test::detail

namespace gemm_test {

int run_benchmark(int argc, char** argv, bool use_context_affinity) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    detail::Options options = detail::parse_options(argc, argv, rank);
    detail::DeviceContext device = detail::initialize_device(
        rank, options.sm_count, use_context_affinity);

    cudaDeviceProp properties{};
    GEMM_TEST_CUDA_CHECK(
        cudaGetDeviceProperties(&properties, device.device_id));

    if (options.verbose && rank == 0) {
        std::printf("MPI Size: %d, GPUs: %d\n",
                    world_size,
                    device.device_count);
        for (int current_rank = 0; current_rank < world_size; ++current_rank) {
            std::printf("  Rank %d -> GPU %d\n",
                        current_rank,
                        current_rank % device.device_count);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0 && options.verbose) {
        std::printf("---------------------------------------------------------------------\n");
        std::printf(
            "MPI GEMM Test: m=%d, n=%d, k=%d, iterations=%d, ranks=%d\n",
            options.m,
            options.n,
            options.k,
            options.iterations,
            world_size);
        std::printf("Transpose A: %s, Transpose B: %s, SM Count: %d, Stream: %d \n",
                    options.transpose_a ? "Yes" : "No",
                    options.transpose_b ? "Yes" : "No",
                    options.sm_count,
                    options.stream_count);
        std::printf("Batch Threshold: %d, Batch Count: %d\n",
                    options.size_threshold,
                    options.batch_count);
        std::printf("Warmup: %d, Lt Workspace: %zu MiB, Lt Algorithms: %d\n",
                    options.warmup,
                    options.workspace_mib,
                    options.lt_algorithm_count);
        std::printf("Tests: Dgemm:%s Zgemm:%s GemmEx(I8):%s LtMatmul(I8):%s Verbose:%s\n",
                    options.run[static_cast<int>(detail::Operation::Dgemm)] ? "Y" : "N",
                    options.run[static_cast<int>(detail::Operation::Zgemm)] ? "Y" : "N",
                    options.run[static_cast<int>(detail::Operation::GemmExInt8)] ? "Y" : "N",
                    options.run[static_cast<int>(detail::Operation::LtMatmulInt8)] ? "Y" : "N",
                    options.verbose ? "Y" : "N");
        std::printf("Device: %s, SM %d.%d, %d SMs\n",
                    properties.name,
                    properties.major,
                    properties.minor,
                    properties.multiProcessorCount);
        std::printf("CUDA Context: %s\n",
                    use_context_affinity ? "execution affinity enabled"
                                         : "default runtime context");
        std::printf("---------------------------------------------------------------------\n");
        std::printf("--- Individual Rank Performance ---\n");
        std::printf("| Rank | Operation          | Time/Iter (ms) | T*OPS    |\n");
        std::printf("|------|--------------------|----------------|----------|\n");
    }

    const std::array<const char*, detail::kOperationCount> operation_names = {
        "Dgemm", "Zgemm", "GemmEx(int8)", "LtMatmul(int8)"};
    std::array<detail::OperationResult, detail::kOperationCount> local_results{};

    if (options.run[static_cast<int>(detail::Operation::Dgemm)]) {
        local_results[static_cast<int>(detail::Operation::Dgemm)] =
            detail::run_dgemm(options, device.effective_sm_count);
    }
    if (options.run[static_cast<int>(detail::Operation::Zgemm)]) {
        local_results[static_cast<int>(detail::Operation::Zgemm)] =
            detail::run_zgemm(options, device.effective_sm_count);
    }
    if (options.run[static_cast<int>(detail::Operation::GemmExInt8)]) {
        local_results[static_cast<int>(detail::Operation::GemmExInt8)] =
            detail::run_gemmex_int8(options, device.effective_sm_count);
    }
    if (options.run[static_cast<int>(detail::Operation::LtMatmulInt8)]) {
        local_results[static_cast<int>(detail::Operation::LtMatmulInt8)] =
            detail::run_ltmatmul_int8(options, device.effective_sm_count);
    }

    if (options.verbose) {
        for (int current_rank = 0; current_rank < world_size; ++current_rank) {
            if (rank == current_rank) {
                for (int index = 0; index < detail::kOperationCount; ++index) {
                    if (options.run[index]) {
                        detail::print_individual_result(
                            rank,
                            operation_names[index],
                            local_results[index],
                            options.iterations);
                    }
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        if (rank == 0) {
            std::printf("|------|--------------------|----------------|----------|\n");
        }
    }

    std::array<float, detail::kOperationCount> local_times{};
    std::array<double, detail::kOperationCount> local_tops{};
    std::array<float, detail::kOperationCount> reduced_times{};
    std::array<double, detail::kOperationCount> reduced_tops{};

    for (int index = 0; index < detail::kOperationCount; ++index) {
        local_times[index] = local_results[index].total_milliseconds;
        local_tops[index] = local_results[index].tops;
        if (options.run[index]) {
            MPI_Reduce(&local_times[index],
                       &reduced_times[index],
                       1,
                       MPI_FLOAT,
                       MPI_SUM,
                       0,
                       MPI_COMM_WORLD);
            MPI_Reduce(&local_tops[index],
                       &reduced_tops[index],
                       1,
                       MPI_DOUBLE,
                       MPI_SUM,
                       0,
                       MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("--- Aggregated Performance ---\n");
        std::printf("| Operation          | Avg Time/Iter (ms) | Total T*OPS |\n");
        std::printf("|--------------------|--------------------|-------------|\n");

        for (int index = 0; index < detail::kOperationCount; ++index) {
            if (!options.run[index]) {
                continue;
            }

            const bool skipped = local_results[index].skipped ||
                                 (std::fabs(reduced_times[index]) <
                                      std::numeric_limits<float>::epsilon() &&
                                  std::fabs(reduced_tops[index]) <
                                      std::numeric_limits<double>::epsilon());
            if (skipped &&
                (index == static_cast<int>(detail::Operation::GemmExInt8) ||
                 index == static_cast<int>(detail::Operation::LtMatmulInt8))) {
                std::printf("| %-18s | %18s | %11s |\n",
                            operation_names[index],
                            "Skipped",
                            "N/A");
                continue;
            }

            const bool batched =
                (index == static_cast<int>(detail::Operation::GemmExInt8) ||
                 index == static_cast<int>(detail::Operation::LtMatmulInt8)) &&
                detail::should_use_batched_mode(options);
            char display_name[40];
            std::snprintf(display_name,
                          sizeof(display_name),
                          "%s%s",
                          operation_names[index],
                          batched ? " (B)" : "");

            const double average_time =
                static_cast<double>(reduced_times[index]) / world_size /
                options.iterations;
            std::printf("| %-18s | %18.3f | %11.3f |\n",
                        display_name,
                        average_time,
                        reduced_tops[index]);
        }
        std::printf("|--------------------|--------------------|-------------|\n");

        const detail::OperationResult& lt_result =
            local_results[static_cast<int>(detail::Operation::LtMatmulInt8)];
        if (options.run[static_cast<int>(detail::Operation::LtMatmulInt8)] &&
            !lt_result.skipped) {
            std::printf(
                "LtMatmul heuristic: returned %d algorithm(s), selected index %d, workspace %zu MiB.\n",
                lt_result.returned_algorithms,
                lt_result.selected_algorithm,
                options.workspace_mib);
        }
    }

    detail::destroy_device_context(device);
    MPI_Finalize();
    return EXIT_SUCCESS;
}

}  // namespace gemm_test
