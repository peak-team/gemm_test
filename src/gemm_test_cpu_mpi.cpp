#include <iostream>
#include <vector>
#include <chrono>
#include <mpi.h>
#include <getopt.h>
#include <string.h>
#include <stdlib.h>
#include <numeric>

// External C function declaration for dgemm. The name might have a trailing
// underscore depending on the BLAS/Fortran compiler name mangling scheme.
extern "C" {
    void dgemm_(const char* transa, const char* transb, const int* m, const int* n,
                const int* k, const double* alpha, const double* a, const int* lda,
                const double* b, const int* ldb, const double* beta, double* c, const int* ldc);
}

// --- Helper Functions ---

// Prints usage information
void print_usage(const char* prog_name, int rank) {
    if (rank == 0) {
        fprintf(stderr, "Usage: %s [options]\n", prog_name);
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "  -m, --m <val>            Set matrix rows M (default: 4096)\n");
        fprintf(stderr, "  -n, --n <val>            Set matrix columns N (default: 4096)\n");
        fprintf(stderr, "  -k, --k <val>            Set matrix inner dimension K (default: 4096)\n");
        fprintf(stderr, "  --mnk <val>              Set M, N, and K to the same value\n");
        fprintf(stderr, "  -a, --transposeA <0|1>   Transpose matrix A (default: 0)\n");
        fprintf(stderr, "  -b, --transposeB <0|1>   Transpose matrix B (default: 0)\n");
        fprintf(stderr, "  -i, --iterations <val>   Number of iterations (default: 10)\n");
        fprintf(stderr, "  -v, --verbose            Enable verbose output (shows individual rank performance)\n");
        fprintf(stderr, "  -h, --help               Show this help message\n");
    }
}

// --- Main Test Function ---
// Returns a pair: { total_milliseconds, tflops }
std::pair<float, double> runDgemmTest(int m, int n, int k, bool transposeA, bool transposeB, int iterations, int rank) {

    // --- DGEMM parameters ---
    // Note: C++ is row-major, BLAS is column-major.
    // To compute C = A * B (row-major), we ask BLAS to compute C' = B' * A' (column-major).
    // This is equivalent. We pass dimensions as (n, m, k) and swap matrix pointers.
    char transa_char = transposeB ? 'T' : 'N'; // B becomes the first matrix
    char transb_char = transposeA ? 'T' : 'N'; // A becomes the second matrix
    double alpha = 1.0;
    double beta = 0.0;

    // Leading dimensions for the original row-major matrices
    int lda_orig = transposeA ? m : k;
    int ldb_orig = transposeB ? k : n;
    int ldc_orig = n;

    // --- Allocate and Initialize Host Matrices ---
    // Sizes are based on original (m,n,k) and transpose flags
    size_t sizeA = static_cast<size_t>(m) * k;
    size_t sizeB = static_cast<size_t>(k) * n;
    size_t sizeC = static_cast<size_t>(m) * n;

    std::vector<double> h_A(sizeA);
    std::vector<double> h_B(sizeB);
    std::vector<double> h_C(sizeC);

    // Initialize with some values
    // Using rank in seed for different random numbers per process
    srand(time(NULL) + rank);
    for(size_t i = 0; i < sizeA; ++i) h_A[i] = static_cast<double>((rand() % 1000 - 500) / 500.0);    
    for(size_t i = 0; i < sizeB; ++i) h_B[i] = static_cast<double>((rand() % 1000 - 500) / 500.0);

    // --- Warm-up Run ---
    dgemm_(&transa_char, &transb_char, &n, &m, &k, &alpha, h_B.data(), &ldb_orig, h_A.data(), &lda_orig, &beta, h_C.data(), &ldc_orig);


    // --- Timed Execution ---
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    for (int i = 0; i < iterations; ++i) {
        dgemm_(&transa_char, &transb_char, &n, &m, &k, &alpha, h_B.data(), &ldb_orig, h_A.data(), &lda_orig, &beta, h_C.data(), &ldc_orig);
    }

    double end_time = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);

    // --- Results Calculation ---
    float total_milliseconds = static_cast<float>((end_time - start_time) * 1000.0);
    double avg_time_s = (end_time - start_time);

    // FLOPS calculation: 2*m*n*k operations for one GEMM
    double total_ops = 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k) * iterations;
    double tflops = (avg_time_s > 1e-9 && iterations > 0) ? (total_ops / avg_time_s) / 1e12 : 0.0;
    return {total_milliseconds, tflops};
}


int main(int argc, char *argv[]) {
    // --- MPI Initialization ---
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // --- Default Parameters ---
    int m = 4096, n = 4096, k = 4096, iterations = 10;
    bool transposeA = false, transposeB = false, verbose = false;

    // --- Command-Line Argument Parsing ---
    static struct option long_options[] = {
        {"m",          required_argument, 0, 'm'},
        {"n",          required_argument, 0, 'n'},
        {"k",          required_argument, 0, 'k'},
        {"transposeA", required_argument, 0, 'a'},
        {"transposeB", required_argument, 0, 'b'},
        {"iterations", required_argument, 0, 'i'},
        {"verbose",    no_argument,       0, 'v'},
        {"help",       no_argument,       0, 'h'},
        {"mnk",        required_argument, 0, '1'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "m:n:k:a:b:i:vh1:", long_options, NULL)) != -1) {
        switch (opt) {
            case 'm': m = atoi(optarg); break;
            case 'n': n = atoi(optarg); break;
            case 'k': k = atoi(optarg); break;
            case 'a': transposeA = atoi(optarg) != 0; break;
            case 'b': transposeB = atoi(optarg) != 0; break;
            case 'i': iterations = atoi(optarg); break;
            case 'v': verbose = true; break;
            case '1': m = n = k = atoi(optarg); break;
            case 'h': print_usage(argv[0], rank); MPI_Finalize(); return 0;
            case '?': // getopt_long already printed an error message.
                      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); break;
            default:  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }
    // --- Print Header (Rank 0 only) ---
    if (rank == 0) {
        printf("---------------------------------------------------------------------\n");
        printf("MPI DGEMM Test: m=%d, n=%d, k=%d, iterations=%d, ranks=%d\n", m, n, k, iterations, size);
        printf("Transpose A: %s, Transpose B: %s\n", transposeA ? "Yes" : "No", transposeB ? "Yes" : "No");
        printf("---------------------------------------------------------------------\n");
        if (verbose) {
            printf("--- Individual Rank Performance ---\n");
            printf("| Rank | Operation | Time/Iter (ms) | TFLOPS   |\n");
            printf("|------|-----------|----------------|----------|\n");
        }
    }

    // --- Run Test and Gather Results ---
    auto result = runDgemmTest(m, n, k, transposeA, transposeB, iterations, rank);
    float my_time_ms = result.first;
    double my_tflops = result.second;

    if (verbose) {
        // Each rank prints its own result. We need a barrier to avoid jumbled output.
        for (int r = 0; r < size; ++r) {
            if (rank == r) {
                printf("| %4d | DGEMM     | %14.3f | %8.3f |\n", rank, (iterations > 0 ? my_time_ms / iterations : 0.0), my_tflops);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    // --- Reduce Results for Aggregated Summary ---
    double total_tflops;
    MPI_Reduce(&my_tflops, &total_tflops, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);             
    float total_time_ms;
    MPI_Reduce(&my_time_ms, &total_time_ms, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);


    // --- Print Final Aggregated Table (Rank 0 only) ---
    if (rank == 0) {
        if (verbose) {
            printf("|------|-----------|----------------|----------|\n"); // Footer for verbose table
        }
        printf("--- Aggregated Performance ---\n");
        printf("| Operation | Avg Time/Iter (ms) | Total TFLOPS |\n");
        printf("|-----------|--------------------|--------------|\n");
        double avg_time_all_ranks = (size > 0 && iterations > 0) ? (total_time_ms / size / iterations) : 0.0;
        printf("| DGEMM     | %18.3f | %12.3f |\n", avg_time_all_ranks, total_tflops);
        printf("|-----------|--------------------|--------------|\n");
    }

    // --- Finalize ---
    MPI_Finalize();
    return 0;
}
