#include "gemm_test_common.hpp"

int main(int argc, char** argv) {
    return gemm_test::run_benchmark(argc, argv, false);
}
