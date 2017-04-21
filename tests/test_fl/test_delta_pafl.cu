#include "test_delta_pafl.cuh"
#include "config.cuh"

RUN_TEST("DELTA_PAFL", test_delta_pafl, 32)
RUN_PERF_TEST("DELTA_PAFL", test_delta_pafl, 32);
RUN_BENCHMARK_TEST("DELTA_PAFL", test_delta_pafl, 32);
