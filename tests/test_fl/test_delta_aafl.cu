#include "test_delta_aafl.cuh"
#include "config.cuh"

RUN_TEST("DELTA_AAFL", test_delta_aafl, 32)
RUN_PERF_TEST("DELTA_AAFL", test_delta_aafl, 32);
RUN_BENCHMARK_TEST("DELTA_AAFL", test_delta_aafl, 32);
