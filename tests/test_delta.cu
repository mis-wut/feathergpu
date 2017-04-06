#include "test_delta.cuh"
#include "config.cuh"

RUN_TEST("DELTA_AFL", test_delta, 32)
RUN_PERF_TEST("DELTA_AFL", test_delta, 32);
RUN_BENCHMARK_TEST("DELTA_AFL", test_delta, 32);

