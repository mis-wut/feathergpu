#include "test_aafl.cuh"
#include "config.cuh"

RUN_TEST("AAFL", test_aafl, 32);
RUN_PERF_TEST("AAFL", test_aafl, 32);
RUN_BENCHMARK_TEST("AAFL", test_aafl, 32);

RUN_TEST("AAFL_OPT", test_aafl_optimistic, 32);
RUN_PERF_TEST("AAFL_OPT", test_aafl_optimistic, 32);
RUN_BENCHMARK_TEST("AAFL_OPT", test_aafl_optimistic, 32);

RUN_TEST("AAFL_PES", test_aafl_pesymistic, 32);
RUN_PERF_TEST("AAFL_PES", test_aafl_pesymistic, 32);
RUN_BENCHMARK_TEST("AAFL_PES", test_aafl_pesymistic, 32);
