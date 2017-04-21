#include "test_afl.cuh"
#include "config.cuh"

// AFL
RUN_TEST("AFL", test_afl, 32);
RUN_TEST("FL", test_afl, 1);

RUN_PERF_TEST("AFL", test_afl, 32);
RUN_PERF_TEST("FL", test_afl, 1);

RUN_BENCHMARK_TEST("AFL", test_afl, 32);
RUN_BENCHMARK_TEST("FL", test_afl, 1);

// AFL CPU
RUN_TEST("AFL_CPU", test_afl_cpu, 32);
RUN_TEST("FL_CPU", test_afl_cpu, 1);

RUN_PERF_TEST("AFL_CPU", test_afl_cpu, 32);
RUN_PERF_TEST("FL_CPU", test_afl_cpu, 1);

RUN_BENCHMARK_TEST("AFL_CPU", test_afl_cpu, 32);
RUN_BENCHMARK_TEST("FL_CPU", test_afl_cpu, 1);

// RAFL CPU
RUN_TEST("RAFL_CPU", test_afl_random_access_cpu, 32);
RUN_TEST("RFL_CPU", test_afl_random_access_cpu, 1);

// RAFL
RUN_TEST("RAFL", test_afl_random_access, 32);
RUN_TEST("RFL", test_afl_random_access, 1);

RUN_PERF_TEST("RAFL", test_afl_random_access, 32);
RUN_PERF_TEST("RFL", test_afl_random_access, 1);

RUN_BENCHMARK_TEST("RAFL", test_afl_random_access, 32);
RUN_BENCHMARK_TEST("RFL", test_afl_random_access, 1);

