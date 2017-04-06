#include "test_afl_signed_experimental.cuh"
#include "config.cuh"

// AFL_SIGNED
RUN_TEST_SIGNED("AFL_SIGNED", test_afl_signed, 32);
/* RUN_TEST_SIGNED("FL_SIGNED", test_afl_signed, 1); */

/* RUN_PERF_TEST("AFL_SIGNED", test_afl_signed, 32); */
/* RUN_PERF_TEST("FL_SIGNED", test_afl_signed, 1); */

/* RUN_BENCHMARK_TEST("AFL_SIGNED", test_afl_signed, 32); */
/* RUN_BENCHMARK_TEST("FL_SIGNED", test_afl_signed, 1); */
