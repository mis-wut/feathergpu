#include "test_real_data.cuh"

RUN_FILE_BENCHMARK_TEST("FL",         test_afl,        int, 1,  "DISCOUNT",        4,  false)
RUN_FILE_BENCHMARK_TEST("AFL",        test_afl,        int, 32, "DISCOUNT",        4,  false)
RUN_FILE_BENCHMARK_TEST("PAFL",       test_pafl,       int, 32, "DISCOUNT",        4,  false)
RUN_FILE_BENCHMARK_TEST("AAFL",       test_aafl,       int, 32, "DISCOUNT",        4,  false)

RUN_FILE_BENCHMARK_TEST("FL",         test_afl,        int, 1,  "QUANTITY",        13, false)
RUN_FILE_BENCHMARK_TEST("AFL",        test_afl,        int, 32, "QUANTITY",        13, false)
RUN_FILE_BENCHMARK_TEST("PAFL",       test_pafl,       int, 32, "QUANTITY",        13, false)
RUN_FILE_BENCHMARK_TEST("AAFL",       test_aafl,       int, 32, "QUANTITY",        13, false)

RUN_FILE_BENCHMARK_TEST("FL",         test_afl,        int, 1,  "PARTKEY",         21, false)
RUN_FILE_BENCHMARK_TEST("AFL",        test_afl,        int, 32, "PARTKEY",         21, false)
RUN_FILE_BENCHMARK_TEST("PAFL",       test_pafl,       int, 32, "PARTKEY",         21, false)
RUN_FILE_BENCHMARK_TEST("AAFL",       test_aafl,       int, 32, "PARTKEY",         21, false)

RUN_FILE_BENCHMARK_TEST("FL",         test_afl,        int, 1,  "sorted_SHIPDATE", 30, false)
RUN_FILE_BENCHMARK_TEST("AAFL",       test_aafl,       int, 32, "sorted_SHIPDATE", 30, false)
RUN_FILE_BENCHMARK_TEST("DELTA_AFL",  test_delta,      int, 32, "sorted_SHIPDATE", 17, false)
RUN_FILE_BENCHMARK_TEST("DELTA_PAFL", test_delta_pafl, int, 32, "sorted_SHIPDATE", 1,  false)
RUN_FILE_BENCHMARK_TEST("DELTA_AAFL", test_delta_aafl, int, 32, "sorted_SHIPDATE", 17, false)

