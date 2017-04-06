#include "test_pafl.cuh"
#include "config.cuh"

/* #define RUN_PTEST(NAME, CNAME, PARAM)\ */
/* TEST_CASE( NAME " test set", "[" NAME "]" ) {\ */
/*     SECTION("int: SMALL data set")   {CNAME <int, PARAM> (0.1).run(SMALL_DATA_SET);}\ */
/*     SECTION("int: MEDIUM data set")  {CNAME <int, PARAM>  (0.1).run(MEDIUM_DATA_SET);}\ */
/* } */

RUN_TEST("PAFL", test_pafl, 32);
RUN_TEST("PFL", test_pafl, 1);

/* #define RUN_PERF_PTEST(NAME, CNAME, PARAM)\ */
/* TEST_CASE( NAME " performance test", "[" NAME "][PERF][hide]" ) {\ */
/*     SECTION("int: PERF data set")   {CNAME <int, PARAM> (0.1).run(PERF_DATA_SET, true);}\ */
/* } */

RUN_PERF_TEST("PAFL", test_pafl, 32);
RUN_PERF_TEST("PFL", test_pafl, 1);

RUN_BENCHMARK_TEST("PAFL", test_pafl, 32);
RUN_BENCHMARK_TEST("PAFL_OPT", test_pafl_optimistic, 32);
