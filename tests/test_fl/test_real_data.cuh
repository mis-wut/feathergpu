#pragma once
#include "catch.hpp"
#include "feathergpu/util/ptx.cuh"

#include "test_fl/test_aafl.cuh"
#include "test_fl/test_afl.cuh"
#include "test_base.cuh"
#include "test_fl/test_delta.cuh"
#include "test_utils/test_macros.cuh"
#include "test_fl/test_pafl.cuh"
#include "test_fl/test_delta_aafl.cuh"
#include "test_fl/test_delta_pafl.cuh"

#define RUN_FILE_BENCHMARK_TEST(NAME, CNAME, TPARAM, IPARAM, FILENAME, COMP_PARAMS, SORT)\
TEST_CASE( NAME " real data benchmark test on " FILENAME, "[.][" NAME "][REAL][" FILENAME "]" ) {\
    SECTION("int: BENCHMARK data set")   {\
            CNAME <TPARAM, IPARAM> test;\
            printf(NAME "; ");\
            CHECK(test.run_on_file("real_data_benchmarks/" FILENAME ".bin", COMP_PARAMS, true, SORT)==0);\
    }\
}
