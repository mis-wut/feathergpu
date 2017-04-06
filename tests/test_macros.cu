#include "catch.hpp"
#include "feathergpu/util/ptx.cuh"
#include <stdio.h>

TEST_CASE("SETNPBITS", "[MACROS]") {
    SECTION("set 1 bit at position")   {
        unsigned int pos, i, value;
        for( pos=0, i=1; pos < 32; pos++, i*=2){
            value = 0;
            SETNPBITS(&value, (unsigned int)1, 1, pos);
            CAPTURE(pos);
            CAPTURE(i);
            CHECK( value == i);
        }
    }

    SECTION("set 1 bit at each position in loop")   {
        unsigned int pos, i, value = 0;
        for( pos=0, i=1; pos < 32; pos++, i*=2){
            SETNPBITS(&value, (unsigned int)1, 1, pos);
            CAPTURE(pos);
            CAPTURE(i);
            CHECK( value == (i << 1 ) - 1);
        }
    }

    SECTION("set 15 in two steps")   {
        unsigned int value = 0;
        SETNPBITS(&value, (unsigned int)3, 2, 0);
        SETNPBITS(&value, (unsigned int)3, 2, 2);
        CHECK( value == 15);

        value = 0;
        SETNPBITS(&value, (unsigned int)1, 1, 0);
        SETNPBITS(&value, (unsigned int)7, 3, 1);
        CHECK( value == 15);
    }
}
