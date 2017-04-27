#include "data.cuh"
#include "feathergpu/util/ptx.cuh"
#include <limits>       // std::numeric_limits


static unsigned long __xorshf96_x=123456789, __xorshf96_y=362436069, __xorshf96_z=521288629;

int compare_binary_file(const void *a, const void *b) {
  return -(*(int *)a - *(int *)b);
}

void init_random_generator(){
    srand (time(NULL));
    __xorshf96_x=(unsigned long) rand();
    __xorshf96_y=(unsigned long) rand();
    __xorshf96_z=(unsigned long) rand();
}

inline unsigned long xorshf96(void) {          //period 2^96-1
// This is only for test purposes so it is optimized for speed (true randomness is not needed)
    unsigned long t;
        __xorshf96_x ^= __xorshf96_x << 16;
        __xorshf96_x ^= __xorshf96_x >> 5;
        __xorshf96_x ^= __xorshf96_x << 1;

        t = __xorshf96_x;
        __xorshf96_x = __xorshf96_y;
        __xorshf96_y = __xorshf96_z;
        __xorshf96_z = t ^ __xorshf96_x ^ __xorshf96_y;

        return __xorshf96_z;
}

template <typename T, typename X>
void __inner_big_random_block( unsigned long size, X mask, T *data)
{
    init_random_generator();
    for (unsigned long i = 0; i < size; i++)
        data[i] = xorshf96() & mask;
}

template <typename T>
void big_random_block( unsigned long size, int limit_bits, T *data)
{
    T mask = NBITSTOMASK(limit_bits);
    __inner_big_random_block(size, mask, data);
}

template <>
void big_random_block( unsigned long size, int limit_bits, unsigned long *data)
{
    unsigned long mask = LNBITSTOMASK(limit_bits);
    __inner_big_random_block(size, mask, data);
}

template <>
void big_random_block( unsigned long size, int limit_bits, long *data)
{
    unsigned long mask = LNBITSTOMASK(limit_bits);
    __inner_big_random_block(size, mask, data);
}

template <typename T, typename X>
void __inner_big_random_block_with_diff_in_abs_radious_values( unsigned long size, X mask, T *data)
{
    /* init_random_generator(); */
    /* data[0] = std::numeric_limits<T>::max() - 1; */
    data[0] = mask;
    T v = 0;
    for (unsigned long i = 1; i < size; i++){
        /* v = (xorshf96() & mask); */
        if( ((long)data[i-1] - (long)v) >= 0 ){ // ensure that data does not go below 0
            if(i%2)
                data[i] = mask;
            else
                data[i] = 0;
        } else {
            data[i] = 0;
        }
    }
}

template <typename T>
void big_random_block_with_diff_in_abs_radious_values( unsigned long size, int limit_bits, T *data)
{
    T mask = NBITSTOMASK(limit_bits);
    __inner_big_random_block_with_diff_in_abs_radious_values(size, mask, data);
}

template <typename T, typename X>
void __inner_big_random_block_with_decreasing_values( unsigned long size, X mask, T *data)
{
    init_random_generator();
    data[0] = std::numeric_limits<T>::max() - 1;
    T v = 0;
    for (unsigned long i = 1; i < size; i++){
        v = (xorshf96() & mask);
        if( ((long)data[i-1] - (long)v) >= 0 ){ // ensure that data does not go below 0
            data[i] = data[i-1] - v;
        } else {
            data[i] = 0;
        }
    }
}

template <typename T>
void big_random_block_with_decreasing_values( unsigned long size, int limit_bits, T *data)
{
    T mask = NBITSTOMASK(limit_bits);
    __inner_big_random_block_with_decreasing_values(size, mask, data);
}

template <>
void big_random_block_with_decreasing_values( unsigned long size, int limit_bits, unsigned long *data)
{
    unsigned long mask = LNBITSTOMASK(limit_bits);
    __inner_big_random_block_with_decreasing_values(size, mask, data);
}

template <>
void big_random_block_with_decreasing_values( unsigned long size, int limit_bits, long *data)
{
    unsigned long mask = LNBITSTOMASK(limit_bits);
    __inner_big_random_block_with_decreasing_values(size, mask, data);
}


template <typename T, typename X>
void __inner_big_random_block_with_decreasing_values_and_outliers( unsigned long size, X mask, T *data, unsigned long outliers_count)
{
    //TODO: fix
    init_random_generator();
    data[0] = std::numeric_limits<T>::max() - 1;
    T v = 0;
    unsigned long max_outliers = outliers_count;
    int max_outliers_in_block = 8;
    for (unsigned long i = 1; i < size; i++){
        v = (xorshf96() & mask);
        if(max_outliers > 0 && max_outliers_in_block > 0 && xorshf96() % 2) {
            max_outliers_in_block --;
            max_outliers --;
            v <<= 1;
        }
        if(i % CWORD_SIZE(X) == 0) max_outliers_in_block = 8;
        if( ((long)data[i-1] - (long)v) >= 0 ){ // ensure that data does not go below 0
            data[i] = data[i-1] - v;
        } else {
            data[i] = 0;
        }
    }
}

template <typename T>
void big_random_block_with_decreasing_values_and_outliers( unsigned long size, int limit_bits, T *data, unsigned long outliers_count)
{
    T mask = NBITSTOMASK(limit_bits);
    __inner_big_random_block_with_decreasing_values_and_outliers(size, mask, data, outliers_count);
}

template <>
void big_random_block_with_decreasing_values_and_outliers( unsigned long size, int limit_bits, unsigned long *data, unsigned long outliers_count)
{
    unsigned long mask = LNBITSTOMASK(limit_bits);
    __inner_big_random_block_with_decreasing_values_and_outliers(size, mask, data, outliers_count);
}

template <>
void big_random_block_with_decreasing_values_and_outliers( unsigned long size, int limit_bits, long *data, unsigned long outliers_count)
{
    unsigned long mask = LNBITSTOMASK(limit_bits);
    __inner_big_random_block_with_decreasing_values_and_outliers(size, mask, data,  outliers_count);
}

template <typename T, typename X>
void __inner_big_random_block_with_decreasing_values_for_aafl( unsigned long size, X mask, T *data)
{
    T max = std::numeric_limits<T>::max();
    data[0] = max;
    T v = max & mask;
    int block_size = CWORD_SIZE(T) * 32;
    for (unsigned long i = 1; i < size; i++){ //ensure load close to maximum for all warps
        data[i] = max - v;
        if((i-1) % block_size == 0)  { //delta always resets per block
            data[i-1] = data[0];
        }
    }
    /* exit(0); */
}
template <typename T>
void big_random_block_with_decreasing_values_for_aafl( unsigned long size, int limit_bits, T *data)
{
    T mask = NBITSTOMASK(limit_bits);
    __inner_big_random_block_with_decreasing_values_for_aafl (size, mask, data);
}

template <>
void big_random_block_with_decreasing_values_for_aafl( unsigned long size, int limit_bits, unsigned long *data)
{
    unsigned long mask = LNBITSTOMASK(limit_bits);
    __inner_big_random_block_with_decreasing_values_for_aafl (size, mask, data);
}

template <>
void big_random_block_with_decreasing_values_for_aafl( unsigned long size, int limit_bits, long *data)
{
    unsigned long mask = LNBITSTOMASK(limit_bits);
   __inner_big_random_block_with_decreasing_values_for_aafl (size, mask, data);
}

template <typename T, typename X>
void __inner_big_random_block_with_outliers( unsigned long size, int outlier_count, int limit_bits, X outlier_mask,  T *data)
{
    init_random_generator();
    big_random_block(size, limit_bits, data);
    unsigned long exception = ULONG_MAX; // Sets all bits
    exception &= outlier_mask;

    for (int i = 0; i < outlier_count; ++i) {
        unsigned long p = xorshf96() % size;
        data[ p ] = exception;
    }
}

template <typename T>
void big_random_block_with_outliers( unsigned long size, int outlier_count, int limit_bits, int outlier_bits,  T *data)
{
    unsigned int mask = NBITSTOMASK(limit_bits + outlier_bits);
    __inner_big_random_block_with_outliers( size, outlier_count, limit_bits, mask,  data);
}

template <>
void big_random_block_with_outliers( unsigned long size, int outlier_count, int limit_bits, int outlier_bits,  long *data)
{
    unsigned long mask = LNBITSTOMASK(limit_bits + outlier_bits);
    __inner_big_random_block_with_outliers( size, outlier_count, limit_bits, mask,  data);
}

template <>
void big_random_block_with_outliers( unsigned long size, int outlier_count, int limit_bits, int outlier_bits,  unsigned long *data)
{
    unsigned long mask = LNBITSTOMASK(limit_bits + outlier_bits);
    __inner_big_random_block_with_outliers( size, outlier_count, limit_bits, mask,  data);
}

template <typename T>
unsigned long compare_arrays(T *in1, T *in2, unsigned long size)
{
    unsigned long count_errors = 0;
    unsigned long precheck = 0;

    // on most platforms memcmp should be faster (i.e. simd optimizations)
    precheck = memcmp(in1, in2, size*sizeof(T));
    if(precheck) // count errors if memcmp fails
        for(unsigned long i = 0; i < size; i++) {
            if(in1[i] != in2[i]) {
                count_errors += 1;
                //compare_arrays_element_print(i, in1[i], in2[i]);
            }
        }
    if (count_errors)
        DPRINT(("<================== ERROR ============= size = %ld errors = %ld\n", size, count_errors));
    return count_errors;
}


#define DATA_FUNC_SPEC(X) \
template void big_random_block <X> ( unsigned long size, int limit_bits, X *data);\
template void big_random_block_with_outliers <X> ( unsigned long size, int outlier_count, int limit_bits, int outlier_bits,  X *data);\
template unsigned long compare_arrays <X> (X *in1, X *in2, unsigned long size);\
template void big_random_block_with_decreasing_values <X> ( unsigned long size, int limit_bits, X *data);\
template void big_random_block_with_decreasing_values_and_outliers <X> ( unsigned long size, int limit_bits, X *data, unsigned long outliers_count);\
template void big_random_block_with_decreasing_values_for_aafl<X>( unsigned long size, int limit_bits, X *data);\
template void big_random_block_with_diff_in_abs_radious_values<X>( unsigned long size, int limit_bits, X *data);

FOR_EACH(DATA_FUNC_SPEC, int, long, unsigned int, unsigned long)
