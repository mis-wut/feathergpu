#pragma once
#include <stdio.h>
#include <iostream>

// Tools
template <typename T> void big_random_block ( unsigned long size, int limit_bits, T *data);
template <typename T> void big_random_block_with_outliers ( unsigned long size, int outlier_count, int limit_bits, int outlier_bits,  T *data);
template <typename T> void big_random_block_with_decreasing_values_and_outliers ( unsigned long size, int limit_bits, T *data, unsigned long outliers_count); 
template <typename T> void big_random_block_with_decreasing_values_for_aafl( unsigned long size, int limit_bits, T *data);

template <typename T> unsigned long compare_arrays ( T *in1, T *in2, unsigned long size);

template <typename T>
void big_random_block_with_decreasing_values( unsigned long size, int limit_bits, T *data);

template <typename T1, typename T2, typename T3>
void inline compare_arrays_element_print(T1 i, T2 a, T3 b)
{
    if(abs(a) != abs(b))
        std::cout<< "Error at " << i << "element "<< a << " != " << b << "\n";
}

int compare_binary_file(const void *a, const void *b);
template <typename T>
void big_random_block_with_diff_in_abs_radious_values( unsigned long size, int limit_bits, T *data);


# define DPRINT(x) printf x
