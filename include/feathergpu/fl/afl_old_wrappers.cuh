#pragma once
#include "feathergpu/fl/containers.cuh"
#include "feathergpu/fl/pafl.cuh"
#include "feathergpu/fl/delta_pafl.cuh"

template <typename T, char CWARP_SIZE>
__host__ void run_pafl_decompress_gpu(
        unsigned int bit_length,
        T *compressed_data,
        T *data,
        unsigned long length,

        T *global_data_patch_values,
        unsigned long *global_data_patch_index,
        unsigned long *global_data_patch_count
        )
{
    unsigned int block_size = CWARP_SIZE * 8; // better occupancy
    unsigned long block_number = (length + block_size * CWARP_SIZE - 1) / (block_size * CWARP_SIZE);

    container_uncompressed<T> udata = {data, length};
    container_fl<T> cdata_fl = {(unsigned char) bit_length, (make_unsigned_t<T> *) compressed_data, length};

    afl_decompress_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (cdata_fl, udata);

    cudaErrorCheck();

    container_pafl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) compressed_data, length, (make_unsigned_t<T> *) global_data_patch_values, global_data_patch_index, global_data_patch_count};

    patch_apply_kernel <T, CWARP_SIZE> <<<block_number * CWARP_SIZE, block_size>>> (udata, cdata);
}

template <typename T, char CWARP_SIZE>
__host__ void run_delta_pafl_decompress_gpu(
        unsigned int bit_length,
        T *compressed_data,
        T* compressed_data_block_start,
        T *data,
        unsigned long length,

        T *global_data_patch_values,
        unsigned long *global_data_patch_index,
        unsigned long *global_data_patch_count
        )
{
    unsigned int block_size = CWARP_SIZE * 8; // better occupancy
    unsigned long block_number = (length + block_size * CWARP_SIZE - 1) / (block_size * CWARP_SIZE);

    container_pafl<T> cdata_pafl = {(unsigned char) bit_length, (make_unsigned_t<T> *) compressed_data, length, (make_unsigned_t<T> *) global_data_patch_values, global_data_patch_index, global_data_patch_count};
    container_uncompressed<T> udata = {data, length};

    patch_apply_kernel <T, CWARP_SIZE> <<<block_number * CWARP_SIZE, block_size>>> (udata, cdata_pafl);
    cudaErrorCheck();

    container_delta_pafl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) compressed_data, length, (make_unsigned_t<T> *) global_data_patch_values, global_data_patch_index, global_data_patch_count, (make_unsigned_t<T> *) compressed_data_block_start};
    delta_pafl_decompress_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (cdata, udata);
}
