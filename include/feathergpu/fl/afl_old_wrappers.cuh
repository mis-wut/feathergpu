#pragma once
#include "feathergpu/fl/containers.cuh"
#include "feathergpu/fl/afl.cuh"
#include "feathergpu/fl/pafl.cuh"
#include "feathergpu/fl/afl_signed_experimental.cuh"
#include "feathergpu/fl/aafl.cuh"
#include "feathergpu/fl/delta_aafl.cuh"
#include "feathergpu/fl/delta_pafl.cuh"
#include "feathergpu/fl/delta.cuh"
#include "feathergpu/fl/delta_signed_experimental.cuh"

template < typename T, char CWARP_SIZE >
__host__ void run_aafl_compress_gpu(
        unsigned long *compressed_data_register,
        unsigned char *warp_bit_lenght,
        unsigned long *warp_position_id,
        T *data,
        T *compressed_data,
        unsigned long length)
{
    const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
    const unsigned long block_number = (length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));

    container_uncompressed<T> udata = {data, length};
    container_aafl<T> cdata = {(make_unsigned_t<T> *) compressed_data, length, warp_bit_lenght, warp_position_id, compressed_data_register};

    aafl_compress_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (udata, cdata);
}

template < typename T, char CWARP_SIZE >
__host__ void run_aafl_decompress_gpu(
        unsigned char *warp_bit_lenght,
        unsigned long *warp_position_id,
        T *compressed_data,
        T *data,
        unsigned long length)
{
    const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
    const unsigned long block_number = (length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));

    container_uncompressed<T> udata = {data, length};
    container_aafl<T> cdata = {(make_unsigned_t<T> *) compressed_data, length, warp_bit_lenght, warp_position_id, NULL};

    aafl_decompress_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (cdata, udata);
}

template < typename T, char CWARP_SIZE >
__host__ void run_delta_afl_compress_gpu(const unsigned int bit_length, T *data, T *compressed_data, T* compressed_data_block_start, unsigned long length)
{
    const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
    const unsigned long block_number = (length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));

    container_uncompressed<T> udata = {data, length};
    container_delta_fl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) compressed_data, length, (make_unsigned_t<T> *)compressed_data_block_start};

    delta_afl_compress_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (udata, cdata);
}

template < typename T, char CWARP_SIZE >
__host__ void run_delta_afl_decompress_gpu(const unsigned int bit_length, T *compressed_data, T* compressed_data_block_start, T *data, unsigned long length)
{
    const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
    const unsigned long block_number = (length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));

    container_uncompressed<T> udata = {data, length};
    container_delta_fl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) compressed_data, length, (make_unsigned_t<T> *) compressed_data_block_start};

    delta_afl_decompress_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (cdata, udata);
}

template < typename T, char CWARP_SIZE >
__host__ void run_afl_compress_signed_gpu(const unsigned int bit_length, T *data, T *compressed_data, unsigned long length)
{
    const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
    const unsigned long block_number = (length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));

    container_uncompressed<T> udata = {data, length};
    container_fl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) compressed_data, length};

    afl_compress_signed_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (udata, cdata);
}

template < typename T, char CWARP_SIZE >
__host__ void run_afl_decompress_signed_gpu(const unsigned int bit_length, T *compressed_data, T *data, unsigned long length)
{
    const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
    const unsigned long block_number = (length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));

    container_uncompressed<T> udata = {data, length};
    container_fl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) compressed_data, length};

    afl_decompress_signed_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (cdata, udata);
}

template < typename T, char CWARP_SIZE >
__host__ void run_delta_afl_compress_signed_gpu(const unsigned int bit_length, T *data, T *compressed_data, T* compressed_data_block_start, unsigned long length)
{
    const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
    const unsigned long block_number = (length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));

    container_uncompressed<T> udata = {data, length};
    container_delta_fl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) compressed_data, length, (make_unsigned_t<T> *) compressed_data_block_start};

    delta_afl_compress_signed_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (udata, cdata);
}

template < typename T, char CWARP_SIZE >
__host__ void run_delta_afl_decompress_signed_gpu(const unsigned int bit_length, T *compressed_data, T* compressed_data_block_start, T *data, unsigned long length)
{
    const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
    const unsigned long block_number = (length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));
    container_uncompressed<T> udata = {data, length};
    container_delta_fl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) compressed_data, length, (make_unsigned_t<T> *) compressed_data_block_start};

    delta_afl_decompress_signed_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (cdata, udata);
}

template < typename T, char CWARP_SIZE >
__host__ void run_delta_aafl_compress_gpu(
        unsigned long *compressed_data_register,
        unsigned char *warp_bit_lenght,
        unsigned long *warp_position_id,
        T *data,
        T *compressed_data,
        T* compressed_data_block_start,
        unsigned long length)
{
    const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
    const unsigned long block_number = (length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));

    container_uncompressed<T> udata = {data, length};
    container_delta_aafl<T> cdata = {(make_unsigned_t<T> *) compressed_data, length, warp_bit_lenght, warp_position_id, compressed_data_register, (make_unsigned_t<T> *) compressed_data_block_start};
    delta_aafl_compress_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (udata, cdata);
}

template < typename T, char CWARP_SIZE >
__host__ void run_delta_aafl_decompress_gpu(
        unsigned char *warp_bit_lenght,
        unsigned long *warp_position_id,
        T* compressed_data_block_start,
        T *compressed_data,
        T *data,
        unsigned long length)
{
    const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
    const unsigned long block_number = (length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));

    container_uncompressed<T> udata = {data, length};
    container_delta_aafl<T> cdata = {(make_unsigned_t<T> *) compressed_data, length, warp_bit_lenght, warp_position_id, NULL, (make_unsigned_t<T> *) compressed_data_block_start};

    delta_aafl_decompress_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (cdata, udata);
}

template <typename T, char CWARP_SIZE>
__host__ void run_pafl_compress_gpu_alternate(
        unsigned int bit_length,
        T *data,
        T *compressed_data,
        unsigned long length,

        T *global_data_patch_values,
        unsigned long *global_data_patch_index,
        unsigned long *global_data_patch_count
        )
{
    const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
    const unsigned long block_number = (length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));

    container_uncompressed<T> udata = {data, length};
    container_pafl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) compressed_data, length, (make_unsigned_t<T> *) global_data_patch_values, global_data_patch_index, global_data_patch_count};

    pafl_compress_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (udata, cdata);

    cudaErrorCheck();
}

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
__host__ void run_delta_pafl_compress_gpu_alternate(
        unsigned int bit_length,
        T *data,
        T *compressed_data,
        T* compressed_data_block_start,
        unsigned long length,

        T *global_data_patch_values,
        unsigned long *global_data_patch_index,
        unsigned long *global_data_patch_count
        )
{
    const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
    const unsigned long block_number = (length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));

    container_uncompressed<T> udata = {data, length};
    container_delta_pafl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) compressed_data, length, (make_unsigned_t<T> *) global_data_patch_values, global_data_patch_index, global_data_patch_count, (make_unsigned_t<T> *) compressed_data_block_start};

    delta_pafl_compress_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (udata, cdata);

    cudaErrorCheck();
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
