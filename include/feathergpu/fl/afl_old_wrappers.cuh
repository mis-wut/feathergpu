#pragma once
#include "feathergpu/fl/containers.cuh"
#include "feathergpu/fl/afl.cuh"

template < typename T, char CWARP_SIZE >
__global__ void afl_decompress_kernel_todo (const unsigned int bit_length, T *compressed_data, T * decompress_data, unsigned long length)
{
    container_uncompressed<T> udata = {decompress_data, length};
    container_fl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) compressed_data, length};

    unsigned long data_id, cdata_id;
    set_cmp_offset<T, CWARP_SIZE>(threadIdx.x, blockIdx.x * blockDim.x, cdata.bit_length, data_id, cdata_id);

    afl_decompress <T, CWARP_SIZE> (cdata_id, data_id, cdata, udata);
}

template <typename T, char CWARP_SIZE>
__forceinline__ __device__  __host__ void afl_compress_todo (const unsigned int bit_length, unsigned long data_id, unsigned long comp_data_id, T *data, T *compressed_data, unsigned long length)
{
    container_uncompressed<T> udata = {data, length};
    container_fl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) compressed_data, length};

    afl_compress<T, CWARP_SIZE>(data_id, comp_data_id, udata, cdata);
}

template <typename T, char CWARP_SIZE>
__forceinline__ __device__ __host__ void afl_decompress_todo (const unsigned int bit_length, unsigned long comp_data_id, unsigned long data_id, T *compressed_data, T *data, unsigned long length)
{
    container_uncompressed<T> udata = {data, length};
    container_fl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) compressed_data, length};
    afl_decompress<T, CWARP_SIZE> (comp_data_id, data_id,  cdata, udata);
}

template < typename T, char CWARP_SIZE >
__host__ void run_afl_compress_value_cpu( const unsigned int bit_length, T *data, T *compressed_data, const unsigned long length)
{
    container_uncompressed<T> udata = {data, length};
    container_fl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) compressed_data, length};
    afl_compress_value_cpu_kernel <T, CWARP_SIZE> ( udata, cdata);
}

template < typename T, char CWARP_SIZE >
__host__ void run_afl_compress_cpu( const unsigned int bit_length, T *data, T *compressed_data, const unsigned long length)
{
    container_uncompressed<T> udata = {data, length};
    container_fl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) compressed_data, length};
    afl_compress_cpu_kernel <T, CWARP_SIZE> ( udata, cdata);
}

template < typename T, char CWARP_SIZE >
__host__ void run_afl_decompress_cpu(const unsigned int bit_length, T *compressed_data, T *decompress_data, unsigned long length)
{
    container_uncompressed<T> udata = {decompress_data, length};
    container_fl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) compressed_data, length};
    afl_decompress_cpu_kernel <T, CWARP_SIZE> (cdata, udata);
}

template < typename T, char CWARP_SIZE >
__host__ void run_afl_compress_gpu(const unsigned int bit_length, T *data, T *compressed_data, unsigned long length)
{
    const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
    const unsigned long block_number = (length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));
    container_uncompressed<T> udata = {data, length};
    container_fl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) compressed_data, length};
    afl_compress_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (udata, cdata);
}

template < typename T, char CWARP_SIZE >
__host__ void run_afl_decompress_gpu(const unsigned int bit_length, T *compressed_data, T *data, unsigned long length)
{
    const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
    const unsigned long block_number = (length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));
    container_uncompressed<T> udata = {data, length};
    container_fl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) compressed_data, length};
    afl_decompress_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (cdata, udata);
}

template < typename T, char CWARP_SIZE >
__host__ void run_afl_decompress_value_gpu(const unsigned int bit_length, T *compressed_data, T *data, unsigned long length)
{
    const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
    const unsigned long block_number = (length + block_size * CWORD_SIZE(T) - 1) / (block_size);
    container_uncompressed<T> udata = {data, length};
    container_fl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) compressed_data, length};
    afl_decompress_value_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (cdata, udata);
}
