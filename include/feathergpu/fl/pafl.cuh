#pragma once
#include "afl.cuh"
#include "delta.cuh"
#include "feathergpu/util/ptx.cuh"
#include "feathergpu/util/cuda.cuh"

#include <math.h>
#include <stdio.h>

template <typename T, char CWARP_SIZE>
__device__  void pafl_compress3 (
        const unsigned int bit_length,
        const unsigned long data_id,
        const unsigned long comp_data_id,

        T *data,
        T *compressed_data,
        unsigned long length,

        T *global_patch_values,
        unsigned long *global_patch_index,
        unsigned long *global_patch_count
        )
{
    if (data_id >= length) return;

    T v1, value = 0;
    unsigned int v1_pos=0, v1_len;
    unsigned long pos=comp_data_id, pos_data=data_id;
    unsigned int exception_counter = 0;

    T exception_buffer[8];
    unsigned long position_mask = 0;
    T mask;
    if (sizeof(T) == sizeof(long)) // TODO
        mask = ~LNBITSTOMASK(bit_length);
    else
        mask = ~NBITSTOMASK(bit_length);

    for (unsigned int i = 0; i < CWORD_SIZE(T) && pos_data < length; ++i)
    {
        v1 = data[pos_data];

        if(v1 & mask){
            exception_buffer[exception_counter] = v1;
            exception_counter ++;
            BIT_SET(position_mask, i);
        }

        pos_data += CWARP_SIZE;

        if (v1_pos >= CWORD_SIZE(T) - bit_length){
            v1_len = CWORD_SIZE(T) - v1_pos;
            value = value | (GETNBITS(v1, v1_len) << v1_pos);

            compressed_data[pos] = value;

            v1_pos = bit_length - v1_len;
            value = GETNPBITS(v1, v1_pos, v1_len);

            pos += CWARP_SIZE;
        } else {
            v1_len = bit_length;
            value = value | (GETNBITS(v1, v1_len) << v1_pos);
            v1_pos += v1_len;
        }
    }
    if (pos_data >= length  && pos_data < length + CWARP_SIZE)
    {
        compressed_data[pos] = value;
    }

    unsigned int lane_id = get_lane_id();
    unsigned long local_counter = 0;

    unsigned int warp_exception_counter = shfl_prefix_sum((int)exception_counter);

    if(lane_id == 31 && warp_exception_counter > 0){
        local_counter = atomicAdd((unsigned long long int *)global_patch_count, (unsigned long long int)warp_exception_counter);
    }

    local_counter = shfl_get_value((long)local_counter, 31);

    for (unsigned int i = 0; i < exception_counter; ++i)
        global_patch_values[local_counter + warp_exception_counter - exception_counter + i] = exception_buffer [i];

    for (unsigned int i = 0, j = 0; i < exception_counter && j < CWORD_SIZE(T); j++){
        if (BIT_CHECK(position_mask, j)) {
            global_patch_index[local_counter + warp_exception_counter - exception_counter + i] = data_id + j * CWARP_SIZE;
            i++;
        }
    }
}

template <typename T, char CWARP_SIZE>
__device__  void delta_pafl_compress3 (
        const unsigned int bit_length,
        const unsigned long data_id,
        const unsigned long comp_data_id,

        T *data,
        T *compressed_data,
        T* compressed_data_block_start,
        unsigned long length,

        T *global_patch_values,
        unsigned long *global_patch_index,
        unsigned long *global_patch_count
        )
{
    if (data_id >= length) return;

    T v1, value = 0;
    unsigned int v1_pos=0, v1_len;
    unsigned long pos=comp_data_id, pos_data=data_id;
    unsigned int exception_counter = 0;

    T exception_buffer[8];
    unsigned long position_mask = 0;
    T mask;
    if (sizeof(T) == sizeof(long)) // TODO
        mask = ~LNBITSTOMASK(bit_length);
    else
        mask = ~NBITSTOMASK(bit_length);

    T zeroLaneValue, v2;
    const unsigned long lane = get_lane_id();
    char neighborId = lane - 1;

    const unsigned long data_block = ( blockIdx.x * blockDim.x) / CWARP_SIZE + threadIdx.x / CWARP_SIZE;

    if (lane == 0 )  {
        neighborId = 31;
        zeroLaneValue = data[pos_data];
        compressed_data_block_start[data_block] = zeroLaneValue;
    }

    for (unsigned int i = 0; i < CWORD_SIZE(T) && pos_data < length; ++i)
    {
        v1 = data[pos_data];

        pos_data += CWARP_SIZE;

        v2 = shfl_get_value(v1, neighborId);

        if (lane == 0)
        {
            // Lane 0 uses data from previous iteration
            v1 = zeroLaneValue - v1;
            zeroLaneValue = v2;
        } else {
            v1 = v2 - v1;
        }

        if(v1 & mask){
            exception_buffer[exception_counter] = v1;
            exception_counter ++;
            BIT_SET(position_mask, i);
        }

        if (v1_pos >= CWORD_SIZE(T) - bit_length){
            v1_len = CWORD_SIZE(T) - v1_pos;
            value = value | (GETNBITS(v1, v1_len) << v1_pos);

            compressed_data[pos] = value;

            v1_pos = bit_length - v1_len;
            value = GETNPBITS(v1, v1_pos, v1_len);

            pos += CWARP_SIZE;
        } else {
            v1_len = bit_length;
            value = value | (GETNBITS(v1, v1_len) << v1_pos);
            v1_pos += v1_len;
        }
    }
    if (pos_data >= length  && pos_data < length + CWARP_SIZE)
    {
        compressed_data[pos] = value;
    }

    unsigned int lane_id = get_lane_id();
    unsigned long local_counter = 0;

    unsigned int warp_exception_counter = shfl_prefix_sum((int)exception_counter);

    if(lane_id == 31 && warp_exception_counter > 0){
        local_counter = atomicAdd((unsigned long long int *)global_patch_count, (unsigned long long int)warp_exception_counter);
    }

    local_counter = shfl_get_value((long)local_counter, 31);

    for (unsigned int i = 0; i < exception_counter; ++i)
        global_patch_values[local_counter + warp_exception_counter - exception_counter + i] = exception_buffer [i];

    for (unsigned int i = 0, j = 0; i < exception_counter && j < CWORD_SIZE(T); j++){
        if (BIT_CHECK(position_mask, j)) {
            global_patch_index[local_counter + warp_exception_counter - exception_counter + i] = data_id + j * CWARP_SIZE;
            i++;
        }
    }
}

template < typename T, char CWARP_SIZE >
__global__ void pafl_compress_kernel (
        const unsigned int bit_length,
        T *data,
        T *compressed_data,
        unsigned long length,

        T *global_data_patch_values,
        unsigned long *global_data_patch_index,
        unsigned long *global_data_patch_count
        )
{
    const unsigned int warp_lane = threadIdx.x % CWARP_SIZE;
    const unsigned long data_block = blockIdx.x * blockDim.x + threadIdx.x - warp_lane;
    const unsigned long data_id = data_block * CWORD_SIZE(T) + warp_lane;
    const unsigned long cdata_id = data_block * bit_length + warp_lane;

    pafl_compress3 <T, CWARP_SIZE> (
            bit_length, data_id, cdata_id, data, compressed_data,
            length,
            global_data_patch_values,
            global_data_patch_index,
            global_data_patch_count);
}

template < typename T, char CWARP_SIZE >
__global__ void delta_pafl_compress_kernel (
        const unsigned int bit_length,
        T *data,
        T *compressed_data,
        T* compressed_data_block_start,
        unsigned long length,

        T *global_data_patch_values,
        unsigned long *global_data_patch_index,
        unsigned long *global_data_patch_count
        )
{
    const unsigned int warp_lane = threadIdx.x % CWARP_SIZE;
    const unsigned long data_block = blockIdx.x * blockDim.x + threadIdx.x - warp_lane;
    const unsigned long data_id = data_block * CWORD_SIZE(T) + warp_lane;
    const unsigned long cdata_id = data_block * bit_length + warp_lane;

    delta_pafl_compress3 <T, CWARP_SIZE> (
            bit_length, data_id, cdata_id, data, compressed_data, compressed_data_block_start,
            length,
            global_data_patch_values,
            global_data_patch_index,
            global_data_patch_count);
}

template <typename T, char CWARP_SIZE>
__global__ void patch_apply_kernel (
        T *decompressed_data,
        unsigned long length,

        T *global_data_patch_values,
        unsigned long *global_data_patch_index,
        unsigned long *global_data_patch_count
        ) //TODO: fix params list
{
    unsigned long tid = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long patch_length = *global_data_patch_count;

    if (tid < patch_length)
    {
        unsigned long idx = global_data_patch_index[tid];
        T val = global_data_patch_values[tid];
        decompressed_data[idx] = val;
    }
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

    pafl_compress_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (
            bit_length,
            data,
            compressed_data,
            length,

            global_data_patch_values,
            global_data_patch_index,
            global_data_patch_count
            );


    cudaErrorCheck();
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

    delta_pafl_compress_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (
            bit_length,
            data,
            compressed_data,
            compressed_data_block_start,
            length,

            global_data_patch_values,
            global_data_patch_index,
            global_data_patch_count
            );

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

    afl_decompress_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (
            bit_length,
            compressed_data,
            data,
            length
            );

    cudaErrorCheck();

    patch_apply_kernel <T, CWARP_SIZE> <<<block_number * CWARP_SIZE, block_size>>> (
            data,
            length,

            global_data_patch_values,
            global_data_patch_index,
            global_data_patch_count
            );
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


    patch_apply_kernel <T, CWARP_SIZE> <<<block_number * CWARP_SIZE, block_size>>> (
            data,
            length,

            global_data_patch_values,
            global_data_patch_index,
            global_data_patch_count
            );
    cudaErrorCheck();

    delta_pafl_decompress_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (
            bit_length,
            compressed_data,
            compressed_data_block_start,
            data,
            length
            );
}
