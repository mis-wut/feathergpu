#pragma once
#include "feathergpu/util/ptx.cuh"

template <typename T, char CWARP_SIZE>
__device__  void delta_afl_compress_todo (
        const unsigned int bit_length,
        unsigned long data_id,
        unsigned long comp_data_id,
        T *data,
        T *compressed_data,
        T* compressed_data_block_start,
        unsigned long length
        )
{
    if (data_id >= length) return;

    T v1, value = 0;
    unsigned int v1_pos=0, v1_len;
    unsigned long pos=comp_data_id, pos_data=data_id;

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
}

template <typename T, char CWARP_SIZE>
__device__ void delta_afl_decompress (
        const unsigned int bit_length,
        unsigned long comp_data_id,
        unsigned long data_id,
        T *compressed_data,
        T* compressed_data_block_start,
        T *data,
        unsigned long length
        )
{
    unsigned long pos = comp_data_id, pos_decomp = data_id;
    unsigned int v1_pos = 0, v1_len;
    T v1, ret;

    const unsigned long lane = get_lane_id();

    if (pos_decomp >= length ) // Decompress not more elements then length
        return;

    v1 = compressed_data[pos];

    T zeroLaneValue = 0, v2 = 0;

    const unsigned long data_block = (blockIdx.x * blockDim.x) / CWARP_SIZE  + threadIdx.x / CWARP_SIZE;

    if (lane == 0) {
       zeroLaneValue = compressed_data_block_start[data_block];
    }

    for (unsigned int i = 0; i < CWORD_SIZE(T) && pos_decomp < length; ++i)
    {
        if (v1_pos >= CWORD_SIZE(T) - bit_length){
            v1_len = CWORD_SIZE(T) - v1_pos;
            ret = GETNPBITS(v1, v1_len, v1_pos);

            pos += CWARP_SIZE;
            v1 = compressed_data[pos];

            v1_pos = bit_length - v1_len;
            ret = ret | ((GETNBITS(v1, v1_pos))<< v1_len);
        } else {
            v1_len = bit_length;
            ret = GETNPBITS(v1, v1_len, v1_pos);
            v1_pos += v1_len;
        }

        ret = shfl_prefix_sum(ret); // prefix sum deltas
        v2 = shfl_get_value(zeroLaneValue, 0);
        ret = v2 - ret;

        data[pos_decomp] = ret;
        pos_decomp += CWARP_SIZE;

        v2 = shfl_get_value(ret, 31);

        if(lane == 0)
            zeroLaneValue = v2;
    }
}

template <typename T, char CWARP_SIZE>
__device__ void delta_pafl_decompress (
        const unsigned int bit_length,
        unsigned long comp_data_id,
        unsigned long data_id,
        T *compressed_data,
        T* compressed_data_block_start,
        T *data,
        unsigned long length
        )
{
    unsigned long pos = comp_data_id, pos_decomp = data_id;
    unsigned int v1_pos = 0, v1_len;
    T v1, ret;

    const unsigned long lane = get_lane_id();

    if (pos_decomp >= length ) // Decompress not more elements then length
        return;

    v1 = compressed_data[pos];

    T zeroLaneValue = 0, v2 = 0;

    const unsigned long data_block = (blockIdx.x * blockDim.x) / CWARP_SIZE  + threadIdx.x / CWARP_SIZE;

    if (lane == 0) {
       zeroLaneValue = compressed_data_block_start[data_block];
    }

    for (unsigned int i = 0; i < CWORD_SIZE(T) && pos_decomp < length; ++i)
    {
        if (v1_pos >= CWORD_SIZE(T) - bit_length){
            v1_len = CWORD_SIZE(T) - v1_pos;
            ret = GETNPBITS(v1, v1_len, v1_pos);

            pos += CWARP_SIZE;
            v1 = compressed_data[pos];

            v1_pos = bit_length - v1_len;
            ret = ret | ((GETNBITS(v1, v1_pos))<< v1_len);
        } else {
            v1_len = bit_length;
            ret = GETNPBITS(v1, v1_len, v1_pos);
            v1_pos += v1_len;
        }

        if(data[pos_decomp] > 0)
            ret = data[pos_decomp];

        ret = shfl_prefix_sum(ret); // prefix sum deltas
        v2 = shfl_get_value(zeroLaneValue, 0);
        ret = v2 - ret;

        data[pos_decomp] = ret;
        pos_decomp += CWARP_SIZE;

        v2 = shfl_get_value(ret, 31);

        if(lane == 0)
            zeroLaneValue = v2;
    }
}

template < typename T, char CWARP_SIZE >
__global__ void delta_afl_compress_kernel (const unsigned int bit_length, T *data, T *compressed_data, T* compressed_data_block_start, unsigned long length)
{
    const unsigned int warp_lane = get_lane_id();
    const unsigned long data_block = blockIdx.x * blockDim.x + threadIdx.x - warp_lane;
    const unsigned long data_id = data_block * CWORD_SIZE(T) + warp_lane;
    const unsigned long cdata_id = data_block * bit_length + warp_lane;

    delta_afl_compress_todo <T, CWARP_SIZE> (bit_length, data_id, cdata_id, data, compressed_data, compressed_data_block_start, length);
}

template < typename T, char CWARP_SIZE >
__global__ void delta_afl_decompress_kernel (const unsigned int bit_length, T *compressed_data, T* compressed_data_block_start, T * decompress_data, unsigned long length)
{
    const unsigned int warp_lane = get_lane_id();
    const unsigned long data_block = blockIdx.x * blockDim.x + threadIdx.x - warp_lane;
    const unsigned long data_id = data_block * CWORD_SIZE(T) + warp_lane;
    const unsigned long cdata_id = data_block * bit_length + warp_lane;

    delta_afl_decompress <T, CWARP_SIZE> (bit_length, cdata_id, data_id, compressed_data, compressed_data_block_start, decompress_data, length);
}


template < typename T, char CWARP_SIZE >
__global__ void delta_pafl_decompress_kernel (const unsigned int bit_length, T *compressed_data, T* compressed_data_block_start, T * decompress_data, unsigned long length)
{
    const unsigned int warp_lane = get_lane_id();
    const unsigned long data_block = blockIdx.x * blockDim.x + threadIdx.x - warp_lane;
    const unsigned long data_id = data_block * CWORD_SIZE(T) + warp_lane;
    const unsigned long cdata_id = data_block * bit_length + warp_lane;

    delta_pafl_decompress <T, CWARP_SIZE> (bit_length, cdata_id, data_id, compressed_data, compressed_data_block_start, decompress_data, length);
}

template < typename T, char CWARP_SIZE >
__host__ void run_delta_afl_compress_gpu(const unsigned int bit_length, T *data, T *compressed_data, T* compressed_data_block_start, unsigned long length)
{
    const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
    const unsigned long block_number = (length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));
    delta_afl_compress_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (bit_length, data, compressed_data, compressed_data_block_start,length);
}

template < typename T, char CWARP_SIZE >
__host__ void run_delta_afl_decompress_gpu(const unsigned int bit_length, T *compressed_data, T* compressed_data_block_start, T *data, unsigned long length)
{
    const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
    const unsigned long block_number = (length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));
    delta_afl_decompress_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (bit_length, compressed_data, compressed_data_block_start,data, length);
}
