#pragma once
#include "feathergpu/util/ptx.cuh"

template <typename T, char CWARP_SIZE>
__device__  __host__ void afl_compress_signed (
        const unsigned int bit_length,
        unsigned long data_id,
        unsigned long comp_data_id,
        T *data,
        T *compressed_data,
        unsigned long length
     )
{

    if (data_id >= length) return;
    // TODO: Compressed data should be always unsigned, fix that latter
    T v1;
    unsigned int uv1;
    unsigned int value = 0;
    unsigned int v1_pos=0, v1_len;
    unsigned long pos=comp_data_id, pos_data=data_id;
    unsigned int sgn = 0;

    for (unsigned int i = 0; i < CWORD_SIZE(T) && pos_data < length; ++i)
    {
        v1 = data[pos_data];

        //TODO: ugly hack, fix that with correct bfe calls
        sgn = ((unsigned int) v1) >> 31;
        uv1 = abs(v1);
        // END: ugly hack

        pos_data += CWARP_SIZE;

        if (v1_pos >= CWORD_SIZE(T) - bit_length){
            v1_len = CWORD_SIZE(T) - v1_pos;

            if (v1_pos == CWORD_SIZE(T) - bit_length) // whole word
                value |= (GETNBITS(uv1, v1_len - 1) | (sgn << (v1_len - 1))) << (v1_pos);
            else // begining of the word
                value |= GETNBITS(uv1, v1_len) << (v1_pos);

            compressed_data[pos] = reinterpret_cast<int&>(value);

            v1_pos = bit_length - v1_len;

            value = 0;
            // if is necessary as otherwise may work with negative bit shifts
            if (v1_pos > 0) // The last part of the word
                value = (GETNPBITS(uv1, v1_pos - 1, v1_len)) | (sgn << (v1_pos - 1));

            pos += CWARP_SIZE;
        } else { // whole word @ one go
            v1_len = bit_length;
            value |= (GETNBITS(uv1, v1_len-1) | (sgn << (v1_len-1))) << v1_pos;
            v1_pos += v1_len;
        }
    }
    if (pos_data >= length  && pos_data < length + CWARP_SIZE)
    {
        compressed_data[pos] = reinterpret_cast<int&>(value);
    }
}

template <typename T, char CWARP_SIZE>
__device__ __host__ void afl_decompress_signed (
        const unsigned int bit_length,
         unsigned long comp_data_id,
         unsigned long data_id,
         T *compressed_data,
         T *data,
         unsigned long length
     )
{
    // TODO: Compressed data should be always unsigned, fix that latter
    unsigned long pos = comp_data_id, pos_decomp = data_id;
    unsigned int v1_pos = 0, v1_len;
    unsigned int v1;
    unsigned int ret;

    if (pos_decomp > length ) // Decompress not more elements then length
        return;
    v1 = reinterpret_cast<unsigned int &>(compressed_data[pos]);
    for (unsigned int i = 0; i < CWORD_SIZE(T) && pos_decomp < length; ++i)
    {
        if (v1_pos >= CWORD_SIZE(T) - bit_length){
            v1_len = CWORD_SIZE(T) - v1_pos;
            ret = GETNPBITS(v1, v1_len, v1_pos);

            pos += CWARP_SIZE;
            v1 = reinterpret_cast<unsigned int &>(compressed_data[pos]);

            v1_pos = bit_length - v1_len;
            ret = ret | (GETNBITS(v1, v1_pos) << v1_len);
        } else {
            v1_len = bit_length;
            ret = GETNPBITS(v1, v1_len, v1_pos);
            v1_pos += v1_len;
        }

        // TODO: dirty hack
        int sgn_multiply = (ret >> (bit_length-1)) ? -1 : 1;
        // END
        ret &= NBITSTOMASK(bit_length-1);

        data[pos_decomp] = sgn_multiply * (int)(ret);
        pos_decomp += CWARP_SIZE;
    }
}

template < typename T, char CWARP_SIZE >
__global__ void afl_compress_signed_kernel (const unsigned int bit_length, T *data, T *compressed_data, unsigned long length)
{
    const unsigned int warp_lane = threadIdx.x % CWARP_SIZE;
    const unsigned long data_block = blockIdx.x * blockDim.x + threadIdx.x - warp_lane;
    const unsigned long data_id = data_block * CWORD_SIZE(T) + warp_lane;
    const unsigned long cdata_id = data_block * bit_length + warp_lane;

    afl_compress_signed <T, CWARP_SIZE> (bit_length, data_id, cdata_id, data, compressed_data, length);
}

template < typename T, char CWARP_SIZE >
__global__ void afl_decompress_signed_kernel (const unsigned int bit_length, T *compressed_data, T * decompress_data, unsigned long length)
{
    const unsigned int warp_lane = threadIdx.x % CWARP_SIZE;
    const unsigned long data_block = blockIdx.x * blockDim.x + threadIdx.x - warp_lane;
    const unsigned long data_id = data_block * CWORD_SIZE(T) + warp_lane;
    const unsigned long cdata_id = data_block * bit_length + warp_lane;

    afl_decompress_signed <T, CWARP_SIZE> (bit_length, cdata_id, data_id, compressed_data, decompress_data, length);
}

template < typename T, char CWARP_SIZE >
__host__ void run_afl_compress_signed_gpu(const unsigned int bit_length, T *data, T *compressed_data, unsigned long length)
{
    const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
    const unsigned long block_number = (length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));
    afl_compress_signed_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (bit_length, data, compressed_data, length);
}

template < typename T, char CWARP_SIZE >
__host__ void run_afl_decompress_signed_gpu(const unsigned int bit_length, T *compressed_data, T *data, unsigned long length)
{
    const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
    const unsigned long block_number = (length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));
    afl_decompress_signed_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (bit_length, compressed_data, data, length);
}
