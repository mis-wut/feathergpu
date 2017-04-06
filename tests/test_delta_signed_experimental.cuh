#pragma once
#include "test_delta.cuh"
#include "feathergpu/fl/delta_signed_experimental.cuh"
template <typename T, char CWARP_SIZE>
class test_delta_signed: public virtual test_delta<T, CWARP_SIZE>
{
    public:

        virtual void initializeData(int bit_length) {
            big_random_block_with_diff_in_abs_radious_values((unsigned long)this->max_size, bit_length - 1, this->host_data);

            //TODO: modify the data in such a way that it will result in some negative deltas
        }

        virtual void compressData(int bit_length) {
            run_delta_afl_compress_signed_gpu <T, CWARP_SIZE> (bit_length, this->dev_data, this->dev_out, this->dev_data_block_start, this->max_size);
        }

        virtual void decompressData(int bit_length) {
            run_delta_afl_decompress_signed_gpu <T, CWARP_SIZE> (bit_length, this->dev_out, this->dev_data_block_start, this->dev_data, this->max_size);
        }
};
