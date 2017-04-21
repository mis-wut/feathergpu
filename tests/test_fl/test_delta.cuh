#pragma once

#include "test_base.cuh"
#include "tools/data.cuh"
#include "feathergpu/fl/delta.cuh"
#include <limits>


template <typename T, char CWARP_SIZE>
class test_delta: public virtual test_base<T, CWARP_SIZE>
{
    public:
        virtual void allocateMemory() {
            test_base <T, CWARP_SIZE>::allocateMemory();
            iner_allocateMemory();
        }

        virtual void iner_allocateMemory() {
            mmCudaMalloc(this->manager, (void **) &this->dev_data_block_start, compression_blocks_count * sizeof(unsigned long));
        }

        virtual void setup(unsigned long max_size) {
            test_base <T, CWARP_SIZE>::setup(max_size);
            iner_setup(max_size);
        }

        virtual void iner_setup(unsigned long max_size) {
            this->compression_blocks_count = (this->compressed_data_size + (sizeof(T) * CWARP_SIZE) - 1) / (sizeof(T) * CWARP_SIZE);
        }

        virtual void initializeData(int bit_length) {
            big_random_block_with_decreasing_values ((unsigned long)this->max_size, bit_length, this->host_data);
        }

        virtual void cleanBeforeCompress() {
            test_base <T, CWARP_SIZE>::cleanBeforeCompress();
            iner_cleanBeforeCompress();
        }

        virtual void iner_cleanBeforeCompress() {
            cudaMemset(this->dev_data_block_start, 0, compression_blocks_count * sizeof(unsigned long));
        }

        virtual void compressData(int bit_length) {
            run_delta_afl_compress_gpu <T, CWARP_SIZE> (bit_length, this->dev_data, this->dev_out, this->dev_data_block_start, this->max_size);
        }

        virtual void decompressData(int bit_length) {
            run_delta_afl_decompress_gpu <T, CWARP_SIZE> (bit_length, this->dev_out, this->dev_data_block_start, this->dev_data, this->max_size);
        }

    protected:
        T *dev_data_block_start;
        unsigned long compression_blocks_count;
};
