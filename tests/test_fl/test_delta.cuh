#pragma once

#include "test_base.cuh"
#include "tools/data.cuh"
#include "feathergpu/fl/delta.cuh"
#include "feathergpu/fl/afl_old_wrappers.cuh"
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
            const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
            const unsigned long block_number = (this->max_size + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));

            container_uncompressed<T> udata = {this->dev_data, this->max_size};
            container_delta_fl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) this->dev_out, this->max_size, (make_unsigned_t<T> *) this->dev_data_block_start};

            delta_afl_compress_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (udata, cdata);
        }

        virtual void decompressData(int bit_length) {
            const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
            const unsigned long block_number = (this->max_size + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));

            container_uncompressed<T> udata = {this->dev_data, this->max_size};
            container_delta_fl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) this->dev_out, this->max_size, (make_unsigned_t<T> *) this->dev_data_block_start};

            delta_afl_decompress_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (cdata, udata);
        }

    protected:
        T *dev_data_block_start;
        unsigned long compression_blocks_count;
};
