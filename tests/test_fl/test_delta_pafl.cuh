#pragma once

#include "test_pafl.cuh"
#include "test_delta.cuh"

template <typename T, char CWARP_SIZE>
class test_delta_pafl: public virtual test_pafl <T, CWARP_SIZE>, public virtual test_delta <T, CWARP_SIZE>
{
    virtual void allocateMemory() {
        test_base<T, CWARP_SIZE>::allocateMemory();
        test_pafl<T, CWARP_SIZE>::iner_allocateMemory();
        test_delta<T, CWARP_SIZE>::iner_allocateMemory();
    }

    virtual void setup(unsigned long max_size) {
        test_base<T, CWARP_SIZE>::setup(max_size);
        test_pafl<T, CWARP_SIZE> ::iner_setup(max_size);
        test_delta<T, CWARP_SIZE>::iner_setup(max_size);
    }
    virtual void initializeData(int bit_length) {
        if(bit_length > 30) bit_length = 30; //TODO:FIX

        big_random_block_with_decreasing_values_and_outliers ((unsigned long)this->max_size, bit_length, this->host_data, this->outlier_count);
    }

    // Clean up before compression
    virtual void cleanBeforeCompress() {
        test_base<T, CWARP_SIZE>::cleanBeforeCompress();
        test_pafl<T, CWARP_SIZE> :: iner_cleanBeforeCompress();
        test_delta<T, CWARP_SIZE>:: iner_cleanBeforeCompress();
    }

    virtual void compressData(int bit_length) {
        container_uncompressed<T> udata = {this->dev_data, this->max_size};
        container_delta_pafl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) this->dev_out, this->max_size, (make_unsigned_t<T> *) this->dev_data_patch_values, this->dev_data_patch_index, this->dev_data_patch_count, (make_unsigned_t<T> *)this->dev_data_block_start};

        compress<T, CWARP_SIZE>(udata, cdata);
    }

    virtual void decompressData(int bit_length) {
        container_uncompressed<T> udata = {this->dev_data, this->max_size};
        container_delta_pafl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) this->dev_out, this->max_size, (make_unsigned_t<T> *) this->dev_data_patch_values, this->dev_data_patch_index, this->dev_data_patch_count, (make_unsigned_t<T> *)this->dev_data_block_start};

        decompress<T, CWARP_SIZE>(cdata, udata);
    }

    virtual void print_compressed_data_size(){
        //TODO: fix this
        unsigned long patch_count;
        unsigned long compression_blocks_count = test_delta<T, CWARP_SIZE>::compression_blocks_count;
        cudaMemcpy(&patch_count, this->dev_data_patch_count, sizeof(unsigned long), cudaMemcpyDeviceToHost);
        printf("Comp ratio %f",  (float)this->max_size / (patch_count * (sizeof(T) + sizeof(long)) +  compression_blocks_count * sizeof(T) + this->compressed_data_size));
        printf(" %d %lu %ld %ld %ld\n" , this->bit_length, this->max_size, this->data_size, this->compressed_data_size, patch_count);
    }
};
