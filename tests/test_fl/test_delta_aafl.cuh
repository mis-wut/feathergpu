#pragma once
#include "test_aafl.cuh"
#include "test_delta.cuh"
#include "feathergpu/fl/delta_aafl.cuh"

template <typename T, char CWARP_SIZE>
class test_delta_aafl: public test_aafl <T, CWARP_SIZE>, public test_delta <T, CWARP_SIZE>
{

    virtual void allocateMemory() {
        test_base<T, CWARP_SIZE>::allocateMemory();
        test_aafl<T, CWARP_SIZE>::iner_allocateMemory();
        test_delta<T, CWARP_SIZE>::iner_allocateMemory();
    }

    virtual void setup(unsigned long max_size) {
        test_base<T, CWARP_SIZE>::setup(max_size);
        test_aafl<T, CWARP_SIZE> ::iner_setup(max_size);
        test_delta<T, CWARP_SIZE>::iner_setup(max_size);
    }

    virtual void initializeData(int bit_length) {
            big_random_block_with_decreasing_values_for_aafl ((unsigned long)this->max_size, bit_length, this->host_data);
    }

    // Clean up before compression
    virtual void cleanBeforeCompress() {
        test_base<T, CWARP_SIZE>::cleanBeforeCompress();
        test_aafl<T, CWARP_SIZE> :: iner_cleanBeforeCompress();
        test_delta<T, CWARP_SIZE>:: iner_cleanBeforeCompress();
    }

    virtual void compressData(int bit_length) {
            container_uncompressed<T> udata = {this->dev_data, this->max_size};
            container_delta_aafl<T> cdata = {(make_unsigned_t<T> *) this->dev_out, this->max_size, this->dev_data_bit_lenght, this->dev_data_position_id, this->dev_data_compressed_data_register, (make_unsigned_t<T> *) this->dev_data_block_start};

            compress <T, CWARP_SIZE> (udata, cdata);
    }

    virtual void decompressData(int bit_length) {
            container_uncompressed<T> udata = {this->dev_data, this->max_size};
            container_delta_aafl<T> cdata = {(make_unsigned_t<T> *) this->dev_out, this->max_size, this->dev_data_bit_lenght, this->dev_data_position_id, NULL, (make_unsigned_t<T> *) this->dev_data_block_start};

            decompress <T, CWARP_SIZE> (cdata, udata);
    }

    virtual void print_compressed_data_size(){
        //TODO: fix this
        unsigned long tmp;
        cudaMemcpy(&tmp, this->dev_data_compressed_data_register, sizeof(unsigned long), cudaMemcpyDeviceToHost);
        printf("Comp ratio %f",  (float)this->data_size / (tmp + test_aafl<T, CWARP_SIZE>::compression_blocks_count * (sizeof(T) + sizeof(long) + sizeof(char))));
        printf(" %d %lu %ld %ld\n" , this->bit_length, this->max_size, this->data_size, this->compressed_data_size);
    }
};
