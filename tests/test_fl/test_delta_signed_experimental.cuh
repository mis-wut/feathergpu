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
            container_uncompressed<T> udata = {this->dev_data, this->max_size};
            container_signed_delta_fl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) this->dev_out, this->max_size, (make_unsigned_t<T> *) this->dev_data_block_start};

            compress<T, CWARP_SIZE> (udata, cdata);
        }

        virtual void decompressData(int bit_length) {
            container_uncompressed<T> udata = {this->dev_data, this->max_size};
            container_signed_delta_fl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) this->dev_out, this->max_size, (make_unsigned_t<T> *) this->dev_data_block_start};

            decompress <T, CWARP_SIZE> (cdata, udata);
        }
};
