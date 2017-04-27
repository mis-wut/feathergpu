#pragma once
#include "test_fl/test_afl.cuh"

template <typename T, int CWARP_SIZE> class test_afl_signed: public test_afl<T, CWARP_SIZE> {
    public:
    virtual void compressData(int bit_length) {
        container_uncompressed<T> udata = {this->dev_data, this->max_size};
        container_signed_fl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) this->dev_out, this->max_size};

        compress <T, CWARP_SIZE>(udata, cdata);
    }

    virtual void decompressData(int bit_length) {
        container_uncompressed<T> udata = {this->dev_data, this->max_size};
        container_signed_fl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) this->dev_out, this->max_size};

        decompress <T, CWARP_SIZE> (cdata, udata);
    }

    virtual void initializeData(int bit_length) {
        big_random_block(this->max_size, bit_length-1, this->host_data);

        // On signed types this will make all odd values negative
        if (std::numeric_limits<T>::is_signed)
            for (unsigned long i = 0; i < this->max_size; i++)
                if (i%2)
                    this->host_data[i] *= -1;
    }
};
