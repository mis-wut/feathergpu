#pragma once
#include "tests/test_afl.cuh"
#include "feathergpu/fl/afl_signed_experimental.cuh"

template <typename T, int CWARP_SIZE> class test_afl_signed: public test_afl<T, CWARP_SIZE> {
    public:
    virtual void compressData(int bit_length) {
        run_afl_compress_signed_gpu <T, CWARP_SIZE> (bit_length, this->dev_data, this->dev_out, this->max_size);
    }

    virtual void decompressData(int bit_length) {
        run_afl_decompress_signed_gpu <T, CWARP_SIZE> (bit_length, this->dev_out, this->dev_data, this->max_size);
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
