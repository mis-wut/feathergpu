#pragma once
#include "test_base.cuh"
#include "feathergpu/fl/default.cuh"

template <typename T, char CWARP_SIZE>
class test_pafl: public virtual test_base<T, CWARP_SIZE>
{
    public:
        virtual void allocateMemory() {
            test_pafl <T, CWARP_SIZE>::iner_allocateMemory();
            test_base <T, CWARP_SIZE>::allocateMemory();
        }

        virtual void iner_allocateMemory() {
            mmCudaMalloc(this->manager, (void **) &this->dev_data_patch_values, outlier_data_size);
            mmCudaMalloc(this->manager, (void **) &this->dev_data_patch_index, (outlier_count + 1024) * sizeof(unsigned long));
            mmCudaMalloc(this->manager, (void **) &this->dev_data_patch_count, sizeof(unsigned long));
        }

        virtual void setup(unsigned long max_size) {
            test_base <T, CWARP_SIZE>::setup(max_size);
            test_pafl <T, CWARP_SIZE>::iner_setup(max_size);
        }

        void iner_setup(unsigned long max_size) {
            this->outlier_percent = 0.1;

            this->outlier_count = max_size * this->outlier_percent;
            this->outlier_data_size = (this->outlier_count + 1024) * sizeof(T);
        }

        virtual void initializeData(int bit_length) {
            int outlier_bits=1;
            big_random_block_with_outliers(this->max_size, this->outlier_count, bit_length, outlier_bits, this->host_data);
        }

        // Clean up before compression
        virtual void cleanBeforeCompress() {
            test_base <T, CWARP_SIZE>::cleanBeforeCompress();
            test_pafl <T, CWARP_SIZE>::iner_cleanBeforeCompress();
        }

        virtual void iner_cleanBeforeCompress() {
            cudaMemset(this->dev_data_patch_count, 0, sizeof(unsigned long));
            cudaMemset(this->dev_data_patch_values, 0, outlier_data_size);
            cudaMemset(this->dev_data_patch_index, 0, (outlier_count +1024) * sizeof(long));
        }

        virtual void compressData(int bit_length) {
            container_uncompressed<T> udata = {this->dev_data, this->max_size};
            container_pafl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) this->dev_out, this->max_size, (make_unsigned_t<T> *) this->dev_data_patch_values, this->dev_data_patch_index, this->dev_data_patch_count};

            compress <T, CWARP_SIZE> (udata, cdata);
        }

        virtual void decompressData(int bit_length) {
            container_uncompressed<T> udata = {this->dev_data, this->max_size};
            container_pafl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) this->dev_out, this->max_size, (make_unsigned_t<T> *) this->dev_data_patch_values, this->dev_data_patch_index, this->dev_data_patch_count};

            decompress <T, CWARP_SIZE> (cdata, udata);
        }

    protected:
        unsigned long *dev_data_patch_index;
        unsigned long *dev_data_patch_count;
        T *dev_data_patch_values;

        unsigned long outlier_count;
        unsigned long outlier_data_size;
        float outlier_percent;
};

template <typename T, char CWARP_SIZE>
class test_pafl_optimistic: public virtual test_pafl<T, CWARP_SIZE>
{
public:
        virtual void initializeData(int bit_length) {
            big_random_block(this->max_size, bit_length, this->host_data);
        }
};
