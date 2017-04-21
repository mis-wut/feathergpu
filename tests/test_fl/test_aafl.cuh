#pragma once

#include "test_base.cuh"
#include "feathergpu/fl/aafl.cuh"
#include "tools/tools.cuh"

template <typename T, int CWARP_SIZE> 
class test_aafl: public virtual test_base<T, CWARP_SIZE> 
{
    public: 
        virtual void allocateMemory() {
            test_base<T, CWARP_SIZE>::allocateMemory();
            iner_allocateMemory();
        }

        virtual void setup(unsigned long max_size) {
            test_base<T, CWARP_SIZE>::setup(max_size);
            iner_setup(max_size);
        }

        // Clean up before compression
        virtual void cleanBeforeCompress() {
            test_base<T, CWARP_SIZE>::cleanBeforeCompress();
            iner_cleanBeforeCompress();
        }

        virtual void compressData(int bit_length) {
            run_aafl_compress_gpu <T, CWARP_SIZE> (this->dev_data_compressed_data_register, this->dev_data_bit_lenght, this->dev_data_position_id, this->dev_data, this->dev_out, this->max_size);
        }

        virtual void decompressData(int bit_length) {
            run_aafl_decompress_gpu <T, CWARP_SIZE> (this->dev_data_bit_lenght, this->dev_data_position_id, this->dev_out, this->dev_data, this->max_size);
        }

        virtual void print_compressed_data_size(){
            //TODO: fix this
            unsigned long tmp;
            cudaMemcpy(&tmp, this->dev_data_compressed_data_register, sizeof(unsigned long), cudaMemcpyDeviceToHost);
                printf("Comp ratio %f",  (float)this->max_size / (tmp + this->compression_blocks_count * (sizeof(long) + sizeof(char))));
                printf(" %d %lu %ld %ld\n" , this->bit_length, this->max_size, this->data_size, this->compressed_data_size);
        }

    protected:
        unsigned char *dev_data_bit_lenght;
        unsigned long *dev_data_position_id;
        unsigned long compression_blocks_count;
        unsigned long *dev_data_compressed_data_register;

        virtual void iner_allocateMemory(){
            mmCudaMalloc(this->manager, (void **) &this->dev_data_position_id, compression_blocks_count * sizeof(unsigned long));
            mmCudaMalloc(this->manager, (void **) &this->dev_data_compressed_data_register, sizeof(long));
            mmCudaMalloc(this->manager, (void **) &this->dev_data_bit_lenght, compression_blocks_count * sizeof(unsigned char));
        }

        virtual void iner_cleanBeforeCompress(){
            cudaMemset(this->dev_data_compressed_data_register, 0, sizeof(unsigned long)); 
            cudaMemset(this->dev_data_bit_lenght, 0, compression_blocks_count * sizeof(unsigned char));
            cudaMemset(this->dev_data_position_id, 0, compression_blocks_count * sizeof(unsigned long));
        }

        virtual void iner_setup(unsigned long max_size) {
            this->compression_blocks_count = (this->compressed_data_size + (sizeof(T) * CWARP_SIZE) - 1) / (sizeof(T) * CWARP_SIZE);
        }
};

template <typename T, int CWARP_SIZE> 
class test_aafl_optimistic: public virtual test_aafl <T, CWARP_SIZE> {
    public:
        virtual void initializeData(int bit_length) {
            cudaMemset(this->host_data, 0, this->data_size); 
        }
};

template <typename T, int CWARP_SIZE> 
class test_aafl_pesymistic: public virtual test_aafl <T, CWARP_SIZE> {
    public:
        virtual void initializeData(int bit_length) {
            T max = std::numeric_limits<T>::max() >> 1;
            unsigned long i;
            for (i = 0; i < this->data_size/sizeof(T); ++i) {
                this->host_data[i] = max;
            }
        }
        virtual int run(unsigned long max_size, bool print = false, unsigned int fixed_bit_lenght=0)
        {
            return test_aafl<T, CWARP_SIZE>::run(max_size, print, sizeof(T) * 8-1);
        }
};
