#pragma once
#include "test_fl/test_afl.cuh"
#include "feathergpu/bitmap/wah_experimental.cuh"

#define CONST_WORD(T,C) (((2 + T) << 30) | C)

void compress_wah(unsigned int *data, unsigned int size, unsigned int *cmp_data)
{
    unsigned int curr_bset = 0,
                 curr_word,
                 k = 0, tmp;

    unsigned int i;
    for (i = 0; i < size; ++i) {
        curr_word = data[i];
        // Take k remaining bits from previous word | 31 - k bits from current word
        curr_bset |= GETNBITS(curr_word, 31 - k);

        // create new compressed word (Literal or RLE)
        if(curr_bset == 0){
            cmp_data[i] = CONST_WORD(0,1);
            /* printf("RLE 0\n"); //zeros */
        } else if(curr_bset == NBITSTOMASK(31)) {
            cmp_data[i] = CONST_WORD(1,1);
            /* printf("RLE 1 %d\n", curr_bset); //ones */
        } else {
            cmp_data[i] = curr_bset;
            /* printf("Literal %u\n", curr_bset); //literal */
        }
        // Set k = 31 - k; store k remianing bits from current word
        tmp = 31 - k;
        k = 32 - (31 - k);
        curr_bset = (curr_word >> tmp) << (31 - k);
    }
    cmp_data[i+1] = curr_bset;
}

template <typename T, int CWARP_SIZE> class test_wah: public test_afl<T, CWARP_SIZE> {
    public:
    virtual void compressData(int bit_length) {
        run_wah_compress <CWARP_SIZE> (this->dev_data, this->dev_out, this->max_size);
    }

    virtual void decompressData(int bit_length) {
        compress_wah(this->host_data, this->data_size, this->host_data2);

        //TODO: temporary reconstruct from GPU data here
        // read 32 ints
        // construct bit array
        /* run_wah_decompress_signed_gpu <T, CWARP_SIZE> (bit_length, this->dev_out, this->dev_data, this->max_size); */
    }

    virtual void initializeData(int bit_length) {
            for (unsigned long i = 0; i < this->max_size; i++)
                    this->host_data[i] = 1;
    }

    virtual void allocateMemory() {
        mmCudaMallocHost(this->manager, (void**)&this->host_data,  this->data_size + 32);
        mmCudaMallocHost(this->manager, (void**)&this->host_data2, this->data_size + 32);

        mmCudaMalloc(this->manager, (void **) &this->dev_out, this->compressed_data_size);
        mmCudaMalloc(this->manager, (void **) &this->dev_data, this->data_size + 32);
    }
};
