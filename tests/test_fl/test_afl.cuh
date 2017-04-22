#pragma once

#include "catch.hpp"
#include "tools/tools.cuh"
#include "feathergpu/fl/afl.cuh"
#include "feathergpu/fl/afl_old_wrappers.cuh"
#include "feathergpu/util/launcher.cuh"
#include "test_base.cuh"

template <typename T, int CWARP_SIZE> class test_afl: public test_base<T, CWARP_SIZE> {
    public:
    virtual void compressData(int bit_length) {

        const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
        const unsigned long block_number = (this->max_size + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));
        container_uncompressed<T> udata = {this->dev_data, this->max_size};
        container_fl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) this->dev_out, this->max_size};
        afl_compress_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (udata, cdata);
    }

    virtual void decompressData(int bit_length) {
        const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
        const unsigned long block_number = (this->max_size + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));
        container_uncompressed<T> udata = {this->dev_data, this->max_size};
        container_fl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) this->dev_out, this->max_size};
        afl_decompress_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (cdata, udata);
    }
};

template <typename T, int CWARP_SIZE> class test_afl_random_access: public test_afl<T, CWARP_SIZE> {
    public:
    virtual void decompressData(int bit_length) {
        const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
        const unsigned long block_number = (this->max_size + block_size * CWORD_SIZE(T) - 1) / (block_size);
        container_uncompressed<T> udata = {this->dev_data, this->max_size};
        container_fl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) this->dev_out, this->max_size};
        afl_decompress_value_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (cdata, udata);
    }
};


template <typename T, int CWARP_SIZE> class test_afl_cpu: public test_afl<T, CWARP_SIZE> {
public:

   virtual void allocateMemory() {
        mmCudaMallocHost(this->manager, (void**)&this->host_data,  this->data_size);
        mmCudaMallocHost(this->manager, (void**)&this->host_data2, this->data_size);

        mmCudaMallocHost(this->manager, (void **)&this->host_out, this->data_size);
    }

    virtual void transferDataToGPU() {}
    virtual void cleanBeforeCompress() {}
    virtual void errorCheck() {}
    virtual void cleanBeforeDecompress() {}
    virtual void transferDataFromGPU() {}

    virtual void compressData(int bit_length) {
        container_fl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) this->host_out, this->max_size};
        container_uncompressed<T> udata = {this->host_data, this->max_size};
        afl_compress_cpu_kernel <T, CWARP_SIZE>( udata, cdata);
    }

    virtual void decompressData(int bit_length) {
        container_fl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) this->host_out, this->max_size};
        container_uncompressed<T> udata = {this->host_data2, this->max_size};
        afl_decompress_cpu_kernel <T, CWARP_SIZE> (cdata, udata);
    }
protected:
        T *host_out;
};

template <typename T, int CWARP_SIZE> class test_afl_random_access_cpu: public test_afl_cpu<T, CWARP_SIZE> {
public:
    virtual void compressData(int bit_length) {
        container_uncompressed<T> udata = {this->host_data, this->max_size};
        container_fl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) this->host_out, this->max_size};
        afl_compress_value_cpu_kernel <T, CWARP_SIZE> ( udata, cdata);
    }

    virtual void decompressData(int bit_length) {
        container_uncompressed<T> udata = {this->host_data2, this->max_size};
        container_fl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) this->host_out, this->max_size};
        afl_decompress_value_cpu_kernel <T, CWARP_SIZE> ( cdata, udata);
    }
protected:
        T *host_out;
};
