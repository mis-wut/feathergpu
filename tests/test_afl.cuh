#pragma once

#include "catch.hpp"
#include "tools/tools.cuh"
#include "feathergpu/fl/afl.cuh"
#include "tests/test_base.cuh"

template <typename T, int CWARP_SIZE> class test_afl: public test_base<T, CWARP_SIZE> {
    public:
    virtual void compressData(int bit_length) {
        run_afl_compress_gpu <T, CWARP_SIZE> (bit_length, this->dev_data, this->dev_out, this->max_size);
    }

    virtual void decompressData(int bit_length) {
        run_afl_decompress_gpu <T, CWARP_SIZE> (bit_length, this->dev_out, this->dev_data, this->max_size);
    }
};

template <typename T, int CWARP_SIZE> class test_afl_random_access: public test_afl<T, CWARP_SIZE> {
    public:
    virtual void decompressData(int bit_length) {
        run_afl_decompress_value_gpu <T, CWARP_SIZE> (bit_length, this->dev_out, this->dev_data, this->max_size);
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
        /* run_afl_compress_cpu <T, CWARP_SIZE> (bit_length, this->host_data, this->host_out, this->max_size); */
        container_fl<T> cdata = {(unsigned char) bit_length, (make_unsigned_t<T> *) this->host_out, this->max_size};
        container_uncompressed<T> udata = {this->host_data, this->max_size};
        /* run_afl_compress_cpu <T, CWARP_SIZE> (bit_length, this->host_data, this->host_out, this->max_size); */
        feather_cpu_launcher(afl_compress_cpu_kernel <T, CWARP_SIZE>, udata, cdata);
    }

    virtual void decompressData(int bit_length) {
        run_afl_decompress_cpu <T, CWARP_SIZE> (bit_length, this->host_out, this->host_data2, this->max_size);
    }
protected:
        T *host_out;
};

template <typename T, int CWARP_SIZE> class test_afl_random_access_cpu: public test_afl_cpu<T, CWARP_SIZE> {
public:
    virtual void compressData(int bit_length) {
        run_afl_compress_value_cpu <T, CWARP_SIZE> (bit_length, this->host_data, this->host_out, this->max_size);
    }

    virtual void decompressData(int bit_length) {
        run_afl_decompress_value_gpu <T, CWARP_SIZE> (bit_length, this->host_out, this->host_data2, this->max_size);
    }
protected:
        T *host_out;
};
