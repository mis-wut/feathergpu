#pragma once

#include <type_traits>

template<typename T>
using make_unsigned_t = typename std::make_unsigned<T>::type;

template <typename T>
struct container_fl {
    unsigned char bit_length;
    make_unsigned_t<T> *data;
    unsigned long length;
};

template <typename T>
struct container_delta_fl {
    unsigned char bit_length;
    make_unsigned_t<T> *data;
    make_unsigned_t<T> *block_start;
    unsigned long length;
};

template <typename T>
struct container_uncompressed {
    T *data;
    unsigned long length;
};

template <typename T>
struct container_aafl {
    const unsigned char bit_length;
    make_unsigned_t<T> *data;
    unsigned long length;

    unsigned char *warp_bit_lenght;
    unsigned long *warp_position_id;

    unsigned long *compressed_data_register;
};
