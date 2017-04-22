#pragma once

#include <type_traits>

template<typename T>
using make_unsigned_t = typename std::make_unsigned<T>::type;

template <typename T>
struct container_uncompressed {
    T *data;
    unsigned long length;
};

template <typename T>
struct container_fl {
    unsigned char bit_length;
    make_unsigned_t<T> *data;
    unsigned long length;
};

template <typename T>
struct container_delta_fl {
    // AFL
    unsigned char bit_length;
    make_unsigned_t<T> *data;
    unsigned long length;
    // DELTA
    make_unsigned_t<T> *block_start;
};

template <typename T>
struct container_aafl {
    make_unsigned_t<T> *data;
    unsigned long length;

    unsigned char *warp_bit_lenght;
    unsigned long *warp_position_id;

    unsigned long *data_register;
};

template <typename T>
struct container_delta_aafl {
    make_unsigned_t<T> *data;
    unsigned long length;

    unsigned char *warp_bit_lenght;
    unsigned long *warp_position_id;

    unsigned long *data_register;

    make_unsigned_t<T> *block_start;
};

template <typename T>
struct container_pafl {
    unsigned char bit_length;
    make_unsigned_t<T> *data;
    unsigned long length;

    make_unsigned_t<T> *patch_values;
    unsigned long *patch_index;
    unsigned long *patch_count;
};

template <typename T>
struct container_delta_pafl {
    unsigned char bit_length;
    make_unsigned_t<T> *data;
    unsigned long length;

    make_unsigned_t<T> *patch_values;
    unsigned long *patch_index;
    unsigned long *patch_count;

    make_unsigned_t<T> *block_start;
};
