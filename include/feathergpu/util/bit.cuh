#pragma once
#include "feathergpu/util/macro.cuh"
//TODO: better distinguishing between signed/unsigned versions
template <typename T>
__device__ __host__ __forceinline__ T SETNPBITS( T *source, T value, const unsigned int num_bits, const unsigned int bit_start)
{
    T mask = BITMASK(T, num_bits);
    *source &= ~(mask<<bit_start); // clear space in source
    *source |= (value & mask) << bit_start; // set values
    return *source;
}

__device__ __host__ __forceinline__ unsigned int GETNPBITS( int source, unsigned int num_bits, unsigned int bit_start)
{
#if __CUDA_ARCH__ > 200  // This improves performance
    unsigned int bits;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"((unsigned int) source), "r"(bit_start), "r"(num_bits));
    return bits;
#else
    return ((source>>bit_start) & NBITSTOMASK(num_bits));
#endif
}

__device__ __host__ __forceinline__ unsigned int GETNPBITS( unsigned int source, unsigned int num_bits, unsigned int bit_start)
{
#if __CUDA_ARCH__ > 200  // This improves performance
    unsigned int bits;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"((unsigned int) source), "r"(bit_start), "r"(num_bits));
    return bits;
#else
    return ((source>>bit_start) & NBITSTOMASK(num_bits));
#endif
}

__device__ __host__ __forceinline__ unsigned long GETNPBITS( long source, unsigned int num_bits, unsigned int bit_start)
{
#if __CUDA_ARCH__ > 200  // This improves performance
    unsigned long bits;
    asm("bfe.u64 %0, %1, %2, %3;" : "=l"(bits) : "l"((unsigned long) source), "r"(bit_start), "r"(num_bits));
    return bits;
#else
    return ((source>>bit_start) & LNBITSTOMASK(num_bits));
#endif
}

__device__ __host__ __forceinline__ unsigned long GETNPBITS( unsigned long source, unsigned int num_bits, unsigned int bit_start)
{
#if __CUDA_ARCH__ > 200  // This improves performance
    unsigned long bits;
    asm("bfe.u64 %0, %1, %2, %3;" : "=l"(bits) : "l"((unsigned long) source), "r"(bit_start), "r"(num_bits));
    return bits;
#else
    return ((source>>bit_start) & LNBITSTOMASK(num_bits));
#endif
}

__device__ __host__ __forceinline__ unsigned long GETNBITS( long source, unsigned int num_bits)
{
#if __CUDA_ARCH__ > 200  // Use bfe implementation
    return GETNPBITS(source, num_bits, 0);
#else // In other case this will be faster
    return ((source) & LNBITSTOMASK(num_bits));
#endif
}

__device__ __host__ __forceinline__ unsigned long GETNBITS( unsigned long source, unsigned int num_bits)
{
#if __CUDA_ARCH__ > 200  // Use bfe implementation
    return GETNPBITS(source, num_bits, 0);
#else // In other case this will be faster
    return ((source) & LNBITSTOMASK(num_bits));
#endif
}

__device__ __host__ __forceinline__ unsigned int GETNBITS( int source, unsigned int num_bits)
{
#if __CUDA_ARCH__ > 200  // Use bfe implementation
    return GETNPBITS(source, num_bits, 0);
#else // In other case this will be faster
    return ((source) & NBITSTOMASK(num_bits));
#endif
}

__device__ __host__ __forceinline__ unsigned int GETNBITS( unsigned int source, unsigned int num_bits)
{
#if __CUDA_ARCH__ > 200  // Use bfe implementation
    return GETNPBITS(source, num_bits, 0);
#else // In other case this will be faster
    return ((source) & NBITSTOMASK(num_bits));
#endif
}

__device__ __host__ __forceinline__ unsigned int BITLEN(unsigned int word)
{
    unsigned int ret=0;
#if __CUDA_ARCH__ > 200
    asm volatile ("bfind.u32 %0, %1;" : "=r"(ret) : "r"(word));
#else
    while (word >>= 1)
      ret++;
#endif
   return ret > 64 ? 0 : ret;
}

__device__ __host__ __forceinline__ unsigned int BITLEN(unsigned long word)
{
    unsigned int ret=0;
#if __CUDA_ARCH__ > 200
    asm volatile ("bfind.u64 %0, %1;" : "=r"(ret) : "l"(word));
#else
    while (word >>= 1)
      ret++;
#endif
   return ret > 64 ? 0 : ret;
}

__device__ __host__ __forceinline__ unsigned int BITLEN(int word)
{
    unsigned int ret=0;
#if __CUDA_ARCH__ > 200
    asm volatile ("bfind.s32 %0, %1;" : "=r"(ret) : "r"(word));
#else
    while (word >>= 1)
      ret++;
#endif
   return ret > 64 ? 0 : ret;
}

__device__ __host__ __forceinline__ unsigned int BITLEN(long word)
{
    unsigned int ret=0;
#if __CUDA_ARCH__ > 200
    asm volatile ("bfind.s64 %0, %1;" : "=r"(ret) : "l"(word));
#else
    while (word >>= 1)
      ret++;
#endif
   return ret > 64 ? 0 : ret;
}

__host__ __device__
inline int ALT_BITLEN( int v)
{
    register unsigned int r; // result of log2(v) will go here
    register unsigned int shift;

    r =     (v > 0xFFFF) << 4; v >>= r;
    shift = (v > 0xFF  ) << 3; v >>= shift; r |= shift;
    shift = (v > 0xF   ) << 2; v >>= shift; r |= shift;
    shift = (v > 0x3   ) << 1; v >>= shift; r |= shift;
    r |= (v >> 1);
    return r+1;
}
