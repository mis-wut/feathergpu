#pragma once
//TODO: better distinguishing between signed/unsigned versions

__forceinline__ __device__ unsigned int get_lane_id (int warp_size=32)
{
    return threadIdx.x % warp_size;
}

template <typename T>
__inline__ __device__ T warpAllReduceMax(T val) {

    val = max(val, __shfl_xor(val,16));
    val = max(val, __shfl_xor(val, 8));
    val = max(val, __shfl_xor(val, 4));
    val = max(val, __shfl_xor(val, 2));
    val = max(val, __shfl_xor(val, 1));

    /*int m = val;*/
    /*for (int mask = warpSize/2; mask > 0; mask /= 2) {*/
        /*m = __shfl_xor(val, mask);*/
        /*val = m > val ? m : val;*/
    /*}*/
    return val;
}

__device__ inline long shfl_up(long value, int i, int width=32)
{
    int lo, hi;
    asm volatile("mov.b64 {%0,%1}, %2;":"=r"(lo),"=r"(hi):"l"(value));

    lo =  __shfl_up(lo, i, width);
    hi =  __shfl_up(hi, i, width);

    asm volatile("mov.b64 %0,{%1,%2};":"=l"(value):"r"(lo),"r"(hi));

    return value;
}

__device__ inline unsigned long shfl_up(unsigned long value, int i, int width=32)
{
    unsigned int lo, hi;
    asm volatile("mov.b64 {%0,%1}, %2;":"=r"(lo),"=r"(hi):"l"(value));

    lo =  __shfl_up(lo, i, width);
    hi =  __shfl_up(hi, i, width);

    asm volatile("mov.b64 %0,{%1,%2};":"=l"(value):"r"(lo),"r"(hi));

    return value;
}

template <typename T>
__device__ inline T shfl_prefix_sum(T value, int width=32)
{
    int lane_id = get_lane_id();

    // Now accumulate in log2(32) steps
#pragma unroll
    for(int i=1; i<=width; i*=2) {
        T n = __shfl_up(value, i);
        if(lane_id >= i) value += n;
    }

    return value;
}

template <typename T>
__device__ inline T shfl_get_value(T value, int laneId, int width=32)
{
    return __shfl(value, laneId, width);
}

__device__ inline long shfl_get_value(long value, int laneId, int width=32)
{
    int lo, hi;
    asm volatile("mov.b64 {%0,%1}, %2;":"=r"(lo),"=r"(hi):"l"(value));

    lo =  __shfl(lo, laneId, width);
    hi =  __shfl(hi, laneId, width);

    asm volatile("mov.b64 %0,{%1,%2};":"=l"(value):"r"(lo),"r"(hi));

    return value;
}

__device__ inline unsigned long shfl_get_value(unsigned long value, int laneId, int width=32)
{
    int lo, hi;
    asm volatile("mov.b64 {%0,%1}, %2;":"=r"(lo),"=r"(hi):"l"(value));

    lo =  __shfl(lo, laneId, width);
    hi =  __shfl(hi, laneId, width);

    asm volatile("mov.b64 %0,{%1,%2};":"=l"(value):"r"(lo),"r"(hi));

    return value;
}
