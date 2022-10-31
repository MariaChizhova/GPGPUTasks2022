#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

__kernel void prefix_sum(__global unsigned int *as, __global unsigned int *bs, unsigned int bit, unsigned int n) {
    unsigned int id = get_global_id(0);
    if (id >= n) {
        return;
    }
    if (((id + 1) >> bit) & 1) {
        bs[id] += as[((id + 1) >> bit) - 1];
    }
}

__kernel void part_sum(__global unsigned int *as, __global unsigned int *next_as, unsigned int n) {
    unsigned int id = get_global_id(0);
    if (id >= n) {
        return;
    }
    next_as[id] = as[2 * id] + as[2 * id + 1];
}