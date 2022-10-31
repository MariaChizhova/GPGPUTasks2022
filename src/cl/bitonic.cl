#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define GROUP_SIZE 128

void local_swap(__local float* l, __local float* r, int increasing) {
    float a = *l;
    float b = *r;
    if ((increasing && a > b) || (!increasing && a < b)) {
        *l = b;
        *r = a;
    }
}

void global_swap(__global float* l, __global float* r, int increasing) {
    float a = *l;
    float b = *r;
    if ((increasing && a > b) || (!increasing && a < b)) {
        *l = b;
        *r = a;
    }
}

__kernel void bitonic_local(__global float *as, unsigned n, unsigned level, unsigned size) {
    unsigned lid = get_local_id(0);
    unsigned id = get_global_id(0);
    __local float data[GROUP_SIZE];
    if (id < n) {
        data[lid] = as[id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int is_down = (id / size) % 2 == 0;
    for (; level > 0; level /= 2) {
        if (id + level < n && id % (2 * level) < level) {
            local_swap(&data[lid], &data[lid + level], is_down);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (id < n) {
        as[id] = data[lid];
    }
}

__kernel void bitonic(__global float *as, unsigned n, unsigned level_size, unsigned size) {
    unsigned int id = get_global_id(0);
    int is_down = (id / size) % 2 == 0;
    if (id + level_size >= n || id % (2 * level_size) >= level_size) {
        return;
    }
    global_swap(&as[id], &as[id + level_size], is_down);
}