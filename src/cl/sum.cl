#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define VALUES_PER_WORK_ITEM 32
#define WORK_GROUP_SIZE 128

__kernel void sum1(__global unsigned int *a, __global unsigned int* res,  unsigned int n) {
    int id = get_global_id(0);
    if (id < n) {
        atomic_add(res, a[id]);
    }
}

__kernel void sum2(__global unsigned int *a, __global unsigned int* res,  unsigned int n) {
    int id = get_global_id(0);
    unsigned int sum = 0;
    for (int i = 0; i < VALUES_PER_WORK_ITEM; i++) {
        int idx = id * VALUES_PER_WORK_ITEM + i;
        if (idx >= n) {
           break;
        }
        sum += a[idx];
    }
    atomic_add(res, sum);
}

__kernel void sum3(__global unsigned int *a, __global unsigned int* res,  unsigned int n) {
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int group_size = get_local_size(0);
    unsigned int sum = 0;
    for (int i = 0; i < VALUES_PER_WORK_ITEM; i++) {
        int idx = group_id * group_size * VALUES_PER_WORK_ITEM + i * group_size + local_id;
        if (idx >= n) {
          break;
        }
        sum += a[idx];
    }
    atomic_add(res, sum);
}

__kernel void sum4(__global unsigned int *a, __global unsigned int* res,  unsigned int n) {
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    __local int local_a[WORK_GROUP_SIZE];
    if (global_id < n) {
      local_a[local_id] = a[global_id];
    } else {
      local_a[local_id] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (!local_id) {
        unsigned int sum = 0;
        for (int i = 0; i < WORK_GROUP_SIZE; i++) {
            sum += local_a[i];
        }
        atomic_add(res, sum);
    }
}

__kernel void sum5(__global unsigned int *a, __global unsigned int* res,  unsigned int n) {
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    __local int local_a[WORK_GROUP_SIZE];
    if (global_id < n) {
      local_a[local_id] = a[global_id];
    } else {
      local_a[local_id] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = WORK_GROUP_SIZE; i > 1; i /= 2) {
        if (2 * local_id < i) {
            local_a[local_id] = local_a[local_id] + local_a[local_id + i / 2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (!local_id) {
        atomic_add(res, local_a[0]);
    }
}