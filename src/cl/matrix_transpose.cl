#define TILE_SIZE 17

__kernel void matrix_transpose(__global float *a, __global float *at, unsigned int m, unsigned int k)
{
    int global_i = get_global_id(0);
    int global_j = get_global_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE];
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    tile[local_j][local_i] = a[global_j * k + global_i];

    barrier(CLK_LOCAL_MEM_FENCE);

    int i = global_i - local_i + local_j;
    int j = global_j + local_i - local_j;

    at[i * m + j] = tile[local_i][local_j];
}