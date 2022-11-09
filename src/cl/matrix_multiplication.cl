#define TILE_SIZE 16
#define THREAD_WORK 4

__kernel void matrix_multiplication1(__global float* a, __global float* b, __global float* c,
                                     unsigned int m, unsigned int k, unsigned int n) {
    int global_i = get_global_id(0);
    int global_j = get_global_id(1);

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    for (int tileK = 0; tileK * TILE_SIZE < k; ++tileK) {
        tileA[local_j][local_i] = a[global_j * k + TILE_SIZE * tileK + local_i];
        tileB[local_j][local_i] = b[(TILE_SIZE * tileK + local_j) * n + global_i];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int l = 0; l < TILE_SIZE; ++l) {
            sum += tileA[local_j][l] * tileB[l][local_i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_j < m && global_i < n) {
      c[global_j * n + global_i] = sum;
    }
}


__kernel void matrix_multiplication2(__global float* a, __global float* b, __global float* c,
                                     unsigned int m, unsigned int k, unsigned int n) {
    int global_i = get_global_id(0);
    int global_j = get_global_id(1);

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float sum[THREAD_WORK];

    for (int w = 0; w < THREAD_WORK; ++w) {
        sum[w] = 0.0f;
    }

    for (int tileK = 0; tileK * TILE_SIZE < k; ++tileK) {
        for (int w = 0; w < THREAD_WORK; w++) {
             tileA[local_j][local_i + w * TILE_SIZE / THREAD_WORK] = a[global_j * k + TILE_SIZE * tileK + local_i + w * TILE_SIZE / THREAD_WORK];
             tileB[local_j][local_i + w * TILE_SIZE / THREAD_WORK] = b[(TILE_SIZE * tileK + local_j) * n + global_i + w * TILE_SIZE / THREAD_WORK];
        }

      barrier(CLK_LOCAL_MEM_FENCE);

      for (int l = 0; l < TILE_SIZE; ++l) {
            float tmp = tileA[local_j][l];
            for (int w = 0; w < THREAD_WORK; ++w) {
                sum[w] += tileB[l][local_i + TILE_SIZE / THREAD_WORK * w] * tmp;
            }
      }

      barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int w = 0; w < THREAD_WORK; w++) {
        if (global_j < m && global_i < n) {
          c[global_j * n + global_i + w * TILE_SIZE / THREAD_WORK] = sum[w];
        }
    }
}