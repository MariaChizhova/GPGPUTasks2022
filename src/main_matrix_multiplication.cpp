#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/matrix_multiplication_cl.h"

#include <vector>
#include <iostream>


int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 1; // TODO пока тестируетесь удобно выставить единицу
    unsigned int M = 1024;
    unsigned int K = 1024;
    unsigned int N = 1024;
    const size_t gflops = ((size_t) M * K * N * 2) / (1000 * 1000 * 1000); // умножить на два, т.к. операция сложения и умножения

    std::vector<float> as(M*K, 0);
    std::vector<float> bs(K*N, 0);
    std::vector<float> cs1(M*N, 0);
    std::vector<float> cs2(M*N, 0);

    FastRandom r(M+K+N);
    for (unsigned int i = 0; i < as.size(); ++i) {
        as[i] = r.nextf();
    }
    for (unsigned int i = 0; i < bs.size(); ++i) {
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for M=" << M << ", K=" << K << ", N=" << N << "!" << std::endl;

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            for (int j = 0; j < M; ++j) {
                for (int i = 0; i < N; ++i) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        sum += as.data()[j * K + k] * bs.data()[k * N + i];
                    }
                    cs1.data()[j * N + i] = sum;
                }
            }
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << gflops / t.lapAvg() << " GFlops" << std::endl;
    }

    const std::vector<float> cs_cpu_reference = cs1;


    gpu::gpu_mem_32f as_gpu, bs_gpu, cs1_gpu, cs2_gpu;
    as_gpu.resizeN(M*K);
    bs_gpu.resizeN(K*N);
    cs1_gpu.resizeN(M*N);
    cs2_gpu.resizeN(M*N);

    as_gpu.writeN(as.data(), M*K);
    bs_gpu.writeN(bs.data(), K*N);

    ocl::Kernel matrix_multiplication1_kernel(matrix_multiplication, matrix_multiplication_length, "matrix_multiplication1");
    matrix_multiplication1_kernel.compile();

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            matrix_multiplication1_kernel.exec(gpu::WorkSize(16, 16, N, M), as_gpu, bs_gpu, cs1_gpu, M, K, N);

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << gflops / t.lapAvg() << " GFlops" << std::endl;
    }

    cs1_gpu.readN(cs1.data(), M*N);

    ocl::Kernel matrix_multiplication2_kernel(matrix_multiplication, matrix_multiplication_length, "matrix_multiplication2");
    matrix_multiplication2_kernel.compile();

    {
      timer t;
      for (int iter = 0; iter < benchmarkingIters; ++iter) {
        matrix_multiplication2_kernel.exec(gpu::WorkSize(4, 16, N / 4, M), as_gpu, bs_gpu, cs2_gpu, M, K, N);

        t.nextLap();
      }
      std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
      std::cout << "GPU: " << gflops / t.lapAvg() << " GFlops" << std::endl;
    }

    cs2_gpu.readN(cs2.data(), M*N);

    // Проверяем корректность результатов
    double diff_sum1 = 0;
    double diff_sum2 = 0;
    for (int i = 0; i < M * N; ++i) {
        double a1 = cs1[i];
        double a2 = cs2[i];
        double b = cs_cpu_reference[i];
        if (a1 != 0.0 && a2 != 0.0 && b != 0.0) {
          double diff1 = fabs(a1 - b) / std::max(fabs(a1), fabs(b));
          diff_sum1 += diff1;

          double diff2 = fabs(a2 - b) / std::max(fabs(a2), fabs(b));
          diff_sum2 += diff2;
        }
    }

    std::cout << "Multiplication 1:\n";
    double diff_avg1 = diff_sum1 / (M * N);
    std::cout << "Average difference: " << diff_avg1 * 100.0 << "%" << std::endl;
    if (diff_avg1 > 0.01) {
        std::cerr << "Too big difference!" << std::endl;
        return 1;
    }

    std::cout << "Multiplication 2:\n";
    double diff_avg2 = diff_sum2 / (M * N);
    std::cout << "Average difference: " << diff_avg2 * 100.0 << "%" << std::endl;
    if (diff_avg2 > 0.01) {
      std::cerr << "Too big difference!" << std::endl;
      return 1;
    }

    return 0;
}
