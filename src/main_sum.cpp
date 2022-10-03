#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include "cl/sum_cl.h"

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)
unsigned int workGroupSize = 128;

void sum(gpu::Device& device,
         std::vector<unsigned int>& a,
         const std::string& name,
         std::size_t iters,
         unsigned int globalWorkSize,
         unsigned int reference_sum,
         unsigned int n) {
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();
    gpu::gpu_mem_32u a_gpu, res_gpu;
    a_gpu.resizeN(n);
    res_gpu.resizeN(1);
    a_gpu.writeN(a.data(), n);
    ocl::Kernel baseline(sum_kernel, sum_kernel_length, name);
    baseline.compile();
    timer t;
    for (int iter = 0; iter < iters; ++iter) {
        unsigned int res = 0;
        res_gpu.writeN(&res, 1);
        baseline.exec(gpu::WorkSize(workGroupSize, globalWorkSize), a_gpu,  res_gpu, n);
        res_gpu.readN(&res, 1);
        EXPECT_THE_SAME(reference_sum, res, "GPU " + name + " results differ from cpu results!");
        t.nextLap();
    }
    std::cout << name + ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << name + ": " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
}

int main(int argc, char **argv)
{
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100*1000*1000;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / n);
        reference_sum += as[i];
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        // TODO: implement on OpenCL
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        unsigned int globalWorkSize = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        const size_t iters = 100;
        sum(device, as, "sum1", iters, globalWorkSize, reference_sum, n);
        sum(device, as, "sum2", iters, globalWorkSize, reference_sum, n);
        sum(device, as, "sum3", iters, globalWorkSize, reference_sum, n);
        sum(device, as, "sum4", iters, globalWorkSize, reference_sum, n);
        sum(device, as, "sum5", iters, globalWorkSize, reference_sum, n);
    }
}
