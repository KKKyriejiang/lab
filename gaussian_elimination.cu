#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>

// CUDA 块大小
#define BLOCK_SIZE 256

// 错误检查宏
#define CUDA_CHECK(err) { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err_) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

/**
 * @brief CUDA 核函数：执行行除法。
 * 每个线程负责第 k 行中的一列。
 */
__global__ void division_kernel(float *d_m, int n, int k) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    float pivot_val = d_m[k * (n + 1) + k];

    // 线程索引 j 对应于列索引
    // 我们需要对第 k 行中从 k+1 到 n 的所有列进行操作
    if (j > k && j <= n) {
        d_m[k * (n + 1) + j] /= pivot_val;
    }
}

/**
 * @brief CUDA 核函数：执行消元步骤。
 * 每个线程块负责一行 (i) 的消元。
 * 块内线程负责不同列 (j) 的计算。
 */
__global__ void elimination_kernel(float *d_m, int n, int k) {
    // 使用共享内存来存储当前行 (i) 的消元因子 A[i][k]
    __shared__ float factor;
    
    // 每个块负责一行，行索引 i 由 blockIdx.x 决定
    int i = k + 1 + blockIdx.x; 

    // 块内第一个线程读取因子并存入共享内存
    if (threadIdx.x == 0) {
        factor = d_m[i * (n + 1) + k];
    }
    // 同步块内所有线程，确保 factor 已被读取
    __syncthreads();

    // 块内线程并行计算该行不同列的元素
    // j 是列索引
    int j = k + 1 + threadIdx.x;
    
    if (i < n && j <= n) {
        d_m[i * (n + 1) + j] -= factor * d_m[k * (n + 1) + j];
    }

    // 再次同步，确保所有减法操作完成
    __syncthreads();

    // 由块内第一个线程将 A[i][k] 置为 0
    if (i < n && threadIdx.x == 0) {
        d_m[i * (n + 1) + k] = 0.0f;
    }
}


// 在 CPU 上初始化矩阵和向量
void init_matrix(std::vector<float>& m, int n) {
    srand(time(0));
    for (int i = 0; i < n; ++i) {
        float row_sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            m[i * (n + 1) + j] = static_cast<float>(rand()) / RAND_MAX * 10.0f;
            row_sum += fabs(m[i * (n + 1) + j]);
        }
        m[i * (n + 1) + i] += row_sum; 
        m[i * (n + 1) + n] = static_cast<float>(rand()) / RAND_MAX * 10.0f;
    }
}

// 在 CPU 上执行高斯消去（用于基准测试）
void cpu_gaussian_elimination(std::vector<float>& m, int n) {
    for (int k = 0; k < n; ++k) {
        float pivot = m[k * (n + 1) + k];
        for (int j = k; j <= n; ++j) {
            m[k * (n + 1) + j] /= pivot;
        }
        for (int i = k + 1; i < n; ++i) {
            float factor = m[i * (n + 1) + k];
            for (int j = k; j <= n; ++j) {
                m[i * (n + 1) + j] -= factor * m[k * (n + 1) + j];
            }
        }
    }
}


int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size_N>" << std::endl;
        return 1;
    }

    const int N = std::atoi(argv[1]);
    if (N <= 0) {
        std::cerr << "Matrix size must be a positive integer." << std::endl;
        return 1;
    }

    const int matrix_rows = N;
    const int augmented_cols = N + 1;
    const size_t matrix_size_bytes = matrix_rows * augmented_cols * sizeof(float);

    // --- CPU 版本 ---
    std::vector<float> h_m_cpu(matrix_rows * augmented_cols);
    init_matrix(h_m_cpu, matrix_rows);
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_gaussian_elimination(h_m_cpu, matrix_rows);
    auto cpu_stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_stop - cpu_start;
    std::cout << "CPU_TIME:" << cpu_duration.count() << std::endl;


    // --- GPU 版本 ---
    std::vector<float> h_m_gpu(matrix_rows * augmented_cols);
    init_matrix(h_m_gpu, matrix_rows);

    float* d_m = nullptr;
    CUDA_CHECK(cudaMalloc(&d_m, matrix_size_bytes));
    CUDA_CHECK(cudaMemcpy(d_m, h_m_gpu.data(), matrix_size_bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));

    for (int k = 0; k < matrix_rows; ++k) {
        // --- 除法步骤 ---
        // 启动足够的线程来覆盖从 k+1 到 N 的所有列
        int div_threads = augmented_cols - 1 - k;
        int div_grid_size = (div_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
        division_kernel<<<div_grid_size, BLOCK_SIZE>>>(d_m, matrix_rows, k);
        CUDA_CHECK(cudaGetLastError());
        
        // 主元自己最后处理, 使用 cudaMemsetDevice 效率更高
        float one = 1.0f;
        CUDA_CHECK(cudaMemcpy(d_m + k * augmented_cols + k, &one, sizeof(float), cudaMemcpyHostToDevice));

        // --- 消元步骤 ---
        // 为从 k+1 到 N-1 的每一行启动一个线程块
        int elim_rows = matrix_rows - 1 - k;
        if (elim_rows > 0) {
            // 块内线程数，处理从 k+1 到 N 的列
            int threads_per_block = augmented_cols - 1 - k;
            // 确保不超过最大线程数
            threads_per_block = (threads_per_block > BLOCK_SIZE) ? BLOCK_SIZE : threads_per_block;

            elimination_kernel<<<elim_rows, threads_per_block>>>(d_m, matrix_rows, k);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "GPU_TIME:" << milliseconds << std::endl;

    CUDA_CHECK(cudaFree(d_m));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
