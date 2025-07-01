#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>

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
 * @brief CUDA 核函数：执行行除法 (保持不变)
 */
__global__ void division_kernel(float *d_m, int n, int k) {
    int j = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    float pivot_val = d_m[k * (n + 1) + k];
    if (j <= n) {
        d_m[k * (n + 1) + j] /= pivot_val;
    }
}

/**
 * @brief CUDA 核函数：按行划分策略执行消元
 * 每个线程块负责一行 (i) 的消元。
 * 块内线程负责不同列 (j) 的计算。
 */
template <int BLOCK_DIM>
__global__ void elimination_kernel_row_partition(float *d_m, int n, int k) {
    // 使用共享内存来存储当前行的消元因子 A[i][k]
    __shared__ float factor;
    
    // 每个块负责一行，行索引 i 由 blockIdx.x 决定
    int i = k + 1 + blockIdx.x; 

    // 块内第一个线程读取因子并存入共享内存
    if (threadIdx.x == 0) {
        factor = d_m[i * (n + 1) + k];
    }
    // 同步块内所有线程，确保 factor 已被所有线程读取
    __syncthreads();

    // 块内线程并行计算该行不同列的元素
    // j 是列索引
    for (int j = k + 1 + threadIdx.x; j <= n; j += BLOCK_DIM) {
        d_m[i * (n + 1) + j] -= factor * d_m[k * (n + 1) + j];
    }

    // 再次同步，确保所有减法操作完成
    __syncthreads();

    // 由块内第一个线程将 A[i][k] 置为 0 (内核融合)
    if (threadIdx.x == 0) {
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

// 启动器函数，根据运行时参数选择模板化内核
void launch_elimination_kernel(float* d_m, int n, int k, int block_dim) {
    // 网格维度：1D，大小为需要处理的行数
    int elim_rows = n - 1 - k;
    dim3 grid(elim_rows);
    // 线程块维度：1D，大小为命令行传入的 block_dim
    dim3 block(block_dim);

    switch (block_dim) {
        // --- 新增的小规模块 ---
        case 32:
            elimination_kernel_row_partition<32><<<grid, block>>>(d_m, n, k);
            break;
        case 64:
            elimination_kernel_row_partition<64><<<grid, block>>>(d_m, n, k);
            break;
        // --- 原有的大规模块 ---
        case 128:
            elimination_kernel_row_partition<128><<<grid, block>>>(d_m, n, k);
            break;
        case 256:
            elimination_kernel_row_partition<256><<<grid, block>>>(d_m, n, k);
            break;
        case 512:
            elimination_kernel_row_partition<512><<<grid, block>>>(d_m, n, k);
            break;
        default:
            std::cerr << "Unsupported block dimension: " << block_dim << std::endl;
            exit(1);
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size_N> <block_dim>" << std::endl;
        // --- 更新帮助信息 ---
        std::cerr << "Supported block_dim for this version: 32, 64, 128, 256, 512" << std::endl;
        return 1;
    }

    const int N = std::atoi(argv[1]);
    const int block_dim = std::atoi(argv[2]);

    // --- 更新参数检查 ---
    if (N <= 0 || (block_dim != 32 && block_dim != 64 && block_dim != 128 && block_dim != 256 && block_dim != 512)) {
        std::cerr << "Invalid arguments." << std::endl;
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
        int div_threads_needed = augmented_cols - 1 - k;
        int div_grid_size = (div_threads_needed + block_dim - 1) / block_dim;
        division_kernel<<<div_grid_size, block_dim>>>(d_m, matrix_rows, k);
        CUDA_CHECK(cudaGetLastError());
        
        float one = 1.0f;
        CUDA_CHECK(cudaMemcpy(d_m + k * augmented_cols + k, &one, sizeof(float), cudaMemcpyHostToDevice));

        // --- 消元步骤 (按行划分) ---
        if (matrix_rows - 1 - k > 0) {
            launch_elimination_kernel(d_m, matrix_rows, k, block_dim);
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