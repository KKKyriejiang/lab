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

// 定义循环展开因子
#define UNROLL_FACTOR 4

/**
 * @brief CUDA 核函数：执行行除法
 */
__global__ void division_kernel(float *d_m, int n, int k) {
    int j = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    float pivot_val = d_m[k * (n + 1) + k];
    if (j <= n) {
        d_m[k * (n + 1) + j] /= pivot_val;
    }
}

/**
 * @brief CUDA 核函数：使用共享内存瓦片和循环展开执行消元
 * @tparam BLOCK_DIM 线程块的维度
 */
template <int BLOCK_DIM>
__global__ void elimination_kernel_unrolled(float *d_m, int n, int k) {
    // 定义共享内存瓦片
    __shared__ float factor_tile[BLOCK_DIM];

    // 计算全局行索引
    int row = k + 1 + blockIdx.y * BLOCK_DIM + threadIdx.y;

    // 1. 加载消元因子到共享内存
    if (row < n && threadIdx.x == 0) { // 每行的第一个线程加载因子
        factor_tile[threadIdx.y] = d_m[row * (n + 1) + k];
    }
    __syncthreads();

    // 2. 循环展开计算
    if (row < n) {
        float factor = factor_tile[threadIdx.y];
        int col_start = k + 1 + blockIdx.x * BLOCK_DIM * UNROLL_FACTOR + threadIdx.x;

        // 使用 #pragma unroll 指示编译器展开此循环
        #pragma unroll
        for (int i = 0; i < UNROLL_FACTOR; ++i) {
            int col = col_start + i * BLOCK_DIM;
            if (col <= n) {
                float pivot_val = d_m[k * (n + 1) + col]; // 主元行仍然从全局内存读取
                d_m[row * (n + 1) + col] -= factor * pivot_val;
            }
        }
    }
    
    __syncthreads();

    // 3. 内核融合：置零
    if (row < n && threadIdx.x == 0) {
        d_m[row * (n + 1) + k] = 0.0f;
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

// 启动器函数
void launch_elimination_kernel(float* d_m, int n, int k, int block_dim) {
    dim3 block(block_dim, block_dim);
    // 因为有循环展开，x维度的网格可以相应减小
    dim3 grid(
        ( (n + 1 - (k + 1)) + (block.x * UNROLL_FACTOR) - 1) / (block.x * UNROLL_FACTOR),
        (n - (k + 1) + block.y - 1) / block.y
    );

    switch (block_dim) {
        case 8:
            elimination_kernel_unrolled<8><<<grid, block>>>(d_m, n, k);
            break;
        case 16:
            elimination_kernel_unrolled<16><<<grid, block>>>(d_m, n, k);
            break;
        case 32:
            elimination_kernel_unrolled<32><<<grid, block>>>(d_m, n, k);
            break;
        default:
            std::cerr << "Unsupported block dimension: " << block_dim << std::endl;
            exit(1);
    }
}


int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size_N> <block_dim>" << std::endl;
        std::cerr << "Supported block_dim: 8, 16, 32" << std::endl;
        return 1;
    }

    const int N = std::atoi(argv[1]);
    const int block_dim = std::atoi(argv[2]);

    if (N <= 0 || (block_dim != 8 && block_dim != 16 && block_dim != 32)) {
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

        // --- 消元步骤 (使用共享内存和循环展开) ---
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