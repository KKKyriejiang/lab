#!/bin/bash

# 清理旧文件
rm -f gaussian_naive results_naive.csv

# 编译平凡算法的 CUDA 代码
echo "Compiling gaussian_naive.cu..."
nvcc gaussian_naive.cu -o gaussian_naive
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

# 定义要测试的矩阵规模
SIZES=(512 1024 2048 4096)
DATA_FILE="results_naive.csv"

# 写入 CSV 文件的表头
# 注意：此版本块大小硬编码，因此表头不包含 BlockDim
echo "Size,CPU_Time(ms),GPU_Time(ms),Speedup" > $DATA_FILE

# 循环执行并收集数据
for N in "${SIZES[@]}"; do
    echo "-------------------------------------"
    echo "Running Naive version for N = $N"
    
    # 运行程序并捕获输出
    # 此程序只接受一个参数：矩阵规模 N
    output=$(./gaussian_naive $N)
    
    # 从输出中提取 CPU 和 GPU 时间
    cpu_time=$(echo "$output" | grep "CPU_TIME" | cut -d':' -f2)
    gpu_time=$(echo "$output" | grep "GPU_TIME" | cut -d':' -f2)
    
    if [ -z "$cpu_time" ] || [ -z "$gpu_time" ]; then
        echo "Failed to get timings for N=$N. Output was:"
        echo "$output"
        continue
    fi

    # 计算加速比 (使用 bc)
    speedup=$(echo "scale=4; $cpu_time / $gpu_time" | bc)
    
    echo "CPU Time: ${cpu_time} ms"
    echo "GPU Time: ${gpu_time} ms"
    echo "Speedup: ${speedup}x"
    
    # 将结果追加到 CSV 文件
    echo "$N,$cpu_time,$gpu_time,$speedup" >> $DATA_FILE
done

echo "-------------------------------------"
echo "Data collection for naive tests finished."
echo "Results have been saved to ${DATA_FILE}."