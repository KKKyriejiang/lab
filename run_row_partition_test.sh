#!/bin/bash

# 清理旧文件
rm -f gaussian_row_partition results_row_partition.csv

# 编译使用行划分策略的 CUDA 代码
echo "Compiling gaussian_row_partition.cu..."
nvcc gaussian_row_partition.cu -o gaussian_row_partition
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

# 定义要测试的矩阵规模和块维度
SIZES=(1024 2048)
BLOCK_DIMS=(32 64 128 256 512) # 根据您的代码，支持 128, 256, 512
DATA_FILE="results_row_partition.csv"

# 写入 CSV 文件的表头
echo "Size,BlockDim,CPU_Time(ms),GPU_Time(ms),Speedup" > $DATA_FILE

# 循环执行并收集数据
for N in "${SIZES[@]}"; do
    for B_DIM in "${BLOCK_DIMS[@]}"; do
        echo "-------------------------------------"
        echo "Running Row Partition version for N = $N with BlockDim = $B_DIM"
        
        # 运行程序并捕获输出
        output=$(./gaussian_row_partition $N $B_DIM)
        
        # 从输出中提取 CPU 和 GPU 时间
        cpu_time=$(echo "$output" | grep "CPU_TIME" | cut -d':' -f2)
        gpu_time=$(echo "$output" | grep "GPU_TIME" | cut -d':' -f2)
        
        if [ -z "$cpu_time" ] || [ -z "$gpu_time" ]; then
            echo "Failed to get timings for N=$N, BlockDim=$B_DIM. Output was:"
            echo "$output"
            continue
        fi

        # 计算加速比 (使用 bc)
        speedup=$(echo "scale=4; $cpu_time / $gpu_time" | bc)
        
        echo "CPU Time: ${cpu_time} ms"
        echo "GPU Time: ${gpu_time} ms"
        echo "Speedup: ${speedup}x"
        
        # 将结果追加到 CSV 文件
        echo "$N,$B_DIM,$cpu_time,$gpu_time,$speedup" >> $DATA_FILE
    done
done

echo "-------------------------------------"
echo "Data collection for row partition tests finished."
echo "Results have been saved to ${DATA_FILE}."
echo "Please download this file and use a spreadsheet program to create the plots."