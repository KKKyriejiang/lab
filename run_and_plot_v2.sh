
// filepath: /home/s2313255/run_block_size_tests.sh
#!/bin/bash

# 清理旧文件
rm -f gaussian_block_test results_block_test.csv

# 编译新的 CUDA 代码
echo "Compiling gaussian_block_test.cu..."
nvcc gaussian_block_test.cu -o gaussian_block_test
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

# 定义要测试的矩阵规模和块维度
SIZES=(1024 2048)
BLOCK_DIMS=(8 16 32) # 测试 8x8, 16x16, 32x32 的块
DATA_FILE="results_block_test.csv"

# 写入 CSV 文件的表头
echo "Size,BlockDim,CPU_Time(ms),GPU_Time(ms),Speedup" > $DATA_FILE

# 循环执行并收集数据
for N in "${SIZES[@]}"; do
    for B_DIM in "${BLOCK_DIMS[@]}"; do
        echo "-------------------------------------"
        echo "Running for matrix size N = $N with block dimension = $B_DIM"
        
        # 运行程序并捕获输出
        output=$(./gaussian_block_test $N $B_DIM)
        
        # 从输出中提取 CPU 和 GPU 时间
        cpu_time=$(echo "$output" | grep "CPU_TIME" | cut -d':' -f2)
        gpu_time=$(echo "$output" | grep "GPU_TIME" | cut -d':' -f2)
        
        if [ -z "$cpu_time" ] || [ -z "$gpu_time" ]; then
            echo "Failed to get timings for N=$N, B_DIM=$B_DIM. Output was:"
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
echo "Data collection for block size tests finished."
echo "Results have been saved to results_block_test.csv."
echo "Please download this file and use a spreadsheet program to create the plots."
