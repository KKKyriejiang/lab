#!/bin/bash

# 清理旧文件
rm -f gaussian_elimination results.dat time_vs_size.png speedup_vs_size.png

# 编译 CUDA 代码
echo "Compiling CUDA code..."
nvcc gaussian_elimination.cu -o gaussian_elimination
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

# 定义要测试的矩阵规模
SIZES=(256 512 1024 2048)
DATA_FILE="results.dat"

# 写入数据文件的表头
echo "# Size CPU_Time(ms) GPU_Time(ms) Speedup" > $DATA_FILE

# 循环执行并收集数据
for N in "${SIZES[@]}"; do
    echo "-------------------------------------"
    echo "Running for matrix size N = $N"
    
    # 运行程序并捕获输出
    output=$(./gaussian_elimination $N)
    
    # 从输出中提取 CPU 和 GPU 时间
    cpu_time=$(echo "$output" | grep "CPU_TIME" | cut -d':' -f2)
    gpu_time=$(echo "$output" | grep "GPU_TIME" | cut -d':' -f2)
    
    if [ -z "$cpu_time" ] || [ -z "$gpu_time" ]; then
        echo "Failed to get timings for N = $N. Output was:"
        echo "$output"
        continue
    fi

    # 计算加速比 (使用 bc)
    speedup=$(echo "scale=4; $cpu_time / $gpu_time" | bc)
    
    echo "CPU Time: ${cpu_time} ms"
    echo "GPU Time: ${gpu_time} ms"
    echo "Speedup: ${speedup}x"
    
    # 将结果追加到数据文件
    echo "$N $cpu_time $gpu_time $speedup" >> $DATA_FILE
done

echo "-------------------------------------"
echo "Generating plots..."

# 使用 gnuplot 绘制第一张图：时间 vs 规模
gnuplot <<- EOF
    set terminal pngcairo size 800,600 enhanced font 'Verdana,10'
    set output 'time_vs_size.png'
    set title "Execution Time vs. Matrix Size"
    set xlabel "Matrix Size (N)"
    set ylabel "Time (milliseconds)"
    set style data histograms
    set style histogram cluster gap 1
    set style fill solid border -1
    set boxwidth 0.9
    set key top left
    set yrange [0:*]
    set logscale y 10
    set format y "10^{%L}"

    plot '$DATA_FILE' using 2:xtic(1) title "CPU Time", \
         '' using 3:xtic(1) title "GPU Time"
EOF

# 使用 gnuplot 绘制第二张图：加速比 vs 规模
gnuplot <<- EOF
    set terminal pngcairo size 800,600 enhanced font 'Verdana,10'
    set output 'speedup_vs_size.png'
    set title "Speedup vs. Matrix Size"
    set xlabel "Matrix Size (N)"
    set ylabel "Speedup (CPU Time / GPU Time)"
    set style data histograms
    set style fill solid border -1
    set boxwidth 0.9
    set yrange [0:*]
    set key off

    plot '$DATA_FILE' using 4:xtic(1) title "Speedup"
EOF

echo "Plots 'time_vs_size.png' and 'speedup_vs_size.png' have been generated."
echo "Script finished."