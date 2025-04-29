#!/bin/bash
# Script to run comprehensive experiments for parallel intersection benchmarks
# Enhanced with cache miss and CPU cycle measurements

# Configuration
BUILD_DIR="./build"
DATA_DIR="./data/benchmark"
RESULTS_DIR="./results/$(date +%Y%m%d_%H%M%S)"
ALGORITHMS=("op" "range" "critical" "binary" "adaptive" "leapfrog" "worksteal")
THREAD_COUNTS=(1 2 4 8 16 32 64 128)
MAX_HARDWARE_THREADS=$(nproc)
REPETITIONS=3

# Create directories
mkdir -p $BUILD_DIR
mkdir -p $DATA_DIR
mkdir -p $RESULTS_DIR

echo "Maximum hardware threads available: $MAX_HARDWARE_THREADS"
echo "Will test with thread counts: ${THREAD_COUNTS[@]}"

# Build the project
echo "Building project..."
cd $BUILD_DIR
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$MAX_HARDWARE_THREADS
cd ..

# Generate test data if it doesn't exist
echo "Checking for test data..."
if [ ! "$(ls -A $DATA_DIR)" ]; then
    echo "Generating test data..."
    3 scripts/generate_data_benchmark.py --output-dir $DATA_DIR
else
    echo "Found existing test data in $DATA_DIR"
fi

# Function to run benchmark with specified parameters
run_benchmark() {
    local algorithm=$1
    local dataset=$2
    local threads=$3
    local repetition=$4
    
    echo "Running $algorithm on $dataset with $threads threads (rep $repetition)..."
    
    # Create output directory
    local output_dir="$RESULTS_DIR/$algorithm/$dataset/threads_$threads"
    mkdir -p "$output_dir"
    
    # Set OMP_NUM_THREADS environment variable to ensure OpenMP uses exactly this many threads
    export OMP_NUM_THREADS=$threads
    
    # Create perf output file
    local perf_output="$output_dir/rep_${repetition}_perf.log"
    
    # Run benchmark with perf stat and capture output
    perf stat -e cycles,instructions,cache-misses,cache-references,context-switches,cpu-migrations,stalled-cycles-frontend,stalled-cycles-backend,branch-misses -o "$perf_output" \
    $BUILD_DIR/intersection_benchmark "$DATA_DIR/$dataset" "$algorithm" "$threads" > "$output_dir/rep_$repetition.log" 2>&1
    
    # Extract execution time and result size
    local exec_time=$(grep "Computation time:" "$output_dir/rep_$repetition.log" | awk '{print $3}')
    local result_size=$(grep "Result size:" "$output_dir/rep_$repetition.log" | awk '{print $3}')
    
    # Extract metrics from perf output
    local cycles=$(grep "cycles" "$perf_output" | head -1 | awk '{print $1}' | sed 's/,//g')
    local instructions=$(grep "instructions" "$perf_output" | head -1 | awk '{print $1}' | sed 's/,//g')
    
    # Calculate IPC (Instructions Per Cycle)
    local ipc="N/A"
    if [ ! -z "$cycles" ] && [ ! -z "$instructions" ] && [ "$cycles" != "0" ]; then
        ipc=$(echo "scale=2; $instructions / $cycles" | bc)
    fi
    
    local cache_misses=$(grep "cache-misses" "$perf_output" | head -1 | awk '{print $1}' | sed 's/,//g')
    local cache_miss_rate=$(grep "cache-misses" "$perf_output" | head -1 | awk '{print $4}')
    local context_switches=$(grep "context-switches" "$perf_output" | head -1 | awk '{print $1}' | sed 's/,//g')
    local cpu_migrations=$(grep "cpu-migrations" "$perf_output" | head -1 | awk '{print $1}' | sed 's/,//g')
    local stalled_frontend=$(grep "stalled-cycles-frontend" "$perf_output" | head -1 | awk '{print $1}' | sed 's/,//g')
    local stalled_backend=$(grep "stalled-cycles-backend" "$perf_output" | head -1 | awk '{print $1}' | sed 's/,//g')
    local branch_misses=$(grep "branch-misses" "$perf_output" | head -1 | awk '{print $1}' | sed 's/,//g')
    
    # Save to summary file
    echo "$algorithm,$dataset,$threads,$repetition,$exec_time,$result_size,$cycles,$instructions,$ipc,$cache_misses,$cache_miss_rate,$context_switches,$cpu_migrations,$stalled_frontend,$stalled_backend,$branch_misses" >> "$RESULTS_DIR/summary.csv"
    
    # Print a brief summary
    echo "  Time: $exec_time seconds, Result size: $result_size"
    echo "  Cache misses: $cache_misses ($cache_miss_rate)"
    echo "  Cycles: $cycles, Instructions: $instructions, IPC: $ipc"
}

# Create summary CSV header
echo "algorithm,dataset,threads,repetition,exec_time,result_size,cycles,instructions,ipc,cache_misses,cache_miss_rate,context_switches,cpu_migrations,stalled_cycles_frontend,stalled_cycles_backend,branch_misses" > "$RESULTS_DIR/summary.csv"

# Get list of datasets
datasets=$(ls $DATA_DIR/*.txt | xargs -n 1 basename)

# Run experiments
for algorithm in "${ALGORITHMS[@]}"; do
    for dataset in $datasets; do
        for threads in "${THREAD_COUNTS[@]}"; do
            # Skip thread counts that exceed hardware capability with a warning
            if [ "$threads" -gt "$MAX_HARDWARE_THREADS" ]; then
                echo "WARNING: Testing with $threads threads on a system with $MAX_HARDWARE_THREADS hardware threads."
                echo "         This will result in oversubscription and may impact performance measurements."
            fi
            
            for rep in $(seq 1 $REPETITIONS); do
                run_benchmark "$algorithm" "$dataset" $threads $rep
            done
        done
    done
done

# Generate analysis
echo "Generating analysis..."
python3 scripts/analyze_results.py --results-dir "$RESULTS_DIR"

# Generate a simple cache miss summary
echo "Creating cache miss and CPU cycle summary..."
echo "algorithm,dataset,threads,avg_exec_time,avg_cycles,avg_instructions,avg_ipc,avg_cache_misses,avg_cache_miss_rate,avg_stalled_frontend,avg_stalled_backend" > "$RESULTS_DIR/cpu_metrics_summary.csv"

# Generate a simple summary of CPU metrics
awk -F, 'BEGIN { OFS="," }
NR > 1 {
    key = $1 "," $2 "," $3;  # algorithm,dataset,threads
    count[key]++;
    time_sum[key] += $5;
    cycles_sum[key] += $7;
    instructions_sum[key] += $8;
    ipc_sum[key] += $9;
    cache_misses_sum[key] += $10;
    cache_miss_rate_sum[key] += $11;
    stalled_frontend_sum[key] += $14;
    stalled_backend_sum[key] += $15;
}
END {
    for (key in count) {
        avg_time = time_sum[key] / count[key];
        avg_cycles = cycles_sum[key] / count[key];
        avg_instructions = instructions_sum[key] / count[key];
        avg_ipc = ipc_sum[key] / count[key];
        avg_cache_misses = cache_misses_sum[key] / count[key];
        avg_cache_miss_rate = cache_miss_rate_sum[key] / count[key];
        avg_stalled_frontend = stalled_frontend_sum[key] / count[key];
        avg_stalled_backend = stalled_backend_sum[key] / count[key];
        
        print key, avg_time, avg_cycles, avg_instructions, avg_ipc, avg_cache_misses, avg_cache_miss_rate, avg_stalled_frontend, avg_stalled_backend;
    }
}' "$RESULTS_DIR/summary.csv" >> "$RESULTS_DIR/cpu_metrics_summary.csv"

echo "Experiments complete. Results saved to $RESULTS_DIR"
echo "CPU metrics summary saved to $RESULTS_DIR/cpu_metrics_summary.csv"