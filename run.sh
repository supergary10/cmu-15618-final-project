#!/bin/bash
# Script to run comprehensive experiments for parallel intersection benchmarks

# Configuration
BUILD_DIR="./build"
DATA_DIR="./data/benchmark"
RESULTS_DIR="./results/$(date +%Y%m%d_%H%M%S)"
ALGORITHMS=("op" "range" "critical" "binary" "adaptive" "leapfrog" "worksteal")
MAX_THREADS=$(nproc)
REPETITIONS=3

# Create directories
mkdir -p $BUILD_DIR
mkdir -p $DATA_DIR
mkdir -p $RESULTS_DIR

# Build the project
echo "Building project..."
cd $BUILD_DIR
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ..

# Generate test data if it doesn't exist
echo "Checking for test data..."
if [ ! "$(ls -A $DATA_DIR)" ]; then
    echo "Generating test data..."
    python3 scripts/generate_data_benchmark.py --output-dir $DATA_DIR
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
    
    # Run benchmark and capture output
    $BUILD_DIR/intersection_benchmark "$DATA_DIR/$dataset" "$algorithm" "$threads" > "$output_dir/rep_$repetition.log" 2>&1
    
    # Extract execution time
    local exec_time=$(grep "Computation time:" "$output_dir/rep_$repetition.log" | awk '{print $3}')
    local result_size=$(grep "Result size:" "$output_dir/rep_$repetition.log" | awk '{print $3}')
    
    # Save to summary file
    echo "$algorithm,$dataset,$threads,$repetition,$exec_time,$result_size" >> "$RESULTS_DIR/summary.csv"
}

# Create summary CSV header
echo "algorithm,dataset,threads,repetition,exec_time,result_size" > "$RESULTS_DIR/summary.csv"

# Get list of datasets
datasets=$(ls $DATA_DIR/*.txt | xargs -n 1 basename)

# Run experiments
for algorithm in "${ALGORITHMS[@]}"; do
    for dataset in $datasets; do
        # Always run sequential version for baseline
        for rep in $(seq 1 $REPETITIONS); do
            run_benchmark "$algorithm" "$dataset" 1 $rep
        done
        
        # Run with different thread counts
        for threads in 2 4 $(seq 8 8 $MAX_THREADS); do
            for rep in $(seq 1 $REPETITIONS); do
                run_benchmark "$algorithm" "$dataset" $threads $rep
            done
        done
    done
done

# Generate analysis
echo "Generating analysis..."
python3 scripts/analyze_results.py --results-dir "$RESULTS_DIR"

echo "Experiments complete. Results saved to $RESULTS_DIR"