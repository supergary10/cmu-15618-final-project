import random
import os
import math
import argparse
import numpy as np
from tqdm import tqdm

def generate_zipfian_distribution(size, alpha=1.0, value_range=(0, 1000000)):
    """Generate values following a Zipfian distribution"""
    # Get range values
    min_val, max_val = value_range
    range_size = max_val - min_val + 1
    
    # Generate Zipfian values
    x = np.random.zipf(alpha, size=size * 2)  # Generate extra to ensure we have enough unique values
    x = [val % range_size + min_val for val in x]  # Map to our range
    
    # Get unique values and return requested size
    unique_vals = list(set(x))
    if len(unique_vals) < size:
        print(f"Warning: Could only generate {len(unique_vals)} unique values using Zipfian distribution")
        return sorted(unique_vals)
    else:
        return sorted(random.sample(unique_vals, size))

def generate_data_benchmark(output_dir, patterns):
    """Generate multiple data files for comprehensive benchmarking"""
    os.makedirs(output_dir, exist_ok=True)
    
    for pattern in patterns:
        name = pattern["name"]
        filename = os.path.join(output_dir, f"{name}.txt")
        
        print(f"Generating dataset: {name}")
        
        # Extract parameters
        num_vectors = pattern.get("num_vectors", 3)
        vector_sizes = pattern.get("vector_sizes", [100, 100, 100])
        overlap_ratio = pattern.get("overlap_ratio", 0.5)
        distribution = pattern.get("distribution", "uniform")
        value_range = pattern.get("value_range", (0, 1000000))
        zipfian_alpha = pattern.get("zipfian_alpha", 1.5)
        
        # Ensure vector_sizes matches num_vectors
        if len(vector_sizes) < num_vectors:
            # Extend with the last value
            vector_sizes.extend([vector_sizes[-1]] * (num_vectors - len(vector_sizes)))
        
        # Calculate overlap size
        min_size = min(vector_sizes)
        overlap_size = int(min_size * overlap_ratio)
        
        # Generate overlap values based on distribution
        if distribution == "uniform":
            overlap_set = set()
            while len(overlap_set) < overlap_size:
                val = random.randint(value_range[0], value_range[1])
                overlap_set.add(val)
        elif distribution == "zipfian":
            overlap_set = set(generate_zipfian_distribution(
                overlap_size, 
                alpha=zipfian_alpha,
                value_range=value_range
            ))
        elif distribution == "clustered":
            # Generate values clustered in a smaller range
            cluster_center = random.randint(value_range[0], value_range[1])
            cluster_radius = int((value_range[1] - value_range[0]) * 0.1)  # 10% of total range
            
            cluster_min = max(value_range[0], cluster_center - cluster_radius)
            cluster_max = min(value_range[1], cluster_center + cluster_radius)
            
            overlap_set = set()
            while len(overlap_set) < overlap_size:
                val = random.randint(cluster_min, cluster_max)
                overlap_set.add(val)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
            
        overlap_values = sorted(list(overlap_set))
        
        # Write vectors to file
        with open(filename, 'w') as f:
            for i in range(num_vectors):
                target_size = vector_sizes[i]
                non_overlap_size = target_size - len(overlap_values)
                
                if non_overlap_size < 0:
                    # If overlap is larger than target, sample from overlap
                    vector_values = random.sample(overlap_values, target_size)
                else:
                    # Generate unique values for this vector
                    unique_vals = set()
                    attempts = 0
                    max_attempts = non_overlap_size * 10
                    
                    while len(unique_vals) < non_overlap_size and attempts < max_attempts:
                        if distribution == "zipfian":
                            # Generate values with zipfian distribution
                            candidates = generate_zipfian_distribution(
                                min(1000, non_overlap_size - len(unique_vals)), 
                                alpha=zipfian_alpha,
                                value_range=value_range
                            )
                            for val in candidates:
                                if val not in overlap_set and val not in unique_vals:
                                    unique_vals.add(val)
                                if len(unique_vals) >= non_overlap_size:
                                    break
                        else:
                            # Simple uniform generation
                            val = random.randint(value_range[0], value_range[1])
                            if val not in overlap_set and val not in unique_vals:
                                unique_vals.add(val)
                        attempts += 1
                    
                    # Combine overlap and unique values
                    vector_values = sorted(overlap_values + list(unique_vals))
                
                # Write to file
                f.write(" ".join(map(str, vector_values)) + "\n")
                
        print(f"  Created {filename} with {num_vectors} vectors")

def main():
    parser = argparse.ArgumentParser(description="Generate benchmark datasets for parallel WCOJ")
    parser.add_argument("--output-dir", type=str, default="data/benchmark", help="Directory to save benchmark files")
    args = parser.parse_args()
    
    # Define benchmark patterns
    benchmark_patterns = [
        # 1. Small dataset with even sizes and high overlap (baseline)
        {
            "name": "small_even_high_overlap",
            "num_vectors": 3,
            "vector_sizes": [100, 100, 100],
            "overlap_ratio": 0.8,
            "distribution": "uniform"
        },
        
        # 2. Small dataset with even sizes and low overlap
        {
            "name": "small_even_low_overlap",
            "num_vectors": 3,
            "vector_sizes": [100, 100, 100],
            "overlap_ratio": 0.2,
            "distribution": "uniform"
        },
        
        # 3. Medium dataset with even sizes
        {
            "name": "medium_even",
            "num_vectors": 3,
            "vector_sizes": [10000, 10000, 10000],
            "overlap_ratio": 0.5,
            "distribution": "uniform"
        },
        
        # 4. Large dataset with even sizes
        {
            "name": "large_even",
            "num_vectors": 3,
            "vector_sizes": [100000, 100000, 100000],
            "overlap_ratio": 0.5,
            "distribution": "uniform"
        },
        
        # 5. Skewed sizes (one small, others large)
        {
            "name": "size_skewed_small_first",
            "num_vectors": 3,
            "vector_sizes": [1000, 50000, 50000],
            "overlap_ratio": 0.5,
            "distribution": "uniform"
        },
        
        # 6. Skewed sizes (one large, others small)
        {
            "name": "size_skewed_large_first",
            "num_vectors": 3,
            "vector_sizes": [50000, 1000, 1000],
            "overlap_ratio": 0.5,
            "distribution": "uniform"
        },
        
        # 7. Zipfian distribution (power-law)
        {
            "name": "zipfian_distribution",
            "num_vectors": 3,
            "vector_sizes": [10000, 10000, 10000],
            "overlap_ratio": 0.5,
            "distribution": "zipfian",
            "zipfian_alpha": 1.5
        },
        
        # 8. Clustered values
        {
            "name": "clustered_values",
            "num_vectors": 3,
            "vector_sizes": [10000, 10000, 10000],
            "overlap_ratio": 0.5,
            "distribution": "clustered"
        },
        
        # 9. More vectors (stress test)
        {
            "name": "many_vectors",
            "num_vectors": 8,
            "vector_sizes": [5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],
            "overlap_ratio": 0.4,
            "distribution": "uniform"
        },
        
        # 10. Extreme case - very large vectors
        {
            "name": "extreme_large",
            "num_vectors": 3,
            "vector_sizes": [500000, 500000, 500000],
            "overlap_ratio": 0.3,
            "distribution": "uniform"
        }
    ]
    
    generate_data_benchmark(args.output_dir, benchmark_patterns)
    print(f"Benchmark data generation complete. Files saved to: {args.output_dir}")

if __name__ == "__main__":
    main()