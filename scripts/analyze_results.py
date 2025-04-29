import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from collections import defaultdict
import re

def load_results(results_dir):
    """Load results from summary CSV file"""
    summary_path = os.path.join(results_dir, "summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    
    df = pd.read_csv(summary_path)
    return df

def extract_dataset_features(dataset_name):
    """Extract dataset features from dataset names"""
    # Parse dataset characteristics from filename
    features = {}
    
    # Extract size category
    if "small_" in dataset_name:
        features["size"] = "small"
    elif "medium_" in dataset_name:
        features["size"] = "medium"
    elif "large_" in dataset_name:
        features["size"] = "large"
    elif "extreme_" in dataset_name:
        features["size"] = "extreme"
    else:
        features["size"] = "unknown"
    
    # Extract distribution
    if "skewed_" in dataset_name:
        features["distribution"] = "skewed"
    elif "zipfian_" in dataset_name:
        features["distribution"] = "zipfian"
    elif "clustered_" in dataset_name:
        features["distribution"] = "clustered"
    elif "uniform_" in dataset_name:
        features["distribution"] = "uniform"
    else:
        features["distribution"] = "unknown"
    
    # Extract overlap
    if "high_overlap" in dataset_name:
        features["overlap"] = "high"
    elif "low_overlap" in dataset_name:
        features["overlap"] = "low"
    else:
        features["overlap"] = "medium"
    
    # Extract number of vectors
    if "many_vectors" in dataset_name:
        features["vectors"] = "many"
    else:
        features["vectors"] = "few"
    
    return features

def calculate_metrics(df):
    """Calculate derived metrics like speedup and efficiency"""
    # Group by algorithm and dataset to calculate speedup
    metrics = []
    
    for (alg, dataset), group in df.groupby(['algorithm', 'dataset']):
        # Get average sequential time (threads=1)
        seq_time_group = group[group['threads'] == 1]
        if seq_time_group.empty:
            print(f"Warning: No sequential (thread=1) timing found for {alg} on {dataset}. Skipping speedup calculation.")
            continue
            
        seq_time = seq_time_group['exec_time'].mean()
        
        for threads, thread_group in group.groupby('threads'):
            avg_time = thread_group['exec_time'].mean()
            min_time = thread_group['exec_time'].min()
            max_time = thread_group['exec_time'].max()
            std_time = thread_group['exec_time'].std() if len(thread_group) > 1 else 0
            
            # Calculate metrics
            speedup = seq_time / avg_time if avg_time > 0 else 0
            efficiency = speedup / threads
            
            # Extract dataset features
            features = extract_dataset_features(dataset)
            
            metrics.append({
                'algorithm': alg,
                'dataset': dataset,
                'threads': threads,
                'avg_time': avg_time,
                'min_time': min_time,
                'max_time': max_time,
                'std_time': std_time,
                'speedup': speedup,
                'efficiency': efficiency,
                'size': features['size'],
                'distribution': features['distribution'],
                'overlap': features['overlap'],
                'vectors': features['vectors']
            })
    
    return pd.DataFrame(metrics)

def plot_speedup_by_algorithm(metrics_df, output_dir):
    """Plot speedup comparison across algorithms using log scale for thread counts"""
    plt.figure(figsize=(14, 10))
    
    # Use log scale for x-axis due to wide range of thread counts
    plt.xscale('log', base=2)
    
    # Group by algorithm
    for algorithm, group in metrics_df.groupby('algorithm'):
        # Calculate average speedup across all datasets
        avg_speedup = group.groupby('threads')['speedup'].mean().reset_index()
        plt.plot(avg_speedup['threads'], avg_speedup['speedup'], marker='o', label=algorithm)
    
    # Add ideal speedup line
    thread_values = sorted(metrics_df['threads'].unique())
    if thread_values:
        max_threads = max(thread_values)
        plt.plot(thread_values, thread_values, 'k--', alpha=0.5, label='Ideal Speedup')
    
    plt.title('Average Speedup vs Thread Count by Algorithm (Log Scale)')
    plt.xlabel('Number of Threads (log scale)')
    plt.ylabel('Speedup')
    plt.legend(title='Algorithm')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ensure x-ticks show all thread count values
    plt.xticks(thread_values, thread_values)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speedup_by_algorithm_log.png'), dpi=300)
    plt.close()
    
    # Also create a version with linear scale
    plt.figure(figsize=(14, 10))
    for algorithm, group in metrics_df.groupby('algorithm'):
        avg_speedup = group.groupby('threads')['speedup'].mean().reset_index()
        plt.plot(avg_speedup['threads'], avg_speedup['speedup'], marker='o', label=algorithm)
    
    if thread_values:
        max_threads = max(thread_values)
        plt.plot(thread_values, thread_values, 'k--', alpha=0.5, label='Ideal Speedup')
    
    plt.title('Average Speedup vs Thread Count by Algorithm (Linear Scale)')
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup')
    plt.legend(title='Algorithm')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(thread_values)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speedup_by_algorithm_linear.png'), dpi=300)
    plt.close()

def plot_efficiency_by_algorithm(metrics_df, output_dir):
    """Plot parallel efficiency comparison across algorithms"""
    plt.figure(figsize=(14, 10))
    
    # Use log scale for x-axis
    plt.xscale('log', base=2)
    
    # Group by algorithm
    for algorithm, group in metrics_df.groupby('algorithm'):
        # Calculate average efficiency across all datasets
        avg_efficiency = group.groupby('threads')['efficiency'].mean().reset_index()
        plt.plot(avg_efficiency['threads'], avg_efficiency['efficiency'], marker='o', label=algorithm)
    
    plt.title('Parallel Efficiency vs Thread Count by Algorithm (Log Scale)')
    plt.xlabel('Number of Threads (log scale)')
    plt.ylabel('Efficiency (Speedup/Threads)')
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Ideal Efficiency')
    plt.legend(title='Algorithm')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ensure x-ticks show all thread count values
    thread_values = sorted(metrics_df['threads'].unique())
    plt.xticks(thread_values, thread_values)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'efficiency_by_algorithm_log.png'), dpi=300)
    plt.close()

def plot_execution_time_comparison(metrics_df, output_dir):
    """Plot execution time comparison for different algorithms at various thread counts"""
    # Create separate plots for different thread counts
    thread_values = sorted(metrics_df['threads'].unique())
    
    for thread_count in [1, 8, 32, max(thread_values)]:
        if thread_count not in thread_values:
            continue
            
        # Filter for this thread count
        thread_df = metrics_df[metrics_df['threads'] == thread_count]
        
        plt.figure(figsize=(14, 8))
        
        # Create a grouped bar chart by algorithm and dataset size
        sns.barplot(data=thread_df, x='algorithm', y='avg_time', hue='size')
        
        plt.title(f'Execution Time by Algorithm and Dataset Size (Threads: {thread_count})')
        plt.xlabel('Algorithm')
        plt.ylabel('Execution Time (seconds)')
        plt.yscale('log')  # Use log scale for better visualization
        plt.legend(title='Dataset Size')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f'execution_time_threads_{thread_count}.png'), dpi=300)
        plt.close()

def plot_speedup_scaling(metrics_df, output_dir):
    """Plot speedup scaling behavior for different dataset sizes"""
    # Create separate plots for each dataset size
    for size in metrics_df['size'].unique():
        size_df = metrics_df[metrics_df['size'] == size]
        
        plt.figure(figsize=(14, 10))
        plt.xscale('log', base=2)
        
        # Group by algorithm
        for algorithm, group in size_df.groupby('algorithm'):
            # Calculate average speedup across datasets of this size
            avg_speedup = group.groupby('threads')['speedup'].mean().reset_index()
            plt.plot(avg_speedup['threads'], avg_speedup['speedup'], marker='o', label=algorithm)
        
        # Add ideal speedup line
        thread_values = sorted(size_df['threads'].unique())
        if thread_values:
            max_threads = max(thread_values)
            plt.plot(thread_values, thread_values, 'k--', alpha=0.5, label='Ideal Speedup')
        
        plt.title(f'Speedup Scaling for {size.capitalize()} Datasets (Log Scale)')
        plt.xlabel('Number of Threads (log scale)')
        plt.ylabel('Speedup')
        plt.legend(title='Algorithm')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(thread_values, thread_values)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f'speedup_scaling_{size}_log.png'), dpi=300)
        plt.close()

def plot_heatmap_algorithm_dataset(metrics_df, output_dir):
    """Create a heatmap of algorithm performance across datasets at specific thread counts"""
    for threads in [8, 32, max(metrics_df['threads'].unique())]:
        # Filter for this thread count
        filtered_df = metrics_df[metrics_df['threads'] == threads]
        
        if filtered_df.empty:
            continue
        
        # Reshape for heatmap (algorithm vs. dataset)
        pivot_speedup = filtered_df.pivot_table(
            index='algorithm', 
            columns='dataset', 
            values='speedup'
        )
        
        plt.figure(figsize=(16, 10))
        sns.heatmap(pivot_speedup, annot=True, cmap='viridis', fmt='.2f')
        
        plt.title(f'Speedup Heatmap: Algorithm vs Dataset (Threads: {threads})')
        plt.xlabel('Dataset')
        plt.ylabel('Algorithm')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f'algorithm_dataset_heatmap_threads_{threads}.png'), dpi=300)
        plt.close()

def create_recommendation_table(metrics_df, output_dir):
    """Create a recommendation table for algorithm selection at different thread counts"""
    # Get best algorithm for different thread counts and dataset types
    thread_counts = sorted(metrics_df['threads'].unique())
    recommendations = []
    
    # Group by thread count, dataset characteristics and find best algorithm
    for threads in thread_counts:
        thread_df = metrics_df[metrics_df['threads'] == threads]
        
        for (size, dist, overlap, vectors), group in thread_df.groupby(['size', 'distribution', 'overlap', 'vectors']):
            if len(group) == 0:
                continue
                
            # Find best algorithm by speedup
            best_alg_idx = group['speedup'].idxmax()
            best_alg = group.loc[best_alg_idx]
            
            recommendation = {
                'threads': threads,
                'size': size,
                'distribution': dist,
                'overlap': overlap,
                'vectors': vectors,
                'best_algorithm': best_alg['algorithm'],
                'speedup': best_alg['speedup'],
                'avg_time': best_alg['avg_time']
            }
            
            recommendations.append(recommendation)
    
    # Convert to DataFrame and save
    rec_df = pd.DataFrame(recommendations)
    rec_df.to_csv(os.path.join(output_dir, 'algorithm_recommendations.csv'), index=False)
    
    return rec_df

def plot_thread_scaling_limit(metrics_df, output_dir):
    """Plot to identify the scaling limit where adding more threads stops helping"""
    # Group by algorithm and dataset
    for (alg, dataset), group in metrics_df.groupby(['algorithm', 'dataset']):
        # Skip if not enough data points
        if len(group) < 3:
            continue
            
        plt.figure(figsize=(12, 8))
        plt.xscale('log', base=2)
        
        # Sort by thread count
        group = group.sort_values('threads')
        
        # Plot speedup
        plt.plot(group['threads'], group['speedup'], marker='o', label='Actual Speedup')
        
        # Add ideal speedup line
        thread_values = sorted(group['threads'].unique())
        if thread_values:
            max_threads = max(thread_values)
            plt.plot(thread_values, thread_values, 'k--', alpha=0.5, label='Ideal Speedup')
        
        # Find point where adding more threads stops helping
        # (when speedup starts decreasing or plateaus)
        max_speedup_idx = group['speedup'].idxmax()
        max_speedup_threads = group.loc[max_speedup_idx, 'threads']
        max_speedup = group.loc[max_speedup_idx, 'speedup']
        
        plt.axvline(x=max_speedup_threads, color='r', linestyle='--', 
                   label=f'Max Effective Threads: {max_speedup_threads}')
        
        plt.title(f'Thread Scaling Limit: {alg} on {dataset}')
        plt.xlabel('Number of Threads (log scale)')
        plt.ylabel('Speedup')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(thread_values, thread_values)
        plt.tight_layout()
        
        # Create directory for scaling limits
        scaling_dir = os.path.join(output_dir, 'scaling_limits')
        os.makedirs(scaling_dir, exist_ok=True)
        
        plt.savefig(os.path.join(scaling_dir, f'scaling_limit_{alg}_{dataset}.png'), dpi=300)
        plt.close()

def generate_summary_report(metrics_df, rec_df, output_dir):
    """Generate a summary report with key findings"""
    with open(os.path.join(output_dir, 'summary_report.md'), 'w') as f:
        f.write("# Parallel WCOJ Benchmark Summary Report\n\n")
        
        # Overall best algorithm
        thread_counts = sorted(metrics_df['threads'].unique())
        max_threads = max(thread_counts)
        max_thread_df = metrics_df[metrics_df['threads'] == max_threads]
        
        if not max_thread_df.empty:
            best_overall_idx = max_thread_df['speedup'].idxmax()
            best_overall = max_thread_df.loc[best_overall_idx]
            
            f.write("## Overall Best Performance\n\n")
            f.write(f"- **Best Algorithm Overall**: {best_overall['algorithm']}\n")
            f.write(f"- **Max Speedup Achieved**: {best_overall['speedup']:.2f}x (with {max_threads} threads)\n")
            f.write(f"- **Dataset**: {best_overall['dataset']}\n")
            f.write(f"- **Execution Time**: {best_overall['avg_time']:.6f} seconds\n\n")
        
        # Thread scaling analysis
        f.write("## Thread Scaling Analysis\n\n")
        f.write("| Algorithm | Best Thread Count | Max Speedup | Efficiency at Max |\n")
        f.write("|-----------|------------------|-------------|-------------------|\n")
        
        for alg, group in metrics_df.groupby('algorithm'):
            # Find thread count with best speedup
            max_speedup_idx = group['speedup'].idxmax()
            max_speedup_threads = group.loc[max_speedup_idx, 'threads']
            max_speedup = group.loc[max_speedup_idx, 'speedup']
            efficiency = group.loc[max_speedup_idx, 'efficiency']
            
            f.write(f"| {alg} | {max_speedup_threads} | {max_speedup:.2f}x | {efficiency:.2f} |\n")
        
        f.write("\n")
        
        # Algorithm performance at different thread counts
        f.write("## Algorithm Performance at Different Thread Counts\n\n")
        
        for threads in [1, 8, 32, max_threads]:
            if threads not in thread_counts:
                continue
                
            thread_df = metrics_df[metrics_df['threads'] == threads]
            
            if thread_df.empty:
                continue
                
            f.write(f"### Thread Count: {threads}\n\n")
            f.write("| Algorithm | Avg Speedup | Max Speedup | Avg Efficiency |\n")
            f.write("|-----------|-------------|-------------|----------------|\n")
            
            for alg, alg_group in thread_df.groupby('algorithm'):
                avg_speedup = alg_group['speedup'].mean()
                max_speedup = alg_group['speedup'].max()
                avg_efficiency = alg_group['efficiency'].mean()
                
                f.write(f"| {alg} | {avg_speedup:.2f}x | {max_speedup:.2f}x | {avg_efficiency:.2f} |\n")
            
            f.write("\n")
        
        # Dataset impact
        f.write("## Impact of Dataset Characteristics\n\n")
        
        # Size impact
        f.write("### Dataset Size Impact\n\n")
        for threads in [8, max_threads]:
            if threads not in thread_counts:
                continue
                
            thread_df = metrics_df[metrics_df['threads'] == threads]
            
            if thread_df.empty:
                continue
                
            f.write(f"Average speedup by dataset size (Threads: {threads}):\n\n")
            size_impact = thread_df.groupby('size')['speedup'].mean().sort_values(ascending=False)
            for size, speedup in size_impact.items():
                f.write(f"- **{size}**: {speedup:.2f}x\n")
            f.write("\n")
        
        # Algorithm scaling behavior by dataset size
        f.write("## Algorithm Scaling Behavior by Dataset Size\n\n")
        
        for size in metrics_df['size'].unique():
            f.write(f"### {size.capitalize()} Datasets\n\n")
            
            size_df = metrics_df[metrics_df['size'] == size]
            
            # Find best thread count for each algorithm on this dataset size
            f.write("| Algorithm | Optimal Thread Count | Max Speedup |\n")
            f.write("|-----------|----------------------|-------------|\n")
            
            for alg, alg_group in size_df.groupby('algorithm'):
                # Find best thread count
                best_idx = alg_group['speedup'].idxmax()
                best_threads = alg_group.loc[best_idx, 'threads']
                max_speedup = alg_group.loc[best_idx, 'speedup']
                
                f.write(f"| {alg} | {best_threads} | {max_speedup:.2f}x |\n")
            
            f.write("\n")
        
        # Scaling limits
        f.write("## Scaling Limits\n\n")
        f.write("The point where adding more threads stops providing significant benefits:\n\n")
        
        scaling_limits = defaultdict(list)
        
        for (alg, dataset), group in metrics_df.groupby(['algorithm', 'dataset']):
            if len(group) < 3:
                continue
                
            # Find thread count with best speedup
            best_idx = group['speedup'].idxmax()
            best_threads = group.loc[best_idx, 'threads']
            
            scaling_limits[alg].append(best_threads)
        
        f.write("| Algorithm | Median Scaling Limit | Min | Max |\n")
        f.write("|-----------|----------------------|-----|-----|\n")
        
        for alg, limits in scaling_limits.items():
            if not limits:
                continue
                
            median_limit = np.median(limits)
            min_limit = min(limits)
            max_limit = max(limits)
            
            f.write(f"| {alg} | {median_limit} | {min_limit} | {max_limit} |\n")
        
        f.write("\n")
        
        # Conclusions
        f.write("## Conclusions\n\n")
        
        # Overall best algorithm
        if 'best_overall' in locals():
            f.write(f"1. The algorithm with best overall performance is **{best_overall['algorithm']}** with a maximum speedup of {best_overall['speedup']:.2f}x using {max_threads} threads.\n\n")
        
        # Thread scaling behavior
        f.write("2. Thread scaling behavior:\n")
        
        for alg, limits in scaling_limits.items():
            if not limits:
                continue
                
            median_limit = np.median(limits)
            f.write(f"   - **{alg}** typically scales well up to **{median_limit}** threads\n")
        
        f.write("\n")
        
        # Algorithm recommendations
        f.write("3. Algorithm recommendations based on dataset characteristics:\n\n")
        
        # For each dataset characteristic, find the most common best algorithm
        for threads in [8, 32, max_threads]:
            if threads not in rec_df['threads'].unique():
                continue
                
            thread_rec_df = rec_df[rec_df['threads'] == threads]
            
            f.write(f"   **Using {threads} threads:**\n")
            
            # Best algorithm by size
            for size in thread_rec_df['size'].unique():
                size_rec = thread_rec_df[thread_rec_df['size'] == size]
                alg_counts = size_rec['best_algorithm'].value_counts()
                if not alg_counts.empty:
                    best_alg = alg_counts.index[0]
                    f.write(f"   - For **{size}** datasets: Use **{best_alg}**\n")
            
            f.write("\n")

def main():
    parser = argparse.ArgumentParser(description="Analyze parallel WCOJ benchmark results")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory containing benchmark results")
    args = parser.parse_args()
    
    # Create plots directory
    plots_dir = os.path.join(args.results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load results
    results_df = load_results(args.results_dir)
    
    # Calculate metrics
    metrics_df = calculate_metrics(results_df)
    
    # Save metrics
    metrics_df.to_csv(os.path.join(args.results_dir, "metrics.csv"), index=False)
    
    # Generate plots
    plot_speedup_by_algorithm(metrics_df, plots_dir)
    plot_efficiency_by_algorithm(metrics_df, plots_dir)
    plot_execution_time_comparison(metrics_df, plots_dir)
    plot_speedup_scaling(metrics_df, plots_dir)
    plot_thread_scaling_limit(metrics_df, plots_dir)
    
    # Plot heatmap
    plot_heatmap_algorithm_dataset(metrics_df, plots_dir)
    
    # Create recommendation table
    rec_df = create_recommendation_table(metrics_df, args.results_dir)
    
    # Generate summary report
    generate_summary_report(metrics_df, rec_df, args.results_dir)
    
    print(f"Analysis complete. Results saved to {args.results_dir}")

if __name__ == "__main__":
    main()