import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <summary.csv>")
        return 1
    
    # Load data
    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        return 1
    
    # Create output directory
    output_dir = "plots_" + os.path.basename(csv_path).replace(".csv", "")
    os.makedirs(output_dir, exist_ok=True)
    
    # Read data
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Convert time to numeric
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    
    # Calculate speedup relative to single thread
    results = []
    for algo, group in df.groupby('algorithm'):
        sequential = group[group['threads'] == 1]['time'].values
        if len(sequential) == 0:
            print(f"Warning: No sequential (thread=1) timing found for {algo}")
            continue
        
        sequential_time = sequential[0]
        
        for _, row in group.iterrows():
            speedup = sequential_time / row['time'] if row['time'] > 0 else 0
            efficiency = speedup / row['threads']
            
            results.append({
                'algorithm': row['algorithm'],
                'threads': row['threads'],
                'time': row['time'],
                'result_size': row['result_size'],
                'speedup': speedup,
                'efficiency': efficiency
            })
    
    metrics_df = pd.DataFrame(results)
    
    # 1. Plot execution time
    plt.figure(figsize=(12, 8))
    plt.xscale('log', base=2)
    plt.yscale('log')
    
    for algo, group in metrics_df.groupby('algorithm'):
        plt.plot(group['threads'], group['time'], marker='o', label=algo)
    
    plt.title('Execution Time vs Thread Count (Log-Log Scale)')
    plt.xlabel('Number of Threads')
    plt.ylabel('Execution Time (seconds)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'execution_time.png'), dpi=300)
    
    # 2. Plot speedup
    plt.figure(figsize=(12, 8))
    plt.xscale('log', base=2)
    
    for algo, group in metrics_df.groupby('algorithm'):
        plt.plot(group['threads'], group['speedup'], marker='o', label=algo)
    
    # Add ideal speedup line
    thread_values = sorted(metrics_df['threads'].unique())
    if thread_values:
        max_threads = max(thread_values)
        plt.plot(thread_values, thread_values, 'k--', alpha=0.5, label='Ideal Speedup')
    
    plt.title('Speedup vs Thread Count')
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'speedup.png'), dpi=300)
    
    # 3. Plot efficiency
    plt.figure(figsize=(12, 8))
    plt.xscale('log', base=2)
    
    for algo, group in metrics_df.groupby('algorithm'):
        plt.plot(group['threads'], group['efficiency'], marker='o', label=algo)
    
    plt.title('Parallel Efficiency vs Thread Count')
    plt.xlabel('Number of Threads')
    plt.ylabel('Efficiency (Speedup/Threads)')
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Ideal Efficiency')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'efficiency.png'), dpi=300)
    
    # 4. Find scaling limits
    scaling_results = []
    for algo, group in metrics_df.groupby('algorithm'):
        max_speedup_idx = group['speedup'].idxmax()
        max_speedup_threads = group.loc[max_speedup_idx, 'threads']
        max_speedup = group.loc[max_speedup_idx, 'speedup']
        
        scaling_results.append({
            'algorithm': algo,
            'max_speedup_threads': max_speedup_threads,
            'max_speedup': max_speedup
        })
    
    scaling_df = pd.DataFrame(scaling_results)
    
    # Plot scaling limit bar chart
    plt.figure(figsize=(12, 8))
    bars = plt.bar(scaling_df['algorithm'], scaling_df['max_speedup_threads'])
    
    plt.title('Thread Count Where Maximum Speedup is Achieved')
    plt.xlabel('Algorithm')
    plt.ylabel('Optimal Thread Count')
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scaling_limits.png'), dpi=300)
    
    # 5. Print summary
    print("\nSummary of Results:")
    print("===================")
    print(f"Total algorithms tested: {len(scaling_df)}")
    print(f"Thread counts tested: {', '.join(map(str, thread_values))}")
    
    # Best algorithm by speedup
    best_algo_idx = scaling_df['max_speedup'].idxmax()
    best_algo = scaling_df.loc[best_algo_idx]
    print(f"\nBest algorithm: {best_algo['algorithm']} with speedup of {best_algo['max_speedup']:.2f}x at {best_algo['max_speedup_threads']} threads")
    
    # Print scaling limits
    print("\nScaling limits (thread count where maximum speedup is achieved):")
    for _, row in scaling_df.iterrows():
        print(f"  {row['algorithm']}: {row['max_speedup_threads']} threads (speedup: {row['max_speedup']:.2f}x)")
    
    print(f"\nPlots saved to {output_dir}/")
    return 0

if __name__ == "__main__":
    sys.exit(main())