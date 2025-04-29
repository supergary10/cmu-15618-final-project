import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from collections import defaultdict
import re

def load_results(results_dir):
    summary_path = os.path.join(results_dir, "summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    df = pd.read_csv(summary_path)
    return df

def extract_dataset_features(dataset_name):
    features = {}
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

    if "high_overlap" in dataset_name:
        features["overlap"] = "high"
    elif "low_overlap" in dataset_name:
        features["overlap"] = "low"
    else:
        features["overlap"] = "medium"

    if "many_vectors" in dataset_name:
        features["vectors"] = "many"
    else:
        features["vectors"] = "few"

    return features

def calculate_metrics(df):
    metrics = []
    for (alg, dataset), group in df.groupby(['algorithm', 'dataset']):
        seq_time_series = group[group['threads'] == 1]['exec_time']
        if seq_time_series.empty:
             print(f"Warning: No data found for threads=1 for algorithm '{alg}' on dataset '{dataset}'. Cannot calculate speedup/efficiency.")
             seq_time = np.nan # Assign NaN if no sequential data
        else:
             seq_time = seq_time_series.mean()


        for threads, thread_group in group.groupby('threads'):
            avg_time = thread_group['exec_time'].mean()
            min_time = thread_group['exec_time'].min()
            max_time = thread_group['exec_time'].max()
            std_time = thread_group['exec_time'].std() if len(thread_group) > 1 else 0

            if pd.isna(seq_time) or avg_time <= 0:
                 speedup = np.nan
                 efficiency = np.nan
            else:
                 speedup = seq_time / avg_time
                 efficiency = speedup / threads

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
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=metrics_df, x='threads', y='speedup', hue='algorithm', marker='o', errorbar='sd')
    max_threads = metrics_df['threads'].max()
    if not pd.isna(max_threads) and max_threads > 0:
        plt.plot([1, max_threads], [1, max_threads], 'k--', alpha=0.5, label='Ideal Speedup')
    plt.title('Speedup vs Thread Count by Algorithm')
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup')
    plt.legend(title='Algorithm')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speedup_by_algorithm.png'), dpi=300)
    plt.close()

def plot_efficiency_by_algorithm(metrics_df, output_dir):
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=metrics_df, x='threads', y='efficiency', hue='algorithm', marker='o')
    plt.title('Parallel Efficiency vs Thread Count by Algorithm')
    plt.xlabel('Number of Threads')
    plt.ylabel('Efficiency (Speedup/Threads)')
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Ideal Efficiency')
    plt.legend(title='Algorithm')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'efficiency_by_algorithm.png'), dpi=300)
    plt.close()

def plot_execution_time_comparison(metrics_df, output_dir):
    max_thread_df = metrics_df.loc[metrics_df.groupby(['algorithm', 'dataset'])['threads'].idxmax()]
    max_thread_df['log_time'] = np.log10(max_thread_df['avg_time'].replace(0, np.nan)) # Avoid log(0)

    plt.figure(figsize=(14, 8))
    sns.barplot(data=max_thread_df, x='algorithm', y='log_time', hue='size')
    plt.title(f'Log Execution Time by Algorithm and Dataset Size (Max Thread Count)')
    plt.xlabel('Algorithm')
    plt.ylabel('Log10(Execution Time) in seconds')
    plt.legend(title='Dataset Size')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'execution_time_comparison.png'), dpi=300)
    plt.close()

def plot_speedup_by_dataset_feature(metrics_df, feature, output_dir):
    plt.figure(figsize=(12, 8))
    max_threads = metrics_df['threads'].max()
    if pd.isna(max_threads) or max_threads == 0:
         print(f"Skipping plot_speedup_by_dataset_feature for {feature} due to invalid max_threads.")
         plt.close()
         return
    filtered_df = metrics_df[metrics_df['threads'] == max_threads]
    if filtered_df.empty:
        print(f"Skipping plot_speedup_by_dataset_feature for {feature} as no data found for max threads ({max_threads}).")
        plt.close()
        return

    sns.barplot(data=filtered_df, x='algorithm', y='speedup', hue=feature)
    plt.title(f'Speedup by Algorithm and Dataset {feature.capitalize()} (Threads: {max_threads})')
    plt.xlabel('Algorithm')
    plt.ylabel('Speedup')
    plt.legend(title=feature.capitalize())
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'speedup_by_{feature}.png'), dpi=300)
    plt.close()

def plot_heatmap_algorithm_dataset(metrics_df, output_dir):
    max_threads = metrics_df['threads'].max()
    if pd.isna(max_threads) or max_threads == 0:
        print("Skipping plot_heatmap_algorithm_dataset due to invalid max_threads.")
        return
    filtered_df = metrics_df[metrics_df['threads'] == max_threads]
    if filtered_df.empty:
        print(f"Skipping plot_heatmap_algorithm_dataset as no data found for max threads ({max_threads}).")
        return

    pivot_speedup = filtered_df.pivot_table(
        index='algorithm',
        columns='dataset',
        values='speedup'
    )
    if pivot_speedup.empty:
        print("Skipping plot_heatmap_algorithm_dataset as pivot table is empty.")
        return

    plt.figure(figsize=(16, 10))
    sns.heatmap(pivot_speedup, annot=True, cmap='viridis', fmt='.2f')
    plt.title(f'Speedup Heatmap: Algorithm vs Dataset (Threads: {max_threads})')
    plt.xlabel('Dataset')
    plt.ylabel('Algorithm')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'algorithm_dataset_heatmap.png'), dpi=300)
    plt.close()

def create_recommendation_table(metrics_df, output_dir):
    recommendations = defaultdict(list)
    feature_combinations = metrics_df.groupby(['size', 'distribution', 'overlap', 'vectors'])
    max_threads = metrics_df['threads'].max()
    if pd.isna(max_threads) or max_threads == 0:
        print("Cannot create recommendation table due to invalid max_threads.")
        return pd.DataFrame()

    for features, group in feature_combinations:
        size, dist, overlap, vectors = features
        max_thread_group = group[group['threads'] == max_threads]
        if max_thread_group.empty:
            continue

        best_alg_idx = max_thread_group['speedup'].idxmax()
        if pd.isna(best_alg_idx): # Handle cases where all speedups might be NaN
             best_alg_row = max_thread_group.iloc[0] # Fallback or skip? Let's skip for now
             print(f"Warning: Could not determine best algorithm for features: {features}")
             continue
        else:
             best_alg_row = max_thread_group.loc[best_alg_idx]


        recommendation = {
            'size': size,
            'distribution': dist,
            'overlap': overlap,
            'vectors': vectors,
            'best_algorithm': best_alg_row['algorithm'],
            'speedup': best_alg_row['speedup'],
            'avg_time': best_alg_row['avg_time']
        }
        recommendations['recommendations'].append(recommendation)

    rec_df = pd.DataFrame(recommendations['recommendations'])
    if not rec_df.empty:
        rec_df.to_csv(os.path.join(output_dir, 'algorithm_recommendations.csv'), index=False)
    return rec_df

def generate_summary_report(metrics_df, rec_df, output_dir):
    with open(os.path.join(output_dir, 'summary_report.md'), 'w') as f:
        f.write("# Parallel WCOJ Benchmark Summary Report\n\n")

        max_threads = metrics_df['threads'].max()
        if pd.isna(max_threads) or max_threads == 0:
            f.write("## Error: Could not generate report due to missing or invalid thread data.\n")
            return

        max_thread_df = metrics_df[metrics_df['threads'] == max_threads]
        if max_thread_df.empty:
            f.write(f"## Error: No data found for max threads ({max_threads}). Cannot generate report.\n")
            return

        best_overall_idx = max_thread_df['speedup'].idxmax()
        if pd.isna(best_overall_idx):
             f.write("## Overall Best Performance\n\n")
             f.write("- Could not determine overall best algorithm (likely due to missing speedup data).\n\n")
        else:
             best_overall = max_thread_df.loc[best_overall_idx]
             f.write("## Overall Best Performance\n\n")
             f.write(f"- **Best Algorithm Overall**: {best_overall['algorithm']}\n")
             f.write(f"- **Max Speedup Achieved**: {best_overall['speedup']:.2f}x (with {max_threads} threads)\n")
             f.write(f"- **Dataset**: {best_overall['dataset']}\n")
             f.write(f"- **Execution Time**: {best_overall['avg_time']:.6f} seconds\n\n")

        f.write("## Algorithm Performance Summary\n\n")
        f.write("| Algorithm | Avg Speedup | Max Speedup | Avg Efficiency | Best Dataset Type |\n")
        f.write("|-----------|-------------|-------------|----------------|-------------------|\n")

        for alg, group in max_thread_df.groupby('algorithm'):
            avg_speedup = group['speedup'].mean()
            max_speedup = group['speedup'].max()
            avg_efficiency = group['efficiency'].mean()

            best_idx = group['speedup'].idxmax()
            if pd.isna(best_idx):
                 best_type = "N/A"
            else:
                 best_dataset = group.loc[best_idx]
                 best_type = f"{best_dataset['size']}, {best_dataset['distribution']}, {best_dataset['overlap']} overlap"

            avg_s_str = f"{avg_speedup:.2f}x" if not pd.isna(avg_speedup) else "N/A"
            max_s_str = f"{max_speedup:.2f}x" if not pd.isna(max_speedup) else "N/A"
            avg_e_str = f"{avg_efficiency:.2f}" if not pd.isna(avg_efficiency) else "N/A"

            f.write(f"| {alg} | {avg_s_str} | {max_s_str} | {avg_e_str} | {best_type} |\n")
        f.write("\n")

        f.write("## Impact of Dataset Characteristics\n\n")

        f.write("### Dataset Size Impact\n\n")
        size_impact = max_thread_df.groupby('size')['speedup'].mean().sort_values(ascending=False)
        f.write("Average speedup by dataset size:\n\n")
        for size, speedup in size_impact.items():
            f.write(f"- **{size}**: {speedup:.2f}x\n" if not pd.isna(speedup) else f"- **{size}**: N/A\n")
        f.write("\n")

        f.write("### Distribution Impact\n\n")
        dist_impact = max_thread_df.groupby('distribution')['speedup'].mean().sort_values(ascending=False)
        f.write("Average speedup by data distribution:\n\n")
        for dist, speedup in dist_impact.items():
            f.write(f"- **{dist}**: {speedup:.2f}x\n" if not pd.isna(speedup) else f"- **{dist}**: N/A\n")
        f.write("\n")

        f.write("### Overlap Impact\n\n")
        overlap_impact = max_thread_df.groupby('overlap')['speedup'].mean().sort_values(ascending=False)
        f.write("Average speedup by overlap level:\n\n")
        for overlap, speedup in overlap_impact.items():
            f.write(f"- **{overlap}**: {speedup:.2f}x\n" if not pd.isna(speedup) else f"- **{overlap}**: N/A\n")
        f.write("\n")

        f.write("### Number of Vectors Impact\n\n")
        vectors_impact = max_thread_df.groupby('vectors')['speedup'].mean().sort_values(ascending=False)
        f.write("Average speedup by number of vectors:\n\n")
        for vectors, speedup in vectors_impact.items():
            f.write(f"- **{vectors}**: {speedup:.2f}x\n" if not pd.isna(speedup) else f"- **{vectors}**: N/A\n")
        f.write("\n")

        if not rec_df.empty:
            f.write("## Algorithm Recommendations\n\n")
            f.write("Based on the benchmark results, here are our recommendations for choosing the optimal algorithm:\n\n")
            f.write("| Dataset Characteristics | Recommended Algorithm | Avg Speedup |\n")
            f.write("|--------------------------|------------------------|-------------|\n")
            for _, row in rec_df.iterrows():
                characteristics = f"{row['size']}, {row['distribution']}, {row['overlap']} overlap, {row['vectors']} vectors"
                speedup_str = f"{row['speedup']:.2f}x" if not pd.isna(row['speedup']) else "N/A"
                f.write(f"| {characteristics} | {row['best_algorithm']} | {speedup_str} |\n")
            f.write("\n")

            f.write("## Conclusions\n\n")
            pattern_counts = defaultdict(int)
            for _, row in rec_df.iterrows():
                pattern_counts[row['best_algorithm']] += 1

            if pattern_counts:
                most_common_alg = max(pattern_counts.items(), key=lambda x: x[1])[0]
                f.write(f"1. The most consistently effective algorithm across different datasets appears to be **{most_common_alg}**.\n\n")
            else:
                f.write("1. Could not determine the most consistently effective algorithm.\n\n")

            size_alg_counts = defaultdict(lambda: defaultdict(int))
            for _, row in rec_df.iterrows():
                size_alg_counts[row['size']][row['best_algorithm']] += 1

            f.write("2. Dataset size impact on algorithm selection:\n")
            for size in sorted(size_alg_counts.keys()):
                 alg_counts = size_alg_counts[size]
                 if alg_counts:
                      best_alg = max(alg_counts.items(), key=lambda x: x[1])[0]
                      f.write(f"   - For **{size}** datasets: **{best_alg}** performs best\n")
                 else:
                      f.write(f"   - For **{size}** datasets: No clear best algorithm found\n")
            f.write("\n")


            dist_alg_counts = defaultdict(lambda: defaultdict(int))
            for _, row in rec_df.iterrows():
                dist_alg_counts[row['distribution']][row['best_algorithm']] += 1

            f.write("3. Data distribution impact on algorithm selection:\n")
            for dist in sorted(dist_alg_counts.keys()):
                 alg_counts = dist_alg_counts[dist]
                 if alg_counts:
                      best_alg = max(alg_counts.items(), key=lambda x: x[1])[0]
                      f.write(f"   - For **{dist}** distributions: **{best_alg}** performs best\n")
                 else:
                      f.write(f"   - For **{dist}** distributions: No clear best algorithm found\n")
            f.write("\n")

            if pattern_counts:
                f.write("4. Overall recommendations:\n")
                f.write("   - For general-purpose use: Use **" + most_common_alg + "**\n")

                small_best = max(size_alg_counts.get('small', {}).items(), key=lambda x: x[1], default=('N/A', 0))[0]
                f.write(f"   - For small datasets: Use **{small_best}**\n")

                large_best = max(size_alg_counts.get('large', {}).items(), key=lambda x: x[1], default=('N/A', 0))[0]
                f.write(f"   - For large datasets: Use **{large_best}**\n")

                skewed_best = max(dist_alg_counts.get('skewed', {}).items(), key=lambda x: x[1], default=('N/A', 0))[0]
                f.write(f"   - For skewed distributions: Use **{skewed_best}**\n")
        else:
             f.write("## Algorithm Recommendations\n\n")
             f.write("No recommendation table generated (likely due to missing data).\n\n")
             f.write("## Conclusions\n\n")
             f.write("Could not generate conclusions due to missing recommendation data.\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze parallel WCOJ benchmark results")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory containing benchmark results (summary.csv)")
    args = parser.parse_args()

    plots_dir = os.path.join(args.results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    try:
        results_df = load_results(args.results_dir)
    except FileNotFoundError as e:
        print(e)
        exit(1)
    except Exception as e:
        print(f"Error loading results: {e}")
        exit(1)


    metrics_df = calculate_metrics(results_df)
    if metrics_df.empty:
         print("Error: No metrics could be calculated. Check input data format and content.")
         exit(1)


    metrics_df.to_csv(os.path.join(args.results_dir, "metrics.csv"), index=False)

    try:
        plot_speedup_by_algorithm(metrics_df, plots_dir)
        plot_efficiency_by_algorithm(metrics_df, plots_dir)
        plot_execution_time_comparison(metrics_df, plots_dir)
        plot_speedup_by_dataset_feature(metrics_df, 'size', plots_dir)
        plot_speedup_by_dataset_feature(metrics_df, 'distribution', plots_dir)
        plot_speedup_by_dataset_feature(metrics_df, 'overlap', plots_dir)
        plot_speedup_by_dataset_feature(metrics_df, 'vectors', plots_dir)
        plot_heatmap_algorithm_dataset(metrics_df, plots_dir)
    except Exception as e:
         print(f"Error during plotting: {e}")
         # Continue to report generation if possible


    rec_df = create_recommendation_table(metrics_df, args.results_dir)


    try:
        generate_summary_report(metrics_df, rec_df, args.results_dir)
    except Exception as e:
         print(f"Error generating summary report: {e}")


    print(f"Analysis complete. Results saved to {args.results_dir}")

if __name__ == "__main__":
    main()
