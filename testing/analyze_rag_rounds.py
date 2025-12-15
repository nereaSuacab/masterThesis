# analyze_rag_rounds.py
import os
import sys
import importlib.util
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define your file paths
ROUNDS = {
    'System Testing': {
        'Round 3': r'testing/system_testing/round3/deepseek/metric_arrays.py',
        'Round 4': r'testing/system_testing/round4/deepseek/metric_arrays.py',
        'Round 5': r'testing/system_testing/round5/deepseek/metric_arrays.py',
        'Round 6': r'testing/system_testing/round6/deepseek/metric_arrays.py',
    },
    'K Testing': {
        'K=3': r'testing/k_testing/k3/deepseek/metric_arrays.py',
        'K=5': r'testing/k_testing/k5/deepseek/metric_arrays.py',
        'K=7': r'testing/k_testing/k7/deepseek/metric_arrays.py',
    }
}

def load_round_data(file_path):
    """Load metric arrays from a Python file"""
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return None, None
    
    spec = importlib.util.spec_from_file_location("metric_arrays", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    dense = {
        'faithfulness': module.faithfulness_dense,
        'context_precision': module.context_precision_dense,
        'context_recall': module.context_recall_dense,
        'noise_sensitivity': module.noise_sensitivity_dense
    }
    
    sparse = {
        'faithfulness': module.faithfulness_sparse,
        'context_precision': module.context_precision_sparse,
        'context_recall': module.context_recall_sparse,
        'noise_sensitivity': module.noise_sensitivity_sparse
    }
    
    return dense, sparse

def load_all_rounds():
    """Load all rounds from defined paths"""
    all_data = {}
    
    for category, rounds in ROUNDS.items():
        print(f"\nLoading {category}...")
        all_data[category] = {
            'dense': [],
            'sparse': [],
            'names': []
        }
        
        for round_name, file_path in rounds.items():
            print(f"  Loading {round_name}: {file_path}")
            dense, sparse = load_round_data(file_path)
            
            if dense is not None and sparse is not None:
                all_data[category]['dense'].append(dense)
                all_data[category]['sparse'].append(sparse)
                all_data[category]['names'].append(round_name)
                print(f"    ✓ Loaded successfully")
            else:
                print(f"    ✗ Failed to load")
    
    return all_data

def compare_rounds(round1, round2, metric_name, approach_name, round_label):
    """Compare metrics between two consecutive rounds"""
    
    # Calculate basic statistics
    r1_mean = np.mean(round1)
    r2_mean = np.mean(round2)
    diff_mean = r2_mean - r1_mean
    pct_change = (diff_mean / r1_mean * 100) if r1_mean != 0 else float('inf')
    
    # Paired t-test
    t_stat, t_pval = stats.ttest_rel(round2, round1)
    
    # Wilcoxon signed-rank test (non-parametric)
    try:
        wilcoxon_stat, wilcoxon_pval = stats.wilcoxon(round2, round1, zero_method='zsplit', alternative='two-sided')
    except ValueError:
        # Handle case where all differences are zero
        wilcoxon_pval = 1.0
    
    # Effect size (Cohen's d for paired samples)
    differences = np.array(round2) - np.array(round1)
    std_diff = np.std(differences, ddof=1)
    cohens_d = np.mean(differences) / std_diff if std_diff != 0 else 0
    
    # Determine significance level
    if wilcoxon_pval < 0.001:
        sig = '***'
    elif wilcoxon_pval < 0.01:
        sig = '**'
    elif wilcoxon_pval < 0.05:
        sig = '*'
    else:
        sig = ''
    
    return {
        'Comparison': round_label,
        'Approach': approach_name,
        'Metric': metric_name,
        'R1 Mean': f'{r1_mean:.3f}',
        'R2 Mean': f'{r2_mean:.3f}',
        'Δ': f'{diff_mean:+.3f}',
        '% Δ': f'{pct_change:+.1f}%' if pct_change != float('inf') else 'N/A',
        'p-value': f'{wilcoxon_pval:.4f}',
        "Cohen's d": f'{cohens_d:.3f}',
        'Sig': sig
    }

def analyze_category(category_name, dense_rounds, sparse_rounds, round_names):
    """Analyze all consecutive round pairs within a category"""
    
    metrics = ['faithfulness', 'context_precision', 'context_recall', 'noise_sensitivity']
    results = []
    
    print(f"\n{'='*140}")
    print(f"{category_name.upper()} - ROUND-TO-ROUND ANALYSIS")
    print(f"{'='*140}")
    print(f"Rounds: {', '.join(round_names)}")
    
    # Analyze Dense rounds
    for i in range(len(dense_rounds) - 1):
        comparison_label = f"{round_names[i]} → {round_names[i+1]}"
        for metric in metrics:
            if len(dense_rounds[i][metric]) > 0 and len(dense_rounds[i+1][metric]) > 0:
                result = compare_rounds(
                    dense_rounds[i][metric], 
                    dense_rounds[i+1][metric], 
                    metric.replace('_', ' ').title(), 
                    'Dense',
                    comparison_label
                )
                results.append(result)
    
    # Analyze Sparse rounds
    for i in range(len(sparse_rounds) - 1):
        comparison_label = f"{round_names[i]} → {round_names[i+1]}"
        for metric in metrics:
            if len(sparse_rounds[i][metric]) > 0 and len(sparse_rounds[i+1][metric]) > 0:
                result = compare_rounds(
                    sparse_rounds[i][metric], 
                    sparse_rounds[i+1][metric], 
                    metric.replace('_', ' ').title(), 
                    'Sparse',
                    comparison_label
                )
                results.append(result)
    
    if len(results) == 0:
        print("No comparisons available (need at least 2 rounds)")
        return pd.DataFrame()
    
    df_results = pd.DataFrame(results)
    
    # Display results
    print(f"\n{'-'*140}")
    print("DETAILED RESULTS:")
    print(f"{'-'*140}")
    print(df_results.to_string(index=False))
    
    # Summary statistics
    print(f"\n{'-'*140}")
    print("SUMMARY:")
    print(f"{'-'*140}")
    
    for approach in ['Dense', 'Sparse']:
        approach_data = df_results[df_results['Approach'] == approach]
        total = len(approach_data)
        if total > 0:
            sig_001 = (approach_data['Sig'] == '***').sum()
            sig_01 = (approach_data['Sig'] == '**').sum()
            sig_05 = (approach_data['Sig'] == '*').sum()
            not_sig = (approach_data['Sig'] == '').sum()
            
            print(f"\n{approach}:")
            print(f"  Total comparisons: {total}")
            print(f"  p<0.001 (***): {sig_001}")
            print(f"  p<0.01 (**): {sig_01}")
            print(f"  p<0.05 (*): {sig_05}")
            print(f"  Not significant: {not_sig}")
    
    # Significant changes
    sig_results = df_results[df_results['Sig'] != '']
    if len(sig_results) > 0:
        print(f"\n{'-'*140}")
        print("SIGNIFICANT CHANGES:")
        print(f"{'-'*140}")
        for _, row in sig_results.iterrows():
            direction = "↑" if row['Δ'].startswith('+') else "↓"
            print(f"  {direction} {row['Comparison']} | {row['Approach']:6s} | {row['Metric']:20s} | {row['Δ']:7s} ({row['% Δ']:7s}) {row['Sig']}")
    
    return df_results

def visualize_category(category_name, dense_rounds, sparse_rounds, round_names, output_prefix):
    """Create visualizations for a category"""
    
    metrics = ['faithfulness', 'context_precision', 'context_recall', 'noise_sensitivity']
    
    # Trend lines
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{category_name} - Metrics Evolution', fontsize=16, fontweight='bold')
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # Calculate means and stds
        dense_means = [np.mean(r[metric]) for r in dense_rounds]
        sparse_means = [np.mean(r[metric]) for r in sparse_rounds]
        dense_stds = [np.std(r[metric]) for r in dense_rounds]
        sparse_stds = [np.std(r[metric]) for r in sparse_rounds]
        
        x_pos = list(range(len(round_names)))
        
        # Plot lines with error bands
        ax.plot(x_pos, dense_means, 'o-', label='Dense', linewidth=2.5, markersize=10, color='#2E86AB')
        ax.fill_between(x_pos, 
                       [m - s for m, s in zip(dense_means, dense_stds)],
                       [m + s for m, s in zip(dense_means, dense_stds)],
                       alpha=0.2, color='#2E86AB')
        
        ax.plot(x_pos, sparse_means, 's-', label='Sparse', linewidth=2.5, markersize=10, color='#A23B72')
        ax.fill_between(x_pos,
                       [m - s for m, s in zip(sparse_means, sparse_stds)],
                       [m + s for m, s in zip(sparse_means, sparse_stds)],
                       alpha=0.2, color='#A23B72')
        
        # Add value labels
        for x, y in zip(x_pos, dense_means):
            ax.text(x, y + 0.04, f'{y:.2f}', ha='center', fontsize=9, color='#2E86AB', fontweight='bold')
        for x, y in zip(x_pos, sparse_means):
            ax.text(x, y - 0.06, f'{y:.2f}', ha='center', fontsize=9, color='#A23B72', fontweight='bold')
        
        ax.set_xlabel('Round', fontsize=11, fontweight='bold')
        ax.set_ylabel('Mean Score', fontsize=11, fontweight='bold')
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(-0.05, 1.1)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(round_names, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_trends.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Box plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{category_name} - Distribution Across Rounds', fontsize=16, fontweight='bold')
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        all_data = []
        labels = []
        colors = []
        
        for i, (d, s) in enumerate(zip(dense_rounds, sparse_rounds)):
            all_data.append(d[metric])
            labels.append(f'D\n{round_names[i]}')
            colors.append('#A6CEE3')
            
            all_data.append(s[metric])
            labels.append(f'S\n{round_names[i]}')
            colors.append('#FB9A99')
        
        bp = ax.boxplot(all_data, labels=labels, patch_artist=True, widths=0.6)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(-0.05, 1.05)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main analysis function"""
    
    print("="*140)
    print("RAG EVALUATION - COMPREHENSIVE ANALYSIS")
    print("="*140)
    
    # Load all data
    all_data = load_all_rounds()
    
    # Store all results
    all_results = {}
    
    # Analyze each category
    for category, data in all_data.items():
        if len(data['dense']) > 0:
            df_results = analyze_category(
                category,
                data['dense'],
                data['sparse'],
                data['names']
            )
            
            if not df_results.empty:
                all_results[category] = df_results
                
                # Create visualizations
                output_prefix = category.lower().replace(' ', '_')
                visualize_category(
                    category,
                    data['dense'],
                    data['sparse'],
                    data['names'],
                    output_prefix
                )
                print(f"\n✓ Visualizations saved for {category}")
    
    # Export all results
    print(f"\n{'='*140}")
    print("EXPORTING RESULTS")
    print(f"{'='*140}")
    
    for category, df_results in all_results.items():
        filename = category.lower().replace(' ', '_')
        csv_path = f'{filename}_analysis.csv'
        df_results.to_csv(csv_path, index=False)
        print(f"✓ {category} results exported to: {filename}_analysis.csv")
    
    print(f"\n{'='*140}")
    print("INTERPRETATION GUIDE")
    print(f"{'='*140}")
    print("• Significance: *** p<0.001, ** p<0.01, * p<0.05")
    print("• Cohen's d: |d| < 0.2 (small), 0.2-0.5 (small-medium), 0.5-0.8 (medium), > 0.8 (large)")
    print("• Positive Δ (+) = improvement, Negative Δ (-) = degradation")
    print("• Wilcoxon signed-rank test (non-parametric, robust for [0,1] bounded data)")
    print(f"{'='*140}")
    
    print("\n✓ ANALYSIS COMPLETE!")

if __name__ == "__main__":
    main()