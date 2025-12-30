"""
Hybrid RAG Analysis and Visualization
Generates comprehensive PDF report with metrics analysis
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# =============================================================================
# DATA - HYBRID RAG METRICS
# =============================================================================
# NOTE: Replace these arrays with your actual hybrid_metric_arrays.py results
# These are placeholder values - run evaluate_hybrid.py first to generate real data

questions = ['q1','q2','q3','q4','q5','q6','q7','q8','q9','q10','q11','q12',
             'q13','q14','q15','q16','q17','q18','q19','q20','q21','q22','q23','q24']

# Hybrid RAG Metrics (replace with your actual data from hybrid_metric_arrays.py)
# faithfulness_hybrid = [0.6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.75, 1.0, 0.67, 1.0, 0.67, 0.83, 0.8, 1.0, 0.75, 0.33, 0.5, 1.0, 0.33, 0.75, 1.0, 0.86]
# context_precision_hybrid = [0.5, 0.58, 1.0, 1.0, 0.58, 1.0, 0.58, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.33, 0.0, 0.5, 1.0]
# context_recall_hybrid = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.4, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0]
# noise_sensitivity_hybrid = [0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.67, 0.0]

faithfulness_hybrid = [0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.75, 0.0, 0.67, 0.67, 1.0, 0.83, 0.67, 0.6, 0.75, 1.0, 0.5, 1.0, 0.6, 0.75, 1.0, 0.57]
context_precision_hybrid = [0.5, 0.58, 1.0, 1.0, 0.58, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.33, 0.0, 0.5, 1.0]
context_recall_hybrid = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0]
noise_sensitivity_hybrid = [0.75, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.67, 0.0, 0.0, 0.0, 0.0, 0.33, 0.0]

# Try to import from generated file if it exists
try:
    from testing.hybrid.rff.hybrid_metric_arrays import (
        faithfulness_hybrid,
        context_precision_hybrid,
        context_recall_hybrid,
        noise_sensitivity_hybrid
    )
    print("✓ Successfully loaded metrics from hybrid_metric_arrays.py")
except ImportError:
    print("⚠ Warning: Could not import from hybrid_metric_arrays.py")
    print("   Using placeholder values. Run evaluate_hybrid.py first to generate real data.")
    print("   Then update this script or ensure hybrid_metric_arrays.py is in the same directory.")

# Verify all arrays have correct length
assert len(questions) == 24, f"questions has {len(questions)} elements (expected 24)"
assert len(faithfulness_hybrid) == 24, f"faithfulness_hybrid has {len(faithfulness_hybrid)} elements (expected 24)"
assert len(context_precision_hybrid) == 24, f"context_precision_hybrid has {len(context_precision_hybrid)} elements (expected 24)"
assert len(context_recall_hybrid) == 24, f"context_recall_hybrid has {len(context_recall_hybrid)} elements (expected 24)"
assert len(noise_sensitivity_hybrid) == 24, f"noise_sensitivity_hybrid has {len(noise_sensitivity_hybrid)} elements (expected 24)"

# Question categories (same as before)
question_categories = {
    'q1': "Solution", 'q2': "Solution", 'q3': "Accessory", 'q4': "Accessory",
    'q5': "Solution", 'q6': "Solution", 'q7': "Solution", 'q8': "Solution",
    'q9': "Product Variant", 'q10': "Product Variant", 'q11': "Product Variant", 'q12': "Solution",
    'q13': "Accessory", 'q14': "Accessory", 'q15': "Accessory", 'q16': "Accessory",
    'q17': "Product Variant", 'q18': "Solution", 'q19': "Product Variant", 'q20': "Product Variant",
    'q21': "Solution", 'q22': "Product Variant", 'q23': "Product Variant", 'q24': "Solution"
}

question_texts = {
    'q1': "ISO standards room acoustics/speech intelligibility",
    'q2': "Façade sound insulation testing",
    'q3': "Calibrated speech intelligibility with DIRAC",
    'q4': "ISO 3382 room acoustics measurements",
    'q5': "ISO 9612 workplace noise exposure",
    'q6': "Environmental noise complaints investigation",
    'q7': "Exhaust noise in vehicles",
    'q8': "Toys and machinery noise emissions",
    'q9': "HBK 2255 long-term environmental monitoring",
    'q10': "HBK 2255 workplace noise ISO 9612",
    'q11': "HBK 2255 sound insulation testing",
    'q12': "Software module isolating noise events",
    'q13': "Loudspeaker airborne sound insulation",
    'q14': "Amplifier for OmniPower concert halls",
    'q15': "Amplifier remote control mobile device",
    'q16': "ISO 3382 reverberation time office buildings",
    'q17': "B&K 2245 sound power measurements",
    'q18': "B&K 2245 ISO 3744 software",
    'q19': "B&K 2245 environmental noise surveys",
    'q20': "Roadside exhaust noise enforcement",
    'q21': "Quick spot checks urban environments",
    'q22': "HBK 2255 basic environmental monitoring",
    'q23': "HBK 2255 environmental + calibrator",
    'q24': "Community noise monitoring mobile app"
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def calc_avg(arr):
    return np.mean(arr)

def calc_std(arr):
    return np.std(arr)

def calc_median(arr):
    return np.median(arr)

def calc_min_max(arr):
    return np.min(arr), np.max(arr)

# =============================================================================
# CREATE PDF
# =============================================================================
with PdfPages('hybrid_rag_analysis.pdf') as pdf:
    
    # =========================================================================
    # PAGE 1: OVERVIEW WITH SUMMARY STATISTICS
    # =========================================================================
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Hybrid RAG - Performance Analysis Overview', fontsize=16, fontweight='bold')

    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1.1 Average Metrics Bar Chart
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['Faithfulness', 'Context\nPrecision', 'Context\nRecall', 'Noise\nSensitivity']
    hybrid_avgs = [
        calc_avg(faithfulness_hybrid), 
        calc_avg(context_precision_hybrid), 
        calc_avg(context_recall_hybrid), 
        calc_avg(noise_sensitivity_hybrid)
    ]

    x = np.arange(len(metrics))
    width = 0.6
    colors = ['#10b981', '#3b82f6', '#8b5cf6', '#ef4444']
    bars = ax1.bar(x, hybrid_avgs, width, color=colors, alpha=0.8)
    
    ax1.set_ylabel('Score')
    ax1.set_title('Hybrid RAG - Average Metrics')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f'{h:.3f}', 
                    xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), 
                    textcoords="offset points", 
                    ha='center', 
                    fontsize=10,
                    fontweight='bold')

    # 1.2 Radar Chart
    ax2 = fig.add_subplot(gs[0, 1], projection='polar')
    categories_radar = ['Faithfulness', 'Ctx Precision', 'Ctx Recall', 'Noise Sens.']
    N = len(categories_radar)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    hybrid_vals = hybrid_avgs + [hybrid_avgs[0]]

    ax2.plot(angles, hybrid_vals, 'o-', linewidth=2, color='#8b5cf6')
    ax2.fill(angles, hybrid_vals, alpha=0.25, color='#8b5cf6')
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories_radar, fontsize=9)
    ax2.set_ylim(0, 1)
    ax2.set_title('Hybrid RAG Profile', pad=20, fontweight='bold')
    ax2.grid(True)
    
    # 1.3 Summary Statistics Table
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')
    
    summary_data = [
        ['Metric', 'Average', 'Median', 'Std Dev', 'Min', 'Max'],
        ['Faithfulness', 
         f'{calc_avg(faithfulness_hybrid):.3f}',
         f'{calc_median(faithfulness_hybrid):.3f}',
         f'{calc_std(faithfulness_hybrid):.3f}',
         f'{np.min(faithfulness_hybrid):.3f}',
         f'{np.max(faithfulness_hybrid):.3f}'],
        ['Context Precision', 
         f'{calc_avg(context_precision_hybrid):.3f}',
         f'{calc_median(context_precision_hybrid):.3f}',
         f'{calc_std(context_precision_hybrid):.3f}',
         f'{np.min(context_precision_hybrid):.3f}',
         f'{np.max(context_precision_hybrid):.3f}'],
        ['Context Recall', 
         f'{calc_avg(context_recall_hybrid):.3f}',
         f'{calc_median(context_recall_hybrid):.3f}',
         f'{calc_std(context_recall_hybrid):.3f}',
         f'{np.min(context_recall_hybrid):.3f}',
         f'{np.max(context_recall_hybrid):.3f}'],
        ['Noise Sensitivity', 
         f'{calc_avg(noise_sensitivity_hybrid):.3f}',
         f'{calc_median(noise_sensitivity_hybrid):.3f}',
         f'{calc_std(noise_sensitivity_hybrid):.3f}',
         f'{np.min(noise_sensitivity_hybrid):.3f}',
         f'{np.max(noise_sensitivity_hybrid):.3f}'],
    ]
    
    table = ax3.table(cellText=summary_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2.0)
    
    # Style the header row
    for i in range(6):
        table[(0, i)].set_facecolor('#8b5cf6')
        table[(0, i)].set_text_props(fontweight='bold', color='white')
    
    ax3.set_title('Descriptive Statistics', pad=20, fontweight='bold', fontsize=12)
    
    # 1.4 Distribution Box Plots
    ax4 = fig.add_subplot(gs[1, 1])
    data_to_plot = [
        faithfulness_hybrid,
        context_precision_hybrid,
        context_recall_hybrid,
        noise_sensitivity_hybrid
    ]
    
    bp = ax4.boxplot(data_to_plot, labels=['Faith.', 'Prec.', 'Recall', 'Noise'], 
                     patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax4.set_ylabel('Score')
    ax4.set_title('Score Distributions')
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_ylim(-0.05, 1.05)

    pdf.savefig(fig)
    plt.close()
    
    # =========================================================================
    # PAGE 2: INDIVIDUAL METRICS BY QUESTION
    # =========================================================================
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    fig.suptitle('Hybrid RAG - Metric Performance by Question', fontsize=16, fontweight='bold')
    
    metrics_data = [
        ('Faithfulness (↑ Higher is Better)', faithfulness_hybrid, '#10b981'),
        ('Context Precision (↑ Higher is Better)', context_precision_hybrid, '#3b82f6'),
        ('Context Recall (↑ Higher is Better)', context_recall_hybrid, '#8b5cf6'),
        ('Noise Sensitivity (↓ Lower is Better)', noise_sensitivity_hybrid, '#ef4444')
    ]
    
    x = np.arange(len(questions))
    
    for idx, (title, data, color) in enumerate(metrics_data):
        ax = axes[idx]
        bars = ax.bar(x, data, color=color, alpha=0.8)
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(questions, fontsize=8, rotation=0)
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', alpha=0.3)
        
        # Add average line
        avg = calc_avg(data)
        ax.axhline(y=avg, color=color, linestyle='--', alpha=0.7, linewidth=2, 
                   label=f'Average: {avg:.3f}')
        ax.legend(loc='upper right')
        
        # Highlight best and worst
        min_idx = np.argmin(data)
        max_idx = np.argmax(data)
        bars[min_idx].set_edgecolor('red')
        bars[min_idx].set_linewidth(2)
        bars[max_idx].set_edgecolor('green')
        bars[max_idx].set_linewidth(2)
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()
    
    # =========================================================================
    # PAGE 3: CATEGORY ANALYSIS
    # =========================================================================
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Hybrid RAG - Performance by Question Category', fontsize=16, fontweight='bold')
    
    # Organize data by category
    category_metrics = {}
    for q, cat in question_categories.items():
        if cat not in category_metrics:
            category_metrics[cat] = {
                'faithfulness': [],
                'precision': [],
                'recall': [],
                'noise': []
            }
        i = questions.index(q)
        category_metrics[cat]['faithfulness'].append(faithfulness_hybrid[i])
        category_metrics[cat]['precision'].append(context_precision_hybrid[i])
        category_metrics[cat]['recall'].append(context_recall_hybrid[i])
        category_metrics[cat]['noise'].append(noise_sensitivity_hybrid[i])
    
    categories = list(category_metrics.keys())
    
    # Calculate averages per category
    faith_avg = [np.mean(category_metrics[c]['faithfulness']) for c in categories]
    prec_avg = [np.mean(category_metrics[c]['precision']) for c in categories]
    recall_avg = [np.mean(category_metrics[c]['recall']) for c in categories]
    noise_avg = [np.mean(category_metrics[c]['noise']) for c in categories]
    
    # Create GridSpec for layout
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # 3.1 Faithfulness by Category
    ax1 = fig.add_subplot(gs[0, 0])
    y = np.arange(len(categories))
    height = 0.6
    bars1 = ax1.barh(y, faith_avg, height, color='#10b981', alpha=0.8)
    ax1.set_xlabel('Average Score')
    ax1.set_title('Faithfulness by Category (↑ Higher is Better)')
    ax1.set_yticks(y)
    ax1.set_yticklabels(categories, fontsize=9)
    ax1.grid(axis='x', alpha=0.3)
    ax1.set_xlim(0, 1)
    
    # Add value labels
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=9)
    
    # 3.2 Context Precision by Category
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.barh(y, prec_avg, height, color='#3b82f6', alpha=0.8)
    ax2.set_xlabel('Average Score')
    ax2.set_title('Context Precision by Category (↑ Higher is Better)')
    ax2.set_yticks(y)
    ax2.set_yticklabels(categories, fontsize=9)
    ax2.grid(axis='x', alpha=0.3)
    ax2.set_xlim(0, 1)
    
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=9)
    
    # 3.3 Context Recall by Category
    ax3 = fig.add_subplot(gs[1, 0])
    bars3 = ax3.barh(y, recall_avg, height, color='#8b5cf6', alpha=0.8)
    ax3.set_xlabel('Average Score')
    ax3.set_title('Context Recall by Category (↑ Higher is Better)')
    ax3.set_yticks(y)
    ax3.set_yticklabels(categories, fontsize=9)
    ax3.grid(axis='x', alpha=0.3)
    ax3.set_xlim(0, 1)
    
    for i, bar in enumerate(bars3):
        width = bar.get_width()
        ax3.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=9)
    
    # 3.4 Noise Sensitivity by Category
    ax4 = fig.add_subplot(gs[1, 1])
    bars4 = ax4.barh(y, noise_avg, height, color='#ef4444', alpha=0.8)
    ax4.set_xlabel('Average Score')
    ax4.set_title('Noise Sensitivity by Category (↓ Lower is Better)', color='#dc2626')
    ax4.set_yticks(y)
    ax4.set_yticklabels(categories, fontsize=9)
    ax4.grid(axis='x', alpha=0.3)
    ax4.set_xlim(0, 1)
    
    for i, bar in enumerate(bars4):
        width = bar.get_width()
        ax4.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=9)
    
    # 3.5 Category Summary Table
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    category_summary = [
        ['Category', 'Questions', 'Faithfulness', 'Ctx Precision', 'Ctx Recall', 'Noise Sens.', 'Overall Score*']
    ]
    
    for i, cat in enumerate(categories):
        q_count = len([q for q, c in question_categories.items() if c == cat])
        overall = (faith_avg[i] + prec_avg[i] + recall_avg[i] - noise_avg[i]) / 3
        category_summary.append([
            cat,
            str(q_count),
            f'{faith_avg[i]:.3f}',
            f'{prec_avg[i]:.3f}',
            f'{recall_avg[i]:.3f}',
            f'{noise_avg[i]:.3f}',
            f'{overall:.3f}'
        ])
    
    table = ax5.table(cellText=category_summary, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)
    
    # Style header
    for i in range(7):
        table[(0, i)].set_facecolor('#8b5cf6')
        table[(0, i)].set_text_props(fontweight='bold', color='white')
    
    # Highlight best category in overall score
    overall_scores = [float(category_summary[i][6]) for i in range(1, len(category_summary))]
    best_cat_idx = overall_scores.index(max(overall_scores)) + 1
    for i in range(7):
        table[(best_cat_idx, i)].set_facecolor('#d4edda')
    
    ax5.set_title('Category Performance Summary\n*Overall = (Faith + Prec + Recall - Noise) / 3', 
                 pad=20, fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()
    
    # =========================================================================
    # PAGE 4: BEST AND WORST PERFORMING QUESTIONS
    # =========================================================================
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Hybrid RAG - Best & Worst Performing Questions', fontsize=16, fontweight='bold')
    
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
    
    # Calculate overall score for each question
    overall_scores = []
    for i in range(len(questions)):
        score = (faithfulness_hybrid[i] + context_precision_hybrid[i] + 
                context_recall_hybrid[i] - noise_sensitivity_hybrid[i]) / 3
        overall_scores.append(score)
    
    # Get sorted indices
    sorted_indices = np.argsort(overall_scores)
    
    # Best 5 questions
    ax1 = fig.add_subplot(gs[0, :])
    best_5_indices = sorted_indices[-5:][::-1]
    best_5_q = [questions[i] for i in best_5_indices]
    best_5_scores = [overall_scores[i] for i in best_5_indices]
    best_5_texts = [question_texts[questions[i]][:50] + '...' for i in best_5_indices]
    
    y_pos = np.arange(len(best_5_q))
    bars = ax1.barh(y_pos, best_5_scores, color='#10b981', alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"{q}\n{text}" for q, text in zip(best_5_q, best_5_texts)], fontsize=9)
    ax1.set_xlabel('Overall Score')
    ax1.set_title('Top 5 Best Performing Questions', fontsize=12, fontweight='bold', color='green')
    ax1.grid(axis='x', alpha=0.3)
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    # Worst 5 questions
    ax2 = fig.add_subplot(gs[1, :])
    worst_5_indices = sorted_indices[:5]
    worst_5_q = [questions[i] for i in worst_5_indices]
    worst_5_scores = [overall_scores[i] for i in worst_5_indices]
    worst_5_texts = [question_texts[questions[i]][:50] + '...' for i in worst_5_indices]
    
    y_pos = np.arange(len(worst_5_q))
    bars = ax2.barh(y_pos, worst_5_scores, color='#ef4444', alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"{q}\n{text}" for q, text in zip(worst_5_q, worst_5_texts)], fontsize=9)
    ax2.set_xlabel('Overall Score')
    ax2.set_title('Top 5 Worst Performing Questions (Need Improvement)', 
                 fontsize=12, fontweight='bold', color='red')
    ax2.grid(axis='x', alpha=0.3)
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

# =============================================================================
# PRINT SUMMARY STATISTICS
# =============================================================================
print("\n" + "="*60)
print("PDF generated successfully: hybrid_rag_analysis.pdf")
print("="*60)
print("\nHYBRID RAG SUMMARY STATISTICS:")
print("-"*60)
print(f"Faithfulness:       μ={calc_avg(faithfulness_hybrid):.3f}, σ={calc_std(faithfulness_hybrid):.3f}")
print(f"Context Precision:  μ={calc_avg(context_precision_hybrid):.3f}, σ={calc_std(context_precision_hybrid):.3f}")
print(f"Context Recall:     μ={calc_avg(context_recall_hybrid):.3f}, σ={calc_std(context_recall_hybrid):.3f}")
print(f"Noise Sensitivity:  μ={calc_avg(noise_sensitivity_hybrid):.3f}, σ={calc_std(noise_sensitivity_hybrid):.3f}")

# Calculate how many questions scored above average for each metric
print("\n" + "-"*60)
print("QUESTIONS ABOVE AVERAGE:")
print("-"*60)
for metric_name, data in [
    ('Faithfulness', faithfulness_hybrid),
    ('Context Precision', context_precision_hybrid),
    ('Context Recall', context_recall_hybrid),
    ('Noise Sensitivity (lower is better)', noise_sensitivity_hybrid)
]:
    avg = calc_avg(data)
    if 'lower is better' in metric_name:
        above = sum(1 for x in data if x < avg)
    else:
        above = sum(1 for x in data if x > avg)
    print(f"{metric_name}: {above}/24 questions")

print("\n" + "="*60)