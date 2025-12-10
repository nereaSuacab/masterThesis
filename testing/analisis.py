import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# =============================================================================
# DATA
# =============================================================================
questions = ['q1','q2','q3','q4','q5','q6','q7','q8','q9','q10','q11','q12',
             'q13','q14','q15','q16','q17','q18','q19','q20','q21','q22','q23','q24']

########################## ROUND 3 #############################

# faithfulness_dense = [0.2, 1, 0.2, 0.8, 0.5, 0.75, 1, 0.5, 0.75, 1, 0.5, 0.67, 1, 0.2, 0.6, 0.33, 0.714, 0.25, 0.4, 1, 0.714, 0.125, 1, 0.5]
# faithfulness_sparse = [0.8, 0.6, 0.4, 0.67, 0.75, 1, 1, 1, 0.6, 0.67, 0.67, 1, 0.5, 0.5, 0.75, 0.67, 0.33, 1, 0.75, 0.5, 0.75, 0.67, 0.5, 0.83]

# context_precision_dense = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
# context_precision_sparse = [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0.5, 0]

# context_recall_dense = [0, 0, 1, 0, 1, 1, 0.5, 0.67, 0, 0, 0, 0.67, 0.67, 0, 0.67, 0, 1, 1, 1, 0.22, 0.5, 1, 0.5, 0.5]
# context_recall_sparse = [0, 0.5, 0.5, 1, 0.47, 0.5, 1, 0.5, 0, 0.5, 0.25, 0.14, 0, 0.17, 0.5, 1, 0, 0.5, 0.67, 0.5, 0, 0, 0.5, 0]

# noise_sensitivity_dense = [0, 0.67, 0, 0, 0, 0.25, 0.67, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# noise_sensitivity_sparse = [0.6, 0, 1, 0.33, 0.5, 0.5, 0, 0.67, 0.2, 0, 0, 0.83, 0, 0, 1, 0, 0, 0.5, 0.75, 0.33, 0.25, 0, 0, 0.67]

########################## ROUND 4 #############################

# faithfulness_dense = [0, 1, 0.67, 0.5, 0.67, 0.75, 1, 0.67, 0.67, 1, 0.5, 0.67, 0.8, 1, 0.4, 0.75, 0.5, 0.25, 0.5, 1, 0.5, 0.33, 0.78, 0.8]
# faithfulness_sparse = [0, 1, 0, 0, 0, 0, 0, 0.25, 0.5, 1, 0.67, 0, 0.25, 0, 0.75, 0.75, 1, 0, 0.25, 0, 1, 0.571, 0, 0.22]

# context_precision_dense = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0.5, 0.5, 1, 1, 0, 0, 1, 1, 1]
# context_precision_sparse = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# context_recall_dense = [0, 0, 0.5, 0, 1, 1, 0.5, 0.67, 0.35, 0, 0, 0.67, 0.67, 0, 0.67, 0, 0, 1, 1, 0.125, 0, 0.57, 0.5, 0.57]
# context_recall_sparse = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]

# noise_sensitivity_dense = [0, 0.25, 0.33, 0, 0, 0.25, 0, 0.33, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0.25, 0, 0.167, 0, 0, 0.8]
# noise_sensitivity_sparse = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

########################## ROUND 5 #############################

# faithfulness_dense = [0.25, 1, 0, 0.33, 0.5, 0.5, 1, 0.67, 0.67, 0, 0.33, 0.6, 0.67, 0.67, 0.5, 0.5, 0.25, 0.2, 0.75, 1, 0.25, 0.75, 0, 0.6]
# faithfulness_sparse = [0.75, 0.67, 0.25, 0.67, 1, 1, 0.5, 1, 0.67, 1, 0.4, 1, 1, 0.5, 0.5, 0.5, 0.5, 1, 0.33, 1, 1, 0.33, 0.5, 1]

# context_precision_dense = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
# context_precision_sparse = [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0]

# context_recall_dense = [0, 0, 1, 0, 1, 1, 0.5, 0.67, 0, 0, 0, 0.67, 0, 0, 0.67, 0, 1, 1, 1, 0.125, 0.5, 1, 0, 0.5]
# context_recall_sparse = [1, 0.5, 0.33, 1, 0.14, 0.5, 0.4, 1, 0.5, 0, 0.25, 0.5, 0.077, 0.16, 0.5, 1, 0, 1, 1, 0.5, 0.5, 0, 0.5, 0]

# noise_sensitivity_dense = [0, 0.5, 0, 0, 0, 0.16, 0, 0.33, 0, 0, 0, 0.2, 0, 0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0.2]
# noise_sensitivity_sparse = [0.5, 0, 0, 0.33, 0.67, 0.2, 0.5, 0.5, 0, 0, 0, 1, 0, 0, 0.75, 0, 0, 1, 0.67, 0.5, 0.33, 0, 0, 0.25]


########################## ROUND 6 #############################

# faithfulness_dense = [0.67, 1.0, 0.5, 0.5, 0.5, 0.5, 1.0, 0.67, 0.4, 0.33, 0.33, 0.6, 0.67, 0.67, 0.67, 0.67, 0.6, 1.0, 0.4, 1.0, 1.0, 0.25, 0.0, 0.6]
# context_precision_dense = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# context_recall_dense = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.67, 0.0, 0.0, 0.0, 0.67, 0.67, 0.0, 0.67, 0.0, 1.0, 1.0, 1.0, 0.12, 0.5, 1.0, 0.67, 0.5]
# noise_sensitivity_dense = [0.0, 0.5, 0.0, 0.0, 0.0, 0.17, 0.0, 0.67, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.67, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.4]

# faithfulness_sparse = [0.67, 1.0, 0.25, 0.67, 1.0, 1.0, 1.0, 0.67, 0.75, 0.75, 1.0, 1.0, 0.67, 0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 1.0, 0.83, 0.8, 0.0, 1.0]
# context_precision_sparse = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]
# context_recall_sparse = [1.0, 0.5, 0.33, 1.0, 0.2, 0.5, 0.17, 0.71, 0.5, 0.5, 0.83, 0.5, 0.07, 0.17, 0.5, 1.0, 0.0, 1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0]
# noise_sensitivity_sparse = [0.17, 0.0, 0.0, 0.67, 0.0, 0.75, 0.33, 0.33, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.25, 0.5, 0.17, 0.0, 0.0, 0.6]

########################## k = 3 ###########################
faithfulness_dense = [0.25, 1.0, 0.0, 0.67, 0.0, 0.67, 0.6, 0.5, 0.2, 0.0, 0.67, 0.5, 0.67, 1.0, 0.5, 1.0, 1.0, 0.33, 0.2, 0.75, 1.0, 0.67, 0.5, 0.5]
context_precision_dense = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
context_recall_dense = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
noise_sensitivity_dense = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]

faithfulness_sparse = [1.0, 1.0, 1.0, 1.0, 0.5, 0.67, 0.67, 0.5, 0.6, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 0.5, 0.67, 1.0, 0.67, 1.0, 1.0, 0.5, 0.0, 1.0]
context_precision_sparse = [1.0, 0.0, 1.0, 1.0, 0.58, 0.58, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
context_recall_sparse = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
noise_sensitivity_sparse = [1.0, 0.0, 1.0, 1.0, 0.5, 0.5, 1.0, 0.5, 0.67, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25]


# Verify all arrays have 24 elements
assert len(questions) == 24, f"questions has {len(questions)} elements"
assert len(faithfulness_dense) == 24, f"faithfulness_dense has {len(faithfulness_dense)} elements"
assert len(faithfulness_sparse) == 24, f"faithfulness_sparse has {len(faithfulness_sparse)} elements"
assert len(context_precision_dense) == 24, f"context_precision_dense has {len(context_precision_dense)} elements"
assert len(context_precision_sparse) == 24, f"context_precision_sparse has {len(context_precision_sparse)} elements"
assert len(context_recall_dense) == 24, f"context_recall_dense has {len(context_recall_dense)} elements"
assert len(context_recall_sparse) == 24, f"context_recall_sparse has {len(context_recall_sparse)} elements"
assert len(noise_sensitivity_dense) == 24, f"noise_sensitivity_dense has {len(noise_sensitivity_dense)} elements"
assert len(noise_sensitivity_sparse) == 24, f"noise_sensitivity_sparse has {len(noise_sensitivity_sparse)} elements"

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

def get_totals(i):
    dense = faithfulness_dense[i] + context_precision_dense[i] + context_recall_dense[i] - noise_sensitivity_dense[i]
    sparse = faithfulness_sparse[i] + context_precision_sparse[i] + context_recall_sparse[i] - noise_sensitivity_sparse[i]
    return dense, sparse

# =============================================================================
# CREATE PDF
# =============================================================================
with PdfPages('rag_dense_vs_sparse_analysis.pdf') as pdf:
    
    # =========================================================================
    # PAGE 1: OVERVIEW (FIXED)
    # =========================================================================
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('RAG Dense vs Sparse - Analysis Overview', fontsize=16, fontweight='bold')

    # Create a GridSpec for better control
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1.1 Average Metrics Bar Chart
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['Faithfulness', 'Context\nPrecision', 'Context\nRecall', 'Noise\nSensitivity']
    dense_avgs = [calc_avg(faithfulness_dense), calc_avg(context_precision_dense), 
                calc_avg(context_recall_dense), calc_avg(noise_sensitivity_dense)]
    sparse_avgs = [calc_avg(faithfulness_sparse), calc_avg(context_precision_sparse),
                calc_avg(context_recall_sparse), calc_avg(noise_sensitivity_sparse)]

    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax1.bar(x - width/2, dense_avgs, width, label='Dense', color='#12365D')
    bars2 = ax1.bar(x + width/2, sparse_avgs, width, label='Sparse', color='#A3D8D5')
    ax1.set_ylabel('Score')
    ax1.set_title('Average Metrics Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3)

    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax1.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

    # 1.2 Radar Chart (FIXED - using add_subplot with polar projection)
    ax2 = fig.add_subplot(gs[0, 1], projection='polar')
    categories_radar = ['Faithfulness', 'Ctx Precision', 'Ctx Recall', 'Noise Sens.']
    N = len(categories_radar)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    dense_vals = dense_avgs + [dense_avgs[0]]
    sparse_vals = sparse_avgs + [sparse_avgs[0]]

    ax2.plot(angles, dense_vals, 'o-', linewidth=2, label='Dense', color='#12365D')
    ax2.fill(angles, dense_vals, alpha=0.25, color='#12365D')
    ax2.plot(angles, sparse_vals, 'o-', linewidth=2, label='Sparse', color='#A3D8D5')
    ax2.fill(angles, sparse_vals, alpha=0.25, color='#A3D8D5')
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories_radar, fontsize=9)
    ax2.set_ylim(0, 1)
    ax2.set_title('Comparative Profile', pad=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    
    # 1.4 Statistics Table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    table_data = [
        ['Metric', 'Dense μ', 'Dense σ', 'Sparse μ', 'Sparse σ', 'Δ (D-S)'],
        ['Faithfulness', f'{calc_avg(faithfulness_dense):.3f}', f'{calc_std(faithfulness_dense):.3f}',
        f'{calc_avg(faithfulness_sparse):.3f}', f'{calc_std(faithfulness_sparse):.3f}',
        f'{calc_avg(faithfulness_dense)-calc_avg(faithfulness_sparse):.3f}'],
        ['Ctx Precision', f'{calc_avg(context_precision_dense):.3f}', f'{calc_std(context_precision_dense):.3f}',
        f'{calc_avg(context_precision_sparse):.3f}', f'{calc_std(context_precision_sparse):.3f}',
        f'{calc_avg(context_precision_dense)-calc_avg(context_precision_sparse):.3f}'],
        ['Ctx Recall', f'{calc_avg(context_recall_dense):.3f}', f'{calc_std(context_recall_dense):.3f}',
        f'{calc_avg(context_recall_sparse):.3f}', f'{calc_std(context_recall_sparse):.3f}',
        f'{calc_avg(context_recall_dense)-calc_avg(context_recall_sparse):.3f}'],
        ['Noise Sens.', f'{calc_avg(noise_sensitivity_dense):.3f}', f'{calc_std(noise_sensitivity_dense):.3f}',
        f'{calc_avg(noise_sensitivity_sparse):.3f}', f'{calc_std(noise_sensitivity_sparse):.3f}',
        f'{calc_avg(noise_sensitivity_dense)-calc_avg(noise_sensitivity_sparse):.3f}'],
    ]
    table = ax4.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    for i in range(6):
        table[(0, i)].set_facecolor('#e5e7eb')
        table[(0, i)].set_text_props(fontweight='bold')
    ax4.set_title('Descriptive Statistics', pad=20)

    pdf.savefig(fig)
    plt.close()
    
    # =========================================================================
    # PAGE 2: INDIVIDUAL METRICS
    # =========================================================================
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    fig.suptitle('RAG Dense vs Sparse - Metric Breakdown by Question', fontsize=16, fontweight='bold')
    
    metrics_data = [
        ('Faithfulness', faithfulness_dense, faithfulness_sparse),
        ('Context Precision', context_precision_dense, context_precision_sparse),
        ('Context Recall', context_recall_dense, context_recall_sparse),
        ('Noise Sensitivity', noise_sensitivity_dense, noise_sensitivity_sparse)
    ]
    
    x = np.arange(len(questions))
    width = 0.35
    
    for idx, (title, dense_data, sparse_data) in enumerate(metrics_data):
        ax = axes[idx]
        ax.bar(x - width/2, dense_data, width, label='Dense', color='#12365D', alpha=0.8)
        ax.bar(x + width/2, sparse_data, width, label='Sparse', color='#A3D8D5', alpha=0.8)
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(questions, fontsize=8)
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=calc_avg(dense_data), color='#12365D', linestyle='--', alpha=0.7, linewidth=1)
        ax.axhline(y=calc_avg(sparse_data), color='#A3D8D5', linestyle='--', alpha=0.7, linewidth=1)
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()
    

# =========================================================================
    # PAGE 3: CATEGORY ANALYSIS (FIXED)
    # =========================================================================
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('RAG Dense vs Sparse - Performance by Question Category', fontsize=16, fontweight='bold')
    
    # Organize data by category
    category_metrics = {}
    for q, cat in question_categories.items():
        if cat not in category_metrics:
            category_metrics[cat] = {
                'faithfulness_dense': [], 'faithfulness_sparse': [],
                'precision_dense': [], 'precision_sparse': [],
                'recall_dense': [], 'recall_sparse': [],
                'noise_dense': [], 'noise_sparse': []
            }
        i = questions.index(q)
        category_metrics[cat]['faithfulness_dense'].append(faithfulness_dense[i])
        category_metrics[cat]['faithfulness_sparse'].append(faithfulness_sparse[i])
        category_metrics[cat]['precision_dense'].append(context_precision_dense[i])
        category_metrics[cat]['precision_sparse'].append(context_precision_sparse[i])
        category_metrics[cat]['recall_dense'].append(context_recall_dense[i])
        category_metrics[cat]['recall_sparse'].append(context_recall_sparse[i])
        category_metrics[cat]['noise_dense'].append(noise_sensitivity_dense[i])
        category_metrics[cat]['noise_sparse'].append(noise_sensitivity_sparse[i])
    
    categories = list(category_metrics.keys())
    
    # Calculate averages per category
    faith_dense_avg = [np.mean(category_metrics[c]['faithfulness_dense']) for c in categories]
    faith_sparse_avg = [np.mean(category_metrics[c]['faithfulness_sparse']) for c in categories]
    prec_dense_avg = [np.mean(category_metrics[c]['precision_dense']) for c in categories]
    prec_sparse_avg = [np.mean(category_metrics[c]['precision_sparse']) for c in categories]
    recall_dense_avg = [np.mean(category_metrics[c]['recall_dense']) for c in categories]
    recall_sparse_avg = [np.mean(category_metrics[c]['recall_sparse']) for c in categories]
    noise_dense_avg = [np.mean(category_metrics[c]['noise_dense']) for c in categories]
    noise_sparse_avg = [np.mean(category_metrics[c]['noise_sparse']) for c in categories]
    
    # Create GridSpec for layout
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # =========================================================================
    # OPCIÓN 1: Individual Metrics by Category (4 subplots)
    # =========================================================================
    
    # 3.1 Faithfulness by Category
    ax1 = fig.add_subplot(gs[0, 0])
    y = np.arange(len(categories))
    height = 0.35
    ax1.barh(y - height/2, faith_dense_avg, height, label='Dense', color='#12365D', alpha=0.8)
    ax1.barh(y + height/2, faith_sparse_avg, height, label='Sparse', color='#A3D8D5', alpha=0.8)
    ax1.set_xlabel('Average Score')
    ax1.set_title('Faithfulness by Category (↑ Higher is Better)')
    ax1.set_yticks(y)
    ax1.set_yticklabels(categories, fontsize=9)
    ax1.legend(loc='lower right')
    ax1.grid(axis='x', alpha=0.3)
    ax1.set_xlim(0, 1)
    
    # 3.2 Context Precision by Category
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.barh(y - height/2, prec_dense_avg, height, label='Dense', color='#12365D', alpha=0.8)
    ax2.barh(y + height/2, prec_sparse_avg, height, label='Sparse', color='#A3D8D5', alpha=0.8)
    ax2.set_xlabel('Average Score')
    ax2.set_title('Context Precision by Category (↑ Higher is Better)')
    ax2.set_yticks(y)
    ax2.set_yticklabels(categories, fontsize=9)
    ax2.legend(loc='lower right')
    ax2.grid(axis='x', alpha=0.3)
    ax2.set_xlim(0, 1)
    
    # 3.3 Context Recall by Category
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.barh(y - height/2, recall_dense_avg, height, label='Dense', color='#12365D', alpha=0.8)
    ax3.barh(y + height/2, recall_sparse_avg, height, label='Sparse', color='#A3D8D5', alpha=0.8)
    ax3.set_xlabel('Average Score')
    ax3.set_title('Context Recall by Category (↑ Higher is Better)')
    ax3.set_yticks(y)
    ax3.set_yticklabels(categories, fontsize=9)
    ax3.legend(loc='lower right')
    ax3.grid(axis='x', alpha=0.3)
    ax3.set_xlim(0, 1)
    
    # 3.4 Noise Sensitivity by Category
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.barh(y - height/2, noise_dense_avg, height, label='Dense', color='#12365D', alpha=0.8)
    ax4.barh(y + height/2, noise_sparse_avg, height, label='Sparse', color='#A3D8D5', alpha=0.8)
    ax4.set_xlabel('Average Score')
    ax4.set_title('Noise Sensitivity by Category (↓ Lower is Better)', color='#dc2626')
    ax4.set_yticks(y)
    ax4.set_yticklabels(categories, fontsize=9)
    ax4.legend(loc='upper right')
    ax4.grid(axis='x', alpha=0.3)
    ax4.set_xlim(0, 1)
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

# =============================================================================
# CALCULATE FINAL STATISTICS (for summary at end)
# =============================================================================
win_dense, win_sparse, tie = 0, 0, 0
for i in range(len(questions)):
    wins_dense = 0
    wins_sparse = 0
    
    # Faithfulness (higher is better)
    if faithfulness_dense[i] > faithfulness_sparse[i]:
        wins_dense += 1
    elif faithfulness_dense[i] < faithfulness_sparse[i]:
        wins_sparse += 1
    
    # Context Precision (higher is better)
    if context_precision_dense[i] > context_precision_sparse[i]:
        wins_dense += 1
    elif context_precision_dense[i] < context_precision_sparse[i]:
        wins_sparse += 1
    
    # Context Recall (higher is better)
    if context_recall_dense[i] > context_recall_sparse[i]:
        wins_dense += 1
    elif context_recall_dense[i] < context_recall_sparse[i]:
        wins_sparse += 1
    
    # Noise Sensitivity (LOWER is better)
    if noise_sensitivity_dense[i] < noise_sensitivity_sparse[i]:
        wins_dense += 1
    elif noise_sensitivity_dense[i] > noise_sensitivity_sparse[i]:
        wins_sparse += 1
    
    # Overall winner for this question
    if wins_dense > wins_sparse:
        win_dense += 1
    elif wins_sparse > wins_dense:
        win_sparse += 1
    else:
        tie += 1
    
