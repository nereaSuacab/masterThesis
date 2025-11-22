import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# =============================================================================
# DATA
# =============================================================================
questions = ['q1','q2','q3','q4','q5','q6','q7','q8','q9','q10','q11','q12',
             'q13','q14','q15','q16','q17','q18','q19','q20','q21','q22','q23','q24']

faithfulness_dense = [0.2,1,0.2,0.8,0.5,0.75,1,0.5,0.75,1,0.5,0.67,1,0.2,0.6,0.33,0.714,0.25,0.4,1,0.714,0.125,1,0.5]
faithfulness_sparse = [0.8,0.6,0.4,0.67,0.75,1,1,1,0.6,0.67,0.67,1,0.5,0.5,0.75,0.67,0.33,1,0.75,0.5,0.75,0.67,0.5,0.83]

context_precision_dense = [0,0,1,1,1,1,1,1,1,1,0,0,1,0,0,0,0,1,1,1,0,0,0,0]
context_precision_sparse = [1,0,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1]

context_recall_dense = [0,0,1,0,1,1,0.5,0.67,0,0,0.67,0.67,0,0.67,0,1,1,1,0.22,0.5,1,0.5,0.5,0.5]
context_recall_sparse = [0,0.5,0.5,1,0.47,0.5,1,0.5,0,0.5,0.25,0.14,0,0.17,0.5,1,0,0.5,0.67,0.5,0,0,0.5,0]

# Original data from your table - please verify these values match your source
noise_sensitivity_dense = [0,0.67,0,0,0,0.25,0.67,0.5,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
noise_sensitivity_sparse = [0.6,0,1,0.33,0.5,0.5,0,0.67,0.2,0,0,0.83,0,0,1,0,0,0.5,0.75,0.33,0.25,0,0,0.67]

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
    'q1': "ISO/Technical", 'q2': "Application", 'q3': "Product+ISO", 'q4': "ISO/Technical",
    'q5': "ISO/Technical", 'q6': "Application", 'q7': "Application", 'q8': "Application",
    'q9': "Product Variant", 'q10': "Product Variant", 'q11': "Product Variant", 'q12': "Software",
    'q13': "Product+ISO", 'q14': "Product Specific", 'q15': "Feature", 'q16': "ISO/Technical",
    'q17': "Product Variant", 'q18': "Software", 'q19': "Product Variant", 'q20': "Application",
    'q21': "Use Case", 'q22': "Product Variant", 'q23': "Product Variant", 'q24': "Use Case"
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
    # PAGE 1: OVERVIEW
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('RAG Dense vs Sparse - Analysis Overview', fontsize=16, fontweight='bold')
    
    # 1.1 Average Metrics Bar Chart
    ax1 = axes[0, 0]
    metrics = ['Faithfulness', 'Context\nPrecision', 'Context\nRecall', 'Noise\nSensitivity']
    dense_avgs = [calc_avg(faithfulness_dense), calc_avg(context_precision_dense), 
                  calc_avg(context_recall_dense), calc_avg(noise_sensitivity_dense)]
    sparse_avgs = [calc_avg(faithfulness_sparse), calc_avg(context_precision_sparse),
                   calc_avg(context_recall_sparse), calc_avg(noise_sensitivity_sparse)]
    
    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax1.bar(x - width/2, dense_avgs, width, label='Dense', color='#3b82f6')
    bars2 = ax1.bar(x + width/2, sparse_avgs, width, label='Sparse', color='#22c55e')
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
    
    # 1.2 Radar Chart
    ax2 = plt.subplot(2, 2, 2, projection='polar')
    categories_radar = ['Faithfulness', 'Ctx Precision', 'Ctx Recall', 'Noise Sens.']
    N = len(categories_radar)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    dense_vals = dense_avgs + [dense_avgs[0]]
    sparse_vals = sparse_avgs + [sparse_avgs[0]]
    
    ax2.plot(angles, dense_vals, 'o-', linewidth=2, label='Dense', color='#3b82f6')
    ax2.fill(angles, dense_vals, alpha=0.25, color='#3b82f6')
    ax2.plot(angles, sparse_vals, 'o-', linewidth=2, label='Sparse', color='#22c55e')
    ax2.fill(angles, sparse_vals, alpha=0.25, color='#22c55e')
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories_radar, fontsize=9)
    ax2.set_title('Comparative Profile', pad=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    
    # 1.3 Win Count Pie
    ax3 = axes[1, 0]
    win_dense, win_sparse, tie = 0, 0, 0
    for i in range(len(questions)):
        d, s = get_totals(i)
        if d > s: win_dense += 1
        elif s > d: win_sparse += 1
        else: tie += 1
    
    colors = ['#3b82f6', '#9ca3af', '#22c55e']
    sizes = [win_dense, tie, win_sparse]
    labels = [f'Dense Wins\n({win_dense})', f'Ties\n({tie})', f'Sparse Wins\n({win_sparse})']
    ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Overall Winner Distribution')
    
    # 1.4 Statistics Table
    ax4 = axes[1, 1]
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
    
    plt.tight_layout()
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
        ax.bar(x - width/2, dense_data, width, label='Dense', color='#3b82f6', alpha=0.8)
        ax.bar(x + width/2, sparse_data, width, label='Sparse', color='#22c55e', alpha=0.8)
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(questions, fontsize=8)
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=calc_avg(dense_data), color='#1e40af', linestyle='--', alpha=0.7, linewidth=1)
        ax.axhline(y=calc_avg(sparse_data), color='#15803d', linestyle='--', alpha=0.7, linewidth=1)
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()
    
    # =========================================================================
    # PAGE 3: CATEGORY ANALYSIS
    # =========================================================================
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('RAG Dense vs Sparse - Performance by Question Category', fontsize=16, fontweight='bold')
    
    category_scores = {}
    for q, cat in question_categories.items():
        if cat not in category_scores:
            category_scores[cat] = {'dense': [], 'sparse': []}
        i = questions.index(q)
        d, s = get_totals(i)
        category_scores[cat]['dense'].append(d)
        category_scores[cat]['sparse'].append(s)
    
    categories = list(category_scores.keys())
    dense_cat_avgs = [np.mean(category_scores[c]['dense']) for c in categories]
    sparse_cat_avgs = [np.mean(category_scores[c]['sparse']) for c in categories]
    
    # 3.1 Category Bar Chart
    ax1 = axes[0]
    y = np.arange(len(categories))
    height = 0.35
    ax1.barh(y - height/2, dense_cat_avgs, height, label='Dense', color='#3b82f6')
    ax1.barh(y + height/2, sparse_cat_avgs, height, label='Sparse', color='#22c55e')
    ax1.set_xlabel('Composite Score')
    ax1.set_title('Average Score by Question Category')
    ax1.set_yticks(y)
    ax1.set_yticklabels(categories)
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)
    
    # 3.2 Winner by Category
    ax2 = axes[1]
    differences = [d - s for d, s in zip(dense_cat_avgs, sparse_cat_avgs)]
    colors = ['#3b82f6' if d > 0 else '#22c55e' for d in differences]
    
    bars = ax2.barh(categories, differences, color=colors)
    ax2.axvline(x=0, color='black', linewidth=0.5)
    ax2.set_xlabel('Score Difference (Dense - Sparse)')
    ax2.set_title('Winner by Category (Blue = Dense wins, Green = Sparse wins)')
    ax2.grid(axis='x', alpha=0.3)
    
    for bar, diff in zip(bars, differences):
        w = bar.get_width()
        ax2.annotate(f'{abs(diff):.2f}',
                    xy=(w, bar.get_y() + bar.get_height()/2),
                    xytext=(5 if w >= 0 else -5, 0),
                    textcoords="offset points",
                    ha='left' if w >= 0 else 'right', va='center', fontsize=9)
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()
    
    # =========================================================================
    # PAGE 4: CORRELATIONS & INSIGHTS
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('RAG Dense vs Sparse - Correlations & Insights', fontsize=16, fontweight='bold')
    
    # 4.1 Precision vs Faithfulness
    ax1 = axes[0, 0]
    ax1.scatter(context_precision_dense, faithfulness_dense, c='#3b82f6', label='Dense', alpha=0.7, s=100)
    ax1.scatter(context_precision_sparse, faithfulness_sparse, c='#22c55e', label='Sparse', alpha=0.7, s=100, marker='s')
    ax1.set_xlabel('Context Precision')
    ax1.set_ylabel('Faithfulness')
    ax1.set_title('Context Precision vs Faithfulness')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)
    
    # 4.2 Recall vs Noise
    ax2 = axes[0, 1]
    ax2.scatter(context_recall_dense, noise_sensitivity_dense, c='#3b82f6', label='Dense', alpha=0.7, s=100)
    ax2.scatter(context_recall_sparse, noise_sensitivity_sparse, c='#22c55e', label='Sparse', alpha=0.7, s=100, marker='s')
    ax2.set_xlabel('Context Recall')
    ax2.set_ylabel('Noise Sensitivity')
    ax2.set_title('Context Recall vs Noise Sensitivity')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 1.1)
    
    # 4.3 Winner by Question
    ax3 = axes[1, 0]
    dense_totals = [get_totals(i)[0] for i in range(len(questions))]
    sparse_totals = [get_totals(i)[1] for i in range(len(questions))]
    diffs = [d - s for d, s in zip(dense_totals, sparse_totals)]
    colors = ['#3b82f6' if d > 0 else '#22c55e' for d in diffs]
    
    ax3.bar(questions, diffs, color=colors)
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_xlabel('Question')
    ax3.set_ylabel('Score Difference (Dense - Sparse)')
    ax3.set_title('Winner by Question')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4.4 Box Plot
    ax4 = axes[1, 1]
    bp_data = [
        faithfulness_dense, faithfulness_sparse,
        context_precision_dense, context_precision_sparse,
        context_recall_dense, context_recall_sparse,
        noise_sensitivity_dense, noise_sensitivity_sparse
    ]
    positions = [1, 1.6, 3, 3.6, 5, 5.6, 7, 7.6]
    bp = ax4.boxplot(bp_data, positions=positions, widths=0.5, patch_artist=True)
    colors_bp = ['#3b82f6', '#22c55e'] * 4
    for patch, color in zip(bp['boxes'], colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax4.set_xticks([1.3, 3.3, 5.3, 7.3])
    ax4.set_xticklabels(['Faithfulness', 'Ctx Precision', 'Ctx Recall', 'Noise Sens.'])
    ax4.set_ylabel('Score')
    ax4.set_title('Score Distribution (Blue=Dense, Green=Sparse)')
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()
    
    # =========================================================================
    # PAGE 5: DETAILED RESULTS TABLE
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 12))
    fig.suptitle('RAG Dense vs Sparse - Detailed Results Table', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    table_data = [['Q', 'Category', 'Faith D/S', 'Prec D/S', 'Rec D/S', 'Noise D/S', 'Total D/S', 'Win']]
    
    for i, q in enumerate(questions):
        d_tot, s_tot = get_totals(i)
        winner = 'D' if d_tot > s_tot else 'S' if s_tot > d_tot else '='
        table_data.append([
            q, question_categories[q][:12],
            f'{faithfulness_dense[i]:.2f}/{faithfulness_sparse[i]:.2f}',
            f'{context_precision_dense[i]:.0f}/{context_precision_sparse[i]:.0f}',
            f'{context_recall_dense[i]:.2f}/{context_recall_sparse[i]:.2f}',
            f'{noise_sensitivity_dense[i]:.2f}/{noise_sensitivity_sparse[i]:.2f}',
            f'{d_tot:.2f}/{s_tot:.2f}', winner
        ])
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.2)
    
    for i in range(8):
        table[(0, i)].set_facecolor('#374151')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    for i in range(1, len(table_data)):
        winner = table_data[i][-1]
        if winner == 'D':
            table[(i, 7)].set_facecolor('#dbeafe')
        elif winner == 'S':
            table[(i, 7)].set_facecolor('#dcfce7')
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()
    
    # =========================================================================
    # PAGE 6: QUESTION DETAILS
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 14))
    fig.suptitle('RAG Dense vs Sparse - Question Details', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    table_data = [['Q', 'Question Text', 'Category']]
    for q in questions:
        table_data.append([q, question_texts[q][:50], question_categories[q]])
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.1)
    
    for i in range(3):
        table[(0, i)].set_facecolor('#374151')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

print("✅ PDF generated: rag_dense_vs_sparse_analysis.pdf")
print(f"\nSummary:")
print(f"  - Dense wins: {win_dense} questions")
print(f"  - Sparse wins: {win_sparse} questions")
print(f"  - Ties: {tie} questions")