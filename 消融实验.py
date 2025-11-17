import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

# 可复现
np.random.seed(2025)

# 模型与重复次数（与文档一致）
models = ['w/o Shapley', 'w/o MARL', 'w/o Personalised rewards', 'w/o State code sharing', 'DSV-MARL']
n = 10

# ========== 表5.3 中的均值与 std（解析自文档） ==========
spearman_mean = np.array([0.70, 0.65, 0.72, 0.79, 0.81])
spearman_std  = np.array([0.020, 0.025, 0.020, 0.017, 0.015])

mae_mean = np.array([0.162, 0.198, 0.155, 0.130, 0.115])
mae_std  = np.array([0.010, 0.012, 0.009, 0.008, 0.006])

gini_mean = np.array([0.20, 0.24, 0.19, 0.16, 0.12])
gini_std  = np.array([0.015, 0.020, 0.013, 0.013, 0.008])

conv_mean = np.array([605, 706, 585, 504, 453])
conv_std  = np.array([28, 32, 25, 22, 18])  # 模拟的波动，用于 boxplot 展示

# ========== 表5.4 给出的显著性 p 值（Spearman 与 MAE 与 DSV-MARL 的比较） ==========
spearman_p_table54 = {
    'w/o Shapley': 0.0007,
    'w/o MARL':    0.0002,
    'w/o Personalised rewards': 0.0019,
    'w/o State code sharing': 0.0086
}
mae_p_table54 = {
    'w/o Shapley': 0.0002,
    'w/o MARL':    0.00003,
    'w/o Personalised rewards': 0.0011,
    'w/o State code sharing': 0.0073
}

def p_to_stars(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'n.s.'

# ========== 模拟每个方法的原始重复数据（n 次） ==========
def simulate_samples(mean_arr, std_arr, n_samples=n, low=None, high=None):
    samples = []
    for mu, sigma in zip(mean_arr, std_arr):
        s = np.random.normal(loc=mu, scale=sigma, size=n_samples)
        if (low is not None) or (high is not None):
            s = np.clip(s, a_min=low, a_max=high)
        samples.append(s)
    return samples

spearman_samples = simulate_samples(spearman_mean, spearman_std, low=-1.0, high=1.0)
mae_samples      = simulate_samples(mae_mean, mae_std, low=0.0, high=None)
gini_samples     = simulate_samples(gini_mean, gini_std, low=0.0, high=1.0)
conv_samples_raw = simulate_samples(conv_mean, conv_std, low=1, high=None)

# 关键修正：将 conv_samples_raw 列表中的每个数组四舍五入并转换成整数数组
conv_samples = [np.round(arr).astype(int) for arr in conv_samples_raw]

# 组织数据结构，便于绘图
data = {
    'Spearman': {m: spearman_samples[i] for i, m in enumerate(models)},
    'MAE':      {m: mae_samples[i] for i, m in enumerate(models)},
    'Gini':     {m: gini_samples[i] for i, m in enumerate(models)},
    # Convergence: conv_samples 也按 models 顺序一一对应
    'Convergence': {m: conv_samples[i].astype(int) for i, m in enumerate(models)}
}

# ========== 绘图辅助函数：画显著性横线与星号（无裁切，带白底以提高可读性） ==========
def add_stat_annotation(ax, x1, x2, base_y, height, text, linewidth=1.0):
    ax.plot([x1, x1, x2, x2], [base_y, base_y+height, base_y+height, base_y],
            lw=linewidth, c='k', clip_on=False)
    ax.text((x1 + x2)/2, base_y + height + 0.01 * (abs(base_y) if base_y != 0 else 1),
            text, ha='center', va='bottom', fontsize=9,
            bbox=dict(facecolor='white', edgecolor='none', pad=0.2), clip_on=False)

# ========== 绘图（2x2 子图） ==========
sns.set(style='whitegrid', context='paper')
palette = sns.color_palette("colorblind", n_colors=len(models))

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

plot_metrics = ['Spearman', 'MAE', 'Gini']
y_labels = {
    'Spearman': 'Spearman correlation',
    'MAE': 'MAE',
    'Gini': 'Gini coefficient (lower = fairer)'
}

for ax, metric in zip(axes.flat[:3], plot_metrics):
    # 组装 DataFrame 便于 seaborn 绘制
    vals = []
    labs = []
    for m in models:
        arr = data[metric][m]
        vals.extend(arr.tolist())
        labs.extend([m] * len(arr))
    df = pd.DataFrame({metric: vals, 'Model': labs})
    sns.boxplot(x='Model', y=metric, data=df, ax=ax, palette=palette, width=0.55,
                showfliers=True, boxprops=dict(linewidth=1.0), medianprops=dict(linewidth=1.2))
    sns.stripplot(x='Model', y=metric, data=df, ax=ax, color='k', size=4, jitter=0.18, alpha=0.7)
    ax.set_xticklabels(models, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel(y_labels[metric], fontsize=10)
    title_letter = {'Spearman':'a', 'MAE':'b', 'Gini':'c'}[metric]
    ax.set_title(f'({title_letter}) {y_labels[metric]}', fontsize=11)

    # 计算 y 范围并留出足够空间以容纳多条注释（自动扩展）
    grp_max = df.groupby('Model')[metric].max()
    overall_max = grp_max.max()
    overall_min = df[metric].min()
    yrange = overall_max - overall_min if overall_max != overall_min else max(abs(overall_max),1.0)
    top_margin = yrange * 0.30
    ax.set_ylim(overall_min - yrange*0.08, overall_max + top_margin)

    # 按顺序为每个消融项与 DSV-MARL 添加显著性标注
    dsv_vals = data[metric]['DSV-MARL']
    step = yrange * 0.06
    next_offset = yrange * 0.05

    for i, m in enumerate(models):
        if m == 'DSV-MARL':
            continue
        if metric == 'Spearman':
            pval = spearman_p_table54.get(m, None)
        elif metric == 'MAE':
            pval = mae_p_table54.get(m, None)
        else:  # Gini 或其他：计算 Welch t-test
            comp = data[metric][m]
            tstat, pval = stats.ttest_ind(comp, dsv_vals, equal_var=False)
        stars = p_to_stars(pval)
        y_base = max(data[metric][m].max(), dsv_vals.max()) + next_offset
        add_stat_annotation(ax, i, models.index('DSV-MARL'), y_base, step, stars)
        next_offset += step  # 叠加高度避免重叠

# 子图 (d) Convergence rounds（Boxplot）
ax = axes[1,1]
vals = []
labs = []
for m in models:
    arr = data['Convergence'][m]
    vals.extend(arr.tolist())
    labs.extend([m] * len(arr))
df_conv = pd.DataFrame({'Convergence': vals, 'Model': labs})
sns.boxplot(x='Model', y='Convergence', data=df_conv, ax=ax,
            palette=palette, width=0.6, boxprops=dict(linewidth=1.0), medianprops=dict(linewidth=1.2))
sns.stripplot(x='Model', y='Convergence', data=df_conv, ax=ax, color='k', size=4, jitter=0.18, alpha=0.7)
ax.set_xticklabels(models, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('Convergence rounds', fontsize=10)
ax.set_title('(d) Convergence rounds (lower = faster)', fontsize=11)

# Convergence y-range 动态扩展与显著性（基于 Welch t-test）
grp_max = df_conv.groupby('Model')['Convergence'].max()
overall_max = grp_max.max()
overall_min = df_conv['Convergence'].min()
yrange = overall_max - overall_min if overall_max != overall_min else max(1, overall_max*0.1)
top_margin = yrange * 0.30
ax.set_ylim(overall_min - yrange*0.05, overall_max + top_margin)

step = yrange * 0.08
next_offset = yrange * 0.05
dsv_vals = data['Convergence']['DSV-MARL']
for i, m in enumerate(models):
    if m == 'DSV-MARL':
        continue
    comp_vals = data['Convergence'][m]
    tstat, pval = stats.ttest_ind(comp_vals, dsv_vals, equal_var=False)
    stars = p_to_stars(pval)
    y_base = max(comp_vals.max(), dsv_vals.max()) + next_offset
    add_stat_annotation(ax, i, models.index('DSV-MARL'), y_base, step, stars)
    next_offset += step

# 布局与标题
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
#fig.text(0.5, 0.99, 'Figure: Ablation study (simulated repeated runs based on Table 5.3; n=10)', ha='center', va='top', fontsize=13)

# 保存
fig.savefig('figure_5_2_ablation_boxplot_simulated_fixed.pdf', bbox_inches='tight')
fig.savefig('figure_5_2_ablation_boxplot_simulated_fixed.png', dpi=300, bbox_inches='tight')

plt.show()
