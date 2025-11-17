import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

np.random.seed(42)

# 模型与重复次数
models = ['Static Shapley', 'IQL', 'VDN', 'COMA', 'DSV-MARL']
n = 10

# 表5.1 给出的均值与 std（用于模拟）
spearman_mean = np.array([0.78, 0.80, 0.82, 0.83, 0.87])
spearman_std  = np.array([0.03, 0.025, 0.028, 0.03, 0.02])

mae_mean = np.array([0.072, 0.066, 0.059, 0.055, 0.045])
mae_std  = np.array([0.008, 0.007, 0.006, 0.005, 0.006])

gini_mean = np.array([0.21, 0.18, 0.16, 0.15, 0.12])
gini_std  = np.array([0.02, 0.015, 0.012, 0.01, 0.01])

conv_models = ['IQL', 'VDN', 'COMA', 'DSV-MARL']
conv_mean = np.array([688, 607, 556, 425])
conv_std  = np.array([50, 40, 35, 30])

def simulate_vals(mean_arr, std_arr, low=None, high=None):
    vals = []
    for mu, sigma in zip(mean_arr, std_arr):
        s = np.random.normal(loc=mu, scale=sigma, size=n)
        if low is not None or high is not None:
            s = np.clip(s, a_min=low, a_max=high)
        vals.append(s)
    return vals

spearman_samples = simulate_vals(spearman_mean, spearman_std, low=-1.0, high=1.0)
mae_samples = simulate_vals(mae_mean, mae_std, low=0.0, high=None)
gini_samples = simulate_vals(gini_mean, gini_std, low=0.0, high=1.0)
conv_samples = simulate_vals(conv_mean, conv_std, low=1, high=None)

data = {
    'Spearman': {m: spearman_samples[i] for i, m in enumerate(models)},
    'MAE':      {m: mae_samples[i] for i, m in enumerate(models)},
    'Gini':     {m: gini_samples[i] for i, m in enumerate(models)},
    'Convergence': {m: np.round(conv_samples[i_con]).astype(int)
                    for i_con, m in enumerate(conv_models)}
}

def p_to_stars(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'n.s.'

def add_stat_annotation(ax, x1, x2, base_y, height, text, linewidth=1.0):
    # 画横线并在上方放置星号；使用 clip_on=False 并给文字白色背景
    ax.plot([x1, x1, x2, x2], [base_y, base_y+height, base_y+height, base_y],
            lw=linewidth, c='k', clip_on=False)
    ax.text((x1+x2)/2, base_y+height+0.005*(abs(base_y) if base_y!=0 else 1),
            text, ha='center', va='bottom', fontsize=9,
            bbox=dict(facecolor='white', edgecolor='none', pad=0.2), clip_on=False)

sns.set(style='whitegrid', context='paper')
palette = sns.color_palette("colorblind", n_colors=len(models))

fig, axes = plt.subplots(2, 2, figsize=(12, 9))  # 稍大画布减少拥挤

plot_metrics = ['Spearman', 'MAE', 'Gini']
y_labels = {
    'Spearman': 'Spearman correlation',
    'MAE': 'MAE',
    'Gini': 'Gini coefficient (lower = fairer)'
}

# 绘制前三个指标的箱线图
for ax, metric in zip(axes.flat[:3], plot_metrics):
    vals = []
    labs = []
    for m in models:
        arr = data[metric][m]
        vals.extend(arr.tolist())
        labs.extend([m]*len(arr))
    df = pd.DataFrame({metric: vals, 'Model': labs})
    sns.boxplot(x='Model', y=metric, data=df, ax=ax, palette=palette, width=0.55,
                showfliers=True, boxprops=dict(linewidth=1.0), medianprops=dict(linewidth=1.2))
    sns.stripplot(x='Model', y=metric, data=df, ax=ax, color='k', size=4, jitter=0.18, alpha=0.7)
    ax.set_xticklabels(models, rotation=22, ha='right', fontsize=9)
    ax.set_ylabel(y_labels[metric], fontsize=10)
    ax.set_title(f'({ "a" if metric=="Spearman" else ("b" if metric=="MAE" else "c") }) {y_labels[metric]}', fontsize=11)

    # 计算当前数据范围并设上限留空间（根据比较次数动态分配）
    group_max = df.groupby('Model')[metric].max()
    overall_max = group_max.max()
    overall_min = df[metric].min()
    yrange = overall_max - overall_min if overall_max != overall_min else abs(overall_max) + 1.0
    top_margin = yrange * 0.30  # 保证足够的顶部空白以容纳多条注释
    ax.set_ylim(overall_min - yrange*0.08, overall_max + top_margin)

    # 对每个基线与 DSV-MARL 做 Welch t-test，并按索引给出逐级高度以避免重叠
    dsv_vals = data[metric]['DSV-MARL']
    # 生成逐步上升的高度（每个比较间隔）
    # 比较顺序与距离 DSV 的 index 顺序无关，只按基线索引依次叠加以保证可读性
    base_heights = []
    step = yrange * 0.06  # 每条标注之间的间隔
    next_height = yrange * 0.05  # 初始高度偏移
    for i, m in enumerate(models):
        if m == 'DSV-MARL':
            base_heights.append(None)
            continue
        comp_vals = data[metric][m]
        tstat, pval = stats.ttest_ind(comp_vals, dsv_vals, equal_var=False)
        stars = p_to_stars(pval)
        # 横线底部 y 坐标由对应两组最大值决定，再加上递增偏移
        y_base = max(comp_vals.max(), dsv_vals.max()) + next_height
        add_stat_annotation(ax, i, models.index('DSV-MARL'), y_base, step, stars)
        next_height += step  # 叠加

# 子图 (d) Convergence — 仅有 conv_models
ax = axes[1,1]
vals = []
labs = []
for m in conv_models:
    arr = data['Convergence'][m]
    vals.extend(arr.tolist())
    labs.extend([m]*len(arr))
df_conv = pd.DataFrame({'Convergence': vals, 'Model': labs})
sns.boxplot(x='Model', y='Convergence', data=df_conv, ax=ax,
            palette=[palette[1], palette[2], palette[3], palette[4]],
            width=0.6, boxprops=dict(linewidth=1.0), medianprops=dict(linewidth=1.2))
sns.stripplot(x='Model', y='Convergence', data=df_conv, ax=ax, color='k', size=4, jitter=0.18, alpha=0.7)
ax.set_xticklabels(conv_models, rotation=22, ha='right', fontsize=9)
ax.set_ylabel('Convergence rounds', fontsize=10)
ax.set_title('(d) Convergence rounds (lower = faster)', fontsize=11)

# Convergence y-range adjust and annotations
group_max = df_conv.groupby('Model')['Convergence'].max()
overall_max = group_max.max()
overall_min = df_conv['Convergence'].min()
yrange = overall_max - overall_min if overall_max != overall_min else overall_max*0.2 + 1
top_margin = yrange * 0.30
ax.set_ylim(overall_min - yrange*0.05, overall_max + top_margin)

dsv_vals = data['Convergence']['DSV-MARL']
step = yrange * 0.08
next_height = yrange * 0.05
for i, m in enumerate(conv_models):
    if m == 'DSV-MARL':
        continue
    comp_vals = data['Convergence'][m]
    tstat, pval = stats.ttest_ind(comp_vals, dsv_vals, equal_var=False)
    stars = p_to_stars(pval)
    y_base = max(comp_vals.max(), dsv_vals.max()) + next_height
    add_stat_annotation(ax, i, conv_models.index('DSV-MARL'), y_base, step, stars)
    next_height += step

# 布局微调
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
# 更稳妥地放置总标题（避免遮挡子图元素）
#fig.text(0.5, 0.99, 'DSV-MARL vs baselines — boxplot of repeated runs (simulated data, n=10)',
 #        ha='center', va='top', fontsize=13)

# 保存高质量文件
fig.savefig('figure_5_1_boxplot_simulated_clean.pdf', bbox_inches='tight')
fig.savefig('figure_5_1_boxplot_simulated_clean.png', dpi=300, bbox_inches='tight')

plt.show()
