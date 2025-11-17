import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pathlib import Path

# ---------- 配置 ----------
np.random.seed(123)
num_agents = 8
T = 100
t_shock = 50
shock_mag = 0.6
shock_width = 12
seeds = 30
out_dir = Path('.')
out_dir.mkdir(parents=True, exist_ok=True)

agents = [f"Agent_{i+1}" for i in range(num_agents)]
times = np.arange(T)

# ---------- 生成 market（多 seed，均值 + 95% CI） ----------
base = 1.0 + 0.004 * times
seasonal = 0.05 * np.sin(2 * np.pi * times / 20)
sigma = shock_width / 3.0
shock_curve = np.exp(-0.5 * ((times - t_shock) / sigma) ** 2)
shock_curve = 1.0 + 1.0 * shock_curve

market_seeds = []
for s in range(seeds):
    obs_noise = np.random.normal(scale=0.02, size=T)
    arr = (base + seasonal) * shock_curve + obs_noise
    market_seeds.append(arr)
market_seeds = np.array(market_seeds)
market_mean = market_seeds.mean(axis=0)
market_lower = np.percentile(market_seeds, 2.5, axis=0)
market_upper = np.percentile(market_seeds, 97.5, axis=0)

# ---------- 生成 agent 贡献并列归一化 ----------
baseline = np.zeros((num_agents, T))
for i in range(num_agents):
    steps = np.random.normal(loc=0.0, scale=0.02, size=T)
    smooth = np.cumsum(steps)
    smooth = (smooth - smooth.min()) + 0.1*(i+1)
    baseline[i] = smooth

sensitive_agents = [0, 1, 2]
shock_inject = shock_mag * np.exp(-0.5 * ((times - t_shock) / sigma) ** 2)
for idx in sensitive_agents:
    baseline[idx] = baseline[idx] * (1.0 + shock_inject)

noise = np.random.normal(scale=0.01, size=baseline.shape)
contrib_raw = baseline + noise
contrib_raw[contrib_raw < 0] = 0.0
col_sums = contrib_raw.sum(axis=0, keepdims=True)
col_sums[col_sums == 0] = 1.0
contrib = contrib_raw / col_sums  # shape (num_agents, T)

# ---------- 累计贡献 ----------
cumulative = contrib.sum(axis=1)
cumulative_pct = cumulative / cumulative.sum()

# ---------- 保存 CSV ----------
df_matrix = pd.DataFrame(contrib, index=agents, columns=[f"t{tt}" for tt in times])
df_matrix.to_csv(out_dir / "shapley_matrix.csv")
rows = []
for i, a in enumerate(agents):
    for tt in range(T):
        rows.append({"agent": a, "time": tt, "contribution": float(contrib[i, tt])})
pd.DataFrame(rows).to_csv(out_dir / "figure3_shapley_long.csv", index=False)
pd.DataFrame(market_seeds).to_csv(out_dir / "market_all_seeds.csv", index=False)

# ---------- 绘图（优化排版，避免遮挡） ----------
fig = plt.figure(figsize=(13, 7), constrained_layout=True)
# 两行三列布局：第一行：top 时序（跨两列）；第二行：热图、colorbar narrow、累计条形
gs = gridspec.GridSpec(nrows=2, ncols=3, height_ratios=[1.0, 6], width_ratios=[8, 0.5, 2],
                       figure=fig)

# 先创建热图轴（底部左侧），再创建顶部共享 x 轴的轴（兼容各种 matplotlib 版本）
ax_heat = fig.add_subplot(gs[1, 0])
ax_color = fig.add_subplot(gs[1, 1])   # narrow axis for colorbar
ax_bar = fig.add_subplot(gs[1, 2])
ax_top = fig.add_subplot(gs[0, 0:2], sharex=ax_heat)

# 顶部 market 时序（均值 + 95% CI）
ax_top.plot(times, market_mean, color='black', lw=1.6)
ax_top.fill_between(times, market_lower, market_upper, color='black', alpha=0.18)
ax_top.axvline(t_shock, color='red', linestyle='--', lw=1)
# 在顶部加入 Shock 标注，放在轴内顶部，不会覆盖热图
ax_top.annotate('Shock', xy=(t_shock, market_mean.max()), xytext=(t_shock+2, market_mean.max()*0.98),
                color='red', fontsize=9, va='top')
ax_top.set_ylabel('Market value', fontsize=10)
ax_top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
ax_top.grid(False)

# 热图（agent x time）
im = ax_heat.imshow(contrib, aspect='auto', interpolation='nearest', cmap='viridis', origin='lower')
ax_heat.set_yticks(np.arange(num_agents))
ax_heat.set_yticklabels(agents, fontsize=10)
# 只显示少量 xticks 并旋转，避免重叠
xticks = np.linspace(0, T-1, min(11, T), dtype=int)
ax_heat.set_xticks(xticks)
ax_heat.set_xticklabels([str(x) for x in xticks], rotation=45, ha='right', fontsize=9)
ax_heat.set_xlabel('Time window', fontsize=10, labelpad=8)
ax_heat.set_title('Shapley contributions (per-window normalized)', fontsize=11)
# 冲击竖线（在热图中微调位置）
ax_heat.axvline(t_shock, color='red', linestyle='--', linewidth=1.0)
# 避免在热图上写入额外文字导致遮挡，这里仅在顶部注释

# Colorbar：使用 narrow axis 安放，避免覆盖热图或 ytick
cbar = fig.colorbar(im, cax=ax_color)
cbar.ax.tick_params(labelsize=9)
cbar.set_label('Contribution', fontsize=9)
# 隐藏 colorbar 轴的左右空白
ax_color.yaxis.set_ticks_position('right')

# 右侧累计贡献横向条（与热图行对齐）
y_pos = np.arange(num_agents)
ax_bar.barh(y_pos, cumulative_pct, align='center', color='grey', alpha=0.85)
ax_bar.set_yticks([])  # 热图已显示 agent 名称
ax_bar.set_xlim(0, cumulative_pct.max() * 1.12)
ax_bar.set_xlabel('Cumulative %', fontsize=9, labelpad=6)
# 设置与热图对齐的 y 范围
ax_bar.set_ylim(-0.5, num_agents - 0.5)

# 美化：减小热图与条形图的内边距，确保不遮挡
for ax in (ax_top, ax_heat, ax_bar):
    ax.tick_params(axis='both', which='major', labelsize=9)

# 最后确保 x 轴范围一致
ax_top.set_xlim(-0.5, T-0.5)
ax_heat.set_xlim(-0.5, T-0.5)

# 保存并显示，使用 bbox_inches='tight' 防止文字被裁剪
png_path = out_dir / "figure3_shapley_heatmap_with_market_optimized.png"
plt.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.show()
print(f"Saved PNG: {png_path}")
