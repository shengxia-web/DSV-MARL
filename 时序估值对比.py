import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- 配置 ----------
np.random.seed(42)
T = 100                       # 时间窗口数量
seeds = 30                    # 不同随机种子数（用于置信区间估计）
t_shock = 50                  # 冲击发生时间点（索引）
shock_magnitude = 1.0         # 冲击增加量（相对于基线）
shock_decay = 10.0            # 冲击指数衰减速率（窗口数）
out_dir = Path('.')           # 输出目录

# ---------- 生成市场真实值（带少量观测噪声以便计算置信带） ----------
t = np.arange(T)
base = 1.0 + 0.005 * t                         # 线性趋势
seasonal = 0.05 * np.sin(2 * np.pi * t / 20)  # 周期性
shock_curve = np.ones(T)
post_idx = t >= t_shock
shock_curve[post_idx] += shock_magnitude * np.exp(-(t[post_idx] - t_shock) / shock_decay)

market_seeds = []
for s in range(seeds):
    obs_noise = np.random.normal(scale=0.02, size=T)
    arr = base + seasonal
    arr = arr * shock_curve + obs_noise
    market_seeds.append(arr)
market_seeds = np.array(market_seeds)  # shape (seeds, T)

# ---------- 生成模型估值（DSV-MARL 与 Static Shapley） ----------
dsv_seeds = []
for s in range(seeds):
    lag = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
    noise = np.random.normal(scale=0.03, size=T)
    est = np.roll(market_seeds[s], lag) + noise
    if lag > 0:
        est[:lag] = market_seeds[s][:lag] + np.random.normal(scale=0.03, size=lag)
    dsv_seeds.append(est)
dsv_seeds = np.array(dsv_seeds)

static_seeds = []
for s in range(seeds):
    long_term = base + seasonal
    resp_coef = np.random.uniform(0.0, 0.2)
    est = long_term * (1 + resp_coef * (shock_curve - 1))
    est = est + np.random.normal(scale=0.05, size=T)
    static_seeds.append(est)
static_seeds = np.array(static_seeds)

# ---------- 统计量：均值与 95% 置信区间（2.5%/97.5% 分位） ----------
def summary_stats(arr):
    mean = np.mean(arr, axis=0)
    lower = np.percentile(arr, 2.5, axis=0)
    upper = np.percentile(arr, 97.5, axis=0)
    return {"mean": mean, "lower": lower, "upper": upper}

market_stats = summary_stats(market_seeds)
dsv_stats = summary_stats(dsv_seeds)
static_stats = summary_stats(static_seeds)

# ---------- 保存 CSV（每列包含均值与上下界） ----------
df = pd.DataFrame({
    "time": t,
    "market_mean": market_stats["mean"],
    "market_lower": market_stats["lower"],
    "market_upper": market_stats["upper"],
    "dsv_mean": dsv_stats["mean"],
    "dsv_lower": dsv_stats["lower"],
    "dsv_upper": dsv_stats["upper"],
    "static_mean": static_stats["mean"],
    "static_lower": static_stats["lower"],
    "static_upper": static_stats["upper"],
})
csv_path = out_dir / "figure1_timeseries_data.csv"
df.to_csv(csv_path, index=False)
print(f"CSV saved to: {csv_path}")

# 如果想保存所有 seed 的原始序列，可以启用以下段落：
save_all_seeds = True
if save_all_seeds:
    # 保存 market seeds、dsv seeds、static seeds 为单独 CSV 文件（每行一个 seed）
    pd.DataFrame(market_seeds).to_csv(out_dir / "market_all_seeds.csv", index=False)
    pd.DataFrame(dsv_seeds).to_csv(out_dir / "dsv_all_seeds.csv", index=False)
    pd.DataFrame(static_seeds).to_csv(out_dir / "static_all_seeds.csv", index=False)
    print("All seed-level CSVs saved.")

# ---------- 绘图（用 matplotlib 风格，不依赖 seaborn） ----------
plt.style.use('ggplot')
plt.figure(figsize=(10, 5))

# 真值（黑色）及置信区间
plt.plot(t, market_stats["mean"], color="black", lw=2.2, label="Market (true)")
plt.fill_between(t, market_stats["lower"], market_stats["upper"], color="black", alpha=0.12)

# DSV-MARL（蓝色）
plt.plot(t, dsv_stats["mean"], color="#1f77b4", lw=1.8, label="DSV-MARL")
plt.fill_between(t, dsv_stats["lower"], dsv_stats["upper"], color="#1f77b4", alpha=0.18)

# Static Shapley（橙色）
plt.plot(t, static_stats["mean"], color="#ff7f0e", lw=1.8, label="Static Shapley")
plt.fill_between(t, static_stats["lower"], static_stats["upper"], color="#ff7f0e", alpha=0.18)

# 冲击时间标注
plt.axvline(t_shock, color="red", linestyle="--", lw=1)
ymin, ymax = plt.ylim()
plt.text(t_shock + 1, ymax*0.95, "Shock", color="red", va="top")

# 图例、标签
plt.xlabel("Time window")
plt.ylabel("Estimated value (normalized)")
# plt.title("Time series valuation: DSV-MARL vs Static Shapley vs Market")
plt.legend(loc="upper left")
plt.tight_layout()

png_path = out_dir / "figure1_timeseries.png"
plt.savefig(png_path, dpi=300)
plt.show()
print(f"PNG saved to: {png_path}")
