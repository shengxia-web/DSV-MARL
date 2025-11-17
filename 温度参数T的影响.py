import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import time
from pathlib import Path

# ---------------- 配置 ----------------
np.random.seed(2025)
out_dir = Path('.')
out_dir.mkdir(parents=True, exist_ok=True)

num_agents = 6           # 代理数量（全枚举可行时建议不太大）
alpha = 0.6              # value 函数非线性指数：v(S) = (sum w_i)^alpha
weights = np.sort(np.random.uniform(0.5, 1.5, size=num_agents))[::-1]  # 从大到小排列，便于观察
agent_names = [f"A{i+1}" for i in range(num_agents)]

# Monte Carlo 参数
M = 2000                 # 每个 τ 下采样的排列数量（单次试验内）
n_trials = 40            # 对每个 τ 做重复试验以评估方差
taus = np.logspace(-2, 2, 12)  # τ 值范围（从 1e-2 到 1e2）

timestr = time.strftime("%Y%m%d-%H%M%S")

# ---------------- matplotlib 风格：优雅回退 ----------------
preferred_style = 'seaborn-darkgrid'
if preferred_style in plt.style.available:
    plt.style.use(preferred_style)
else:
    # 若 seaborn 样式不可用，回退到 matplotlib 内置样式
    fallback = 'default'  # 或 'classic'
    plt.style.use(fallback)
    print(f"Style '{preferred_style}' not available. Using fallback style '{fallback}'.")

# ---------------- 定义 value 与精确 Shapley ----------------
def v_of_S(S):
    if len(S) == 0:
        return 0.0
    # S 可以是 list/set；使用 numpy 索引求和
    return float(np.sum(weights[list(S)]) ** alpha)

def exact_shapley(num_agents):
    shap = np.zeros(num_agents)
    count = 0
    for p in itertools.permutations(range(num_agents)):
        count += 1
        present = set()
        for i in p:
            before = v_of_S(present)
            present.add(i)
            after = v_of_S(present)
            shap[i] += (after - before)
    shap /= count
    return shap

print("Computing exact (uniform-permutation) Shapley...")
t0 = time.time()
shap_true = exact_shapley(num_agents)
t_exact = time.time() - t0
print(f"Done. Time {t_exact:.3f}s. True Shapley: {np.round(shap_true, 6)}")

# ---------------- Plackett-Luce 排列采样（数值稳定实现） ----------------
def sample_pl_permutation(weights, tau):
    """
    使用 Plackett-Luce 顺序抽样，数值稳定：
    logits = weights / tau, 使用 exp(logits - max_logit) 防止 overflow
    如果剩余 scores 全为 0（下溢），则选择 logits 最大项（确定性退化）。
    """
    n = len(weights)
    remaining = list(range(n))
    perm = []
    # 预计算 logits（用 weights / tau）
    # 注意：tau 不能为 0（不会传 0），但可能非常小
    logits_full = np.array(weights, dtype=float) / float(tau)
    # sequentially choose
    while remaining:
        logits = logits_full[remaining]
        # 数值稳定化
        max_logit = np.max(logits)
        stabilized = logits - max_logit
        scores = np.exp(stabilized)
        ssum = scores.sum()
        if ssum == 0 or not np.isfinite(ssum):
            # 下溢/非有限：退化到选择 logits 最大的元素
            # 选择剩余中 logits 最大的索引
            idx_local = int(np.argmax(logits))
        else:
            probs = scores / ssum
            # 如果数值问题导致 probs 中有 nan，退化为 argmax
            if np.any(~np.isfinite(probs)):
                idx_local = int(np.argmax(logits))
            else:
                idx_local = np.random.choice(len(remaining), p=probs)
        perm.append(remaining.pop(idx_local))
    return perm

# Monte Carlo estimator given tau and M
def mc_shapley_pl(weights, tau, M):
    n = len(weights)
    shap_est = np.zeros(n)
    for _ in range(M):
        perm = sample_pl_permutation(weights, tau)
        present = set()
        for i in perm:
            before = v_of_S(present)
            present.add(i)
            after = v_of_S(present)
            shap_est[i] += (after - before)
    return shap_est / M

# ---------------- 对不同 tau 进行多次试验，记录 MAE 与个体估计 ----------------
records = []
per_agent_estimates = {tau: {i: [] for i in range(num_agents)} for tau in taus}

for tau in taus:
    mae_list = []
    trial_times = []
    for trial in range(n_trials):
        t1 = time.time()
        est = mc_shapley_pl(weights, tau, M)
        t2 = time.time()
        mae = np.mean(np.abs(est - shap_true))
        mae_list.append(mae)
        trial_times.append(t2 - t1)
        # 保存每个 agent 的估计
        for i in range(num_agents):
            per_agent_estimates[tau][i].append(est[i])
    records.append({
        'tau': tau,
        'mae_mean': np.mean(mae_list),
        'mae_std': np.std(mae_list, ddof=1),
        'time_mean': np.mean(trial_times)
    })
    print(f"tau={tau:.3g}: MAE mean={np.mean(mae_list):.6f}, std={np.std(mae_list, ddof=1):.6f}, time~{np.mean(trial_times):.3f}s")

df_records = pd.DataFrame(records)
df_records.to_csv(out_dir / f"tau_effect_summary_{timestr}.csv", index=False)

# 保存 per-agent 原始数据（长表）
rows = []
for tau in taus:
    for i in range(num_agents):
        for val in per_agent_estimates[tau][i]:
            rows.append({'tau': tau, 'agent': agent_names[i], 'estimate': val})
pd.DataFrame(rows).to_csv(out_dir / f"tau_effect_per_agent_{timestr}.csv", index=False)

# ---------------- 绘图 ----------------
fig, axes = plt.subplots(ncols=2, figsize=(12, 5), gridspec_kw={'width_ratios':[2,1]})

# 左图：MAE vs tau（对数 x），误差条为 std
ax = axes[0]
ax.errorbar(df_records['tau'], df_records['mae_mean'], yerr=df_records['mae_std'],
            fmt='o-', capsize=4, color='C0', label='MAE (mean ± std)')
ax.set_xscale('log')
ax.set_xlabel(r'Temperature $\tau$ (log scale)')
ax.set_ylabel('Mean absolute error vs true Shapley')
ax.set_title('Effect of temperature τ on Shapley estimator (PL-sampled permutations)')
ax.grid(True, linestyle=':', alpha=0.6)
ax.axhline(0.0, color='gray', linestyle='--', linewidth=1, alpha=0.6)
ax.legend()

# 右图：对若干典型 agents 画估计随 tau 的均值 ± std
ax2 = axes[1]
selected_agents = [0, 1, 2]  # 选前三个权重大/典型 agent，可改
for i in selected_agents:
    means = [np.mean(per_agent_estimates[tau][i]) for tau in taus]
    stds = [np.std(per_agent_estimates[tau][i], ddof=1) for tau in taus]
    ax2.errorbar(taus, means, yerr=stds, fmt='o-', capsize=3, label=f"{agent_names[i]} (w={weights[i]:.2f})")
    # 画真值水平线
    ax2.hlines(shap_true[i], xmin=taus.min(), xmax=taus.max(), colors='C'+str(i), linestyles='--', alpha=0.6)

ax2.set_xscale('log')
ax2.set_xlabel(r'Temperature $\tau$ (log scale)')
ax2.set_ylabel('Estimated Shapley value')
ax2.set_title('Per-agent estimates vs τ')
ax2.legend(fontsize=9)
ax2.grid(False)

# plt.suptitle('Influence of temperature parameter τ on PL-weighted Shapley estimation', fontsize=12)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

png_path = out_dir / f"tau_effect_shapley_{timestr}.png"
plt.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.show()
print(f"Saved plot: {png_path}")
print("Saved CSVs:", df_records.shape, "and per-agent long table.")

