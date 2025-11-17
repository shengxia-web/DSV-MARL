import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import time
from pathlib import Path

# ---------- 配置 ----------
np.random.seed(2025)
out_dir = Path('.')
out_dir.mkdir(parents=True, exist_ok=True)

num_agents = 6           # agent 数量（注意：全枚举复杂度为 num_agents!，6! = 720 可接受）
n_trials = 60            # 对每个 M 的重复次数（用于估计方差）
M_list = [10, 30, 100, 300, 1000, 3000, 10000]  # 不同的蒙特卡洛采样次数
alpha = 0.6              # 非线性参数，用于构造 value(S) = (sum w_i)^alpha，使得 Shapley 非平凡

# ---------- 构造问题：每个 agent 的基础权重（可看作信息量/信号强度） ----------
weights = np.sort(np.random.uniform(0.5, 1.5, size=num_agents))[::-1]  # 排序便于可视化（从大到小）
agent_names = [f"A{i+1}" for i in range(num_agents)]

# 定义 value 函数：对给定 coalition S（索引列表）返回 scalar value
def v_of_S(S):
    if len(S) == 0:
        return 0.0
    s = weights[list(S)].sum()
    return s**alpha

# ---------- 精确 Shapley（通过对所有排列求边际贡献平均） ----------
def exact_shapley(num_agents):
    perms = itertools.permutations(range(num_agents))
    shap = np.zeros(num_agents)
    count = 0
    for p in perms:
        count += 1
        present = set()
        for i in p:
            before = v_of_S(present)
            present.add(i)
            after = v_of_S(present)
            shap[i] += (after - before)
    shap = shap / count
    return shap

t0 = time.time()
shap_true = exact_shapley(num_agents)
t_exact = time.time() - t0
print(f"Exact Shapley computed in {t_exact:.3f}s. True Shapley: {shap_true}")

# ---------- 蒙特卡洛估计函数（从所有排列中采样 M 次） ----------
def mc_shapley(num_agents, M):
    shap_est = np.zeros(num_agents)
    for _ in range(M):
        perm = np.random.permutation(num_agents)
        present = set()
        for i in perm:
            before = v_of_S(present)
            present.add(i)
            after = v_of_S(present)
            shap_est[i] += (after - before)
    return shap_est / M

# ---------- 对不同 M 做多次重复试验，记录每次的估计与误差 ----------
records = []
agent_samples = {m: {i: [] for i in range(num_agents)} for m in M_list}  # 保存每次 trial 的每 agent 估计（用于箱线图）
for m in M_list:
    mae_list = []
    rms_list = []
    times_m = []
    for trial in range(n_trials):
        t1 = time.time()
        est = mc_shapley(num_agents, m)
        t2 = time.time()
        mae = np.mean(np.abs(est - shap_true))              # mean absolute error over agents
        rmse = np.sqrt(np.mean((est - shap_true)**2))
        mae_list.append(mae)
        rms_list.append(rmse)
        times_m.append(t2 - t1)
        # 记录单个 agent 估计（后面绘箱线图）：这里按 agent 保留
        for i in range(num_agents):
            agent_samples[m][i].append(est[i])
    records.append({
        "M": m,
        "mae_mean": np.mean(mae_list),
        "mae_std": np.std(mae_list, ddof=1),
        "rmse_mean": np.mean(rms_list),
        "time_mean": np.mean(times_m)
    })
    print(f"M={m}: MAE mean={np.mean(mae_list):.5f}, MAE std={np.std(mae_list, ddof=1):.5f}, time per trial ~{np.mean(times_m):.4f}s")

df_records = pd.DataFrame(records)
df_records.to_csv(out_dir / "mc_convergence_summary.csv", index=False)

# ---------- 绘图：左图 MAE vs M（对数 x），右图 选定 agent 的估计分布箱线图 ----------
fig, axes = plt.subplots(ncols=2, figsize=(12, 5), gridspec_kw={'width_ratios':[2,1]})

# 左：MAE 随 M（点 + 误差条），并加 1/sqrt(M) 参考线
ax = axes[0]
ax.errorbar(df_records['M'], df_records['mae_mean'], yerr=df_records['mae_std'],
            fmt='o-', capsize=4, label='MC MAE (mean ± std)', color='C0')
ax.set_xscale('log')
ax.set_xlabel('Monte Carlo samples M (log scale)')
ax.set_ylabel('Mean absolute error (averaged over agents)')
ax.set_title('Convergence of MC Shapley estimator vs M')
# 参考 1/sqrt(M) 线：按首点缩放
M_arr = np.array(df_records['M'])
ref = df_records['mae_mean'].iloc[0] * (M_arr[0]**0.5) * (1.0 / np.sqrt(M_arr))
ax.plot(M_arr, ref, '--', color='gray', label=r'Ref: 1/sqrt(M) (scaled)')
ax.legend()
ax.grid(True, linestyle=':', alpha=0.6)

# 右：为几个典型 M（挑选 M_list 中的 3 个不同规模）绘箱线图，展示某个 agent 的估计分布
ax2 = axes[1]
selected_agent = 0  # 展示第一个 agent（可改）
# 选择要绘的 M：小、中、大
selected_Ms = [M_list[0], M_list[2], M_list[-1]]
data_to_plot = [agent_samples[m][selected_agent] for m in selected_Ms]
ax2.boxplot(data_to_plot, labels=[f"M={m}" for m in selected_Ms], patch_artist=True,
            boxprops=dict(facecolor='lightgray', color='black'))
ax2.axhline(shap_true[selected_agent], color='red', linestyle='--', label='True Shapley')
ax2.set_title(f'Estimator distribution for {agent_names[selected_agent]}')
ax2.set_ylabel('Estimated Shapley value')
ax2.legend()
ax2.grid(False)

# plt.suptitle('Effect of Monte Carlo sample size M on Shapley estimation', fontsize=12)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

png_path = out_dir / "mc_convergence_shapley.png"
plt.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.show()
print(f"Saved figure: {png_path}")
