#!/usr/bin/env python3
"""运行三种 TSP 算法的对比"""

import sys
import os
import time

# 自动激活项目 .venv
def _activate_venv():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    venv_dir = os.path.join(project_dir, '.venv')
    if os.path.isdir(venv_dir):
        py_ver = f'python{sys.version_info.major}.{sys.version_info.minor}'
        site_packages = os.path.join(venv_dir, 'lib', py_ver, 'site-packages')
        if os.path.isdir(site_packages) and site_packages not in sys.path:
            sys.path.insert(0, site_packages)
_activate_venv()

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, '.')
from tsp_core import (load_cities, build_distance_matrix,
                      random_search_loop, simulated_annealing_loop,
                      genetic_algorithm_loop)

# --------------- 公共参数 ---------------
ITERATIONS = 1000000
LOG_INTERVAL = 100000
SA_T_START = 100.0
SA_T_END = 0.001
SA_ALPHA = 0.9999

# 遗传算法参数（总评估次数对齐到 ITERATIONS）
GA_POP = 100
GA_GENS = ITERATIONS // GA_POP  # 使总评估次数一致
GA_MUTATION = 0.05
GA_ELITE = 10

print("=" * 60)
print("TSP 算法对比")
print("=" * 60)

# 加载数据
print("\n加载城市数据...")
cities = load_cities("data/TSP1.txt")
n = len(cities)
print(f"城市数量: {n}")

print("构建距离矩阵...")
dist = build_distance_matrix(cities)
print("距离矩阵已构建 (size: {}x{})\n".format(n, n))

# 方法1：随机采样
print("=" * 60)
print("方法1: 随机采样 (Random Search)")
print("=" * 60)
t0 = time.time()
rs_best, rs_path, rs_iters, rs_vals = random_search_loop(
    dist, ITERATIONS, LOG_INTERVAL
)
rs_time = time.time() - t0
print(f"  最优路径: {rs_best:.4f}  耗时: {rs_time:.2f}s")

# 方法2：模拟退火
print("\n" + "=" * 60)
print("方法2: 模拟退火 (Simulated Annealing)")
print("=" * 60)
t0 = time.time()
sa_best, sa_path, sa_iters, sa_vals = simulated_annealing_loop(
    dist, ITERATIONS, LOG_INTERVAL, SA_T_START, SA_T_END, SA_ALPHA
)
sa_time = time.time() - t0
print(f"  最优路径: {sa_best:.4f}  耗时: {sa_time:.2f}s")

# 方法3：遗传算法
print("\n" + "=" * 60)
print("方法3: 遗传算法 (Genetic Algorithm)")
print("=" * 60)
print(f"  种群={GA_POP}, 代数={GA_GENS}, 变异率={GA_MUTATION}, 精英={GA_ELITE}")
t0 = time.time()
ga_best, ga_path, ga_gens, ga_vals = genetic_algorithm_loop(
    dist, GA_GENS, GA_GENS // 10, GA_POP, GA_MUTATION, GA_ELITE
)
ga_time = time.time() - t0
print(f"  最优路径: {ga_best:.4f}  耗时: {ga_time:.2f}s")

# ============ 绘制对比图 ============
# 提取城市坐标
xs = [city[0] for city in cities]
ys = [city[1] for city in cities]

# 遗传算法的横坐标转为评估次数（每代评估 pop_size 次）
ga_evals = [g * GA_POP for g in ga_gens]

# 图1：收敛曲线对比（对数坐标）
fig1, ax_conv = plt.subplots(figsize=(12, 6))
ax_conv.plot(rs_iters, rs_vals, linewidth=2, color='#1f77b4', alpha=0.5,
             label=f'随机采样 (最优={rs_best:.0f})')
ax_conv.plot(sa_iters, sa_vals, linewidth=2.5, color='#d62728',
             label=f'模拟退火 (最优={sa_best:.0f})')
ax_conv.plot(ga_evals, ga_vals, linewidth=2.5, color='#2ca02c',
             label=f'遗传算法 (最优={ga_best:.0f})')
ax_conv.set_xlabel('路径评估次数', fontsize=13)
ax_conv.set_ylabel('最优路径长度（对数坐标）', fontsize=13)
ax_conv.set_title('收敛曲线对比（对数坐标）', fontsize=15, fontweight='bold')
ax_conv.set_yscale('log')
ax_conv.legend(fontsize=11, loc='upper right')
ax_conv.grid(True, alpha=0.3, which='both')
plt.tight_layout()
os.makedirs('output', exist_ok=True)
plt.savefig('output/convergence.png', dpi=150, bbox_inches='tight')
print(f"\n收敛曲线已保存至 output/convergence.png")
plt.close()

# 图2：最终结果柱状图
fig2, ax_bar = plt.subplots(figsize=(10, 6))
methods = ['随机采样', '模拟退火', '遗传算法']
values = [rs_best, sa_best, ga_best]
times_list = [rs_time, sa_time, ga_time]
colors = ['#1f77b4', '#d62728', '#2ca02c']
bars = ax_bar.bar(methods, values, color=colors, width=0.5)
ax_bar.set_ylabel('最优路径长度', fontsize=13)
ax_bar.set_title('最终结果对比', fontsize=15, fontweight='bold')
for bar, val, t in zip(bars, values, times_list):
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{val:.2f}', ha='center', fontsize=13, fontweight='bold')
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f'{t:.1f}s', ha='center', fontsize=12, color='white', fontweight='bold')
ax_bar.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('output/result_bar.png', dpi=150, bbox_inches='tight')
print(f"柱状图已保存至 output/result_bar.png")
plt.close()

# 图3：三种算法路径可视化（分开画）
fig3, axes = plt.subplots(1, 3, figsize=(18, 6))

# 随机采样
axes[0].scatter(xs, ys, s=5, c='gray', alpha=0.4, zorder=1)
path_rs = rs_path + [rs_path[0]]
axes[0].plot([xs[i] for i in path_rs], [ys[i] for i in path_rs],
             linewidth=0.4, color='#1f77b4', alpha=0.4, zorder=2)
axes[0].set_title(f'随机采样\n最优={rs_best:.2f}, 耗时={rs_time:.1f}s', fontsize=13, fontweight='bold')
axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.2)

# 模拟退火
axes[1].scatter(xs, ys, s=5, c='gray', alpha=0.4, zorder=1)
path_sa = sa_path + [sa_path[0]]
axes[1].plot([xs[i] for i in path_sa], [ys[i] for i in path_sa],
             linewidth=0.4, color='#d62728', alpha=0.4, zorder=2)
axes[1].set_title(f'模拟退火\n最优={sa_best:.2f}, 耗时={sa_time:.1f}s', fontsize=13, fontweight='bold')
axes[1].set_aspect('equal')
axes[1].grid(True, alpha=0.2)

# 遗传算法
axes[2].scatter(xs, ys, s=5, c='gray', alpha=0.4, zorder=1)
path_ga = ga_path + [ga_path[0]]
axes[2].plot([xs[i] for i in path_ga], [ys[i] for i in path_ga],
             linewidth=0.4, color='#2ca02c', alpha=0.4, zorder=2)
axes[2].set_title(f'遗传算法\n最优={ga_best:.2f}, 耗时={ga_time:.1f}s', fontsize=13, fontweight='bold')
axes[2].set_aspect('equal')
axes[2].grid(True, alpha=0.2)

plt.suptitle(f'TSP 求解路径对比 ({n} 城市)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('output/paths.png', dpi=150, bbox_inches='tight')
print(f"路径对比图已保存至 output/paths.png")
plt.close()

# ============ 模拟退火 vs 遗传算法 单独对比图 ============
fig2 = plt.figure(figsize=(16, 18))

# 顶部：大型收敛曲线对比（用对数 Y 轴让差异更明显）
ax_conv = plt.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=1)
ax_conv.plot(sa_iters, sa_vals, linewidth=3, color='#d62728', marker='o',
             markersize=6, markevery=1, label=f'模拟退火 (最终={sa_best:.0f})',
             zorder=3)
ax_conv.plot(ga_evals, ga_vals, linewidth=3, color='#2ca02c', marker='s',
             markersize=6, markevery=1, label=f'遗传算法 (最终={ga_best:.0f})',
             zorder=3)
# 填充两者之间的差距
ax_conv.fill_between(sa_iters, sa_vals, alpha=0.1, color='#d62728')
ax_conv.fill_between(ga_evals, ga_vals, alpha=0.1, color='#2ca02c')
ax_conv.set_xlabel('路径评估次数', fontsize=14)
ax_conv.set_ylabel('最优路径长度', fontsize=14)
ax_conv.set_title('收敛曲线对比（模拟退火 vs 遗传算法）', fontsize=15, fontweight='bold')
ax_conv.legend(fontsize=12, loc='upper right')
ax_conv.grid(True, alpha=0.3)
# 标注最终差距
ax_conv.annotate(f'差距: {sa_best - ga_best:.0f}\n({(sa_best - ga_best) / sa_best * 100:.1f}%)',
                 xy=(max(ga_evals), ga_vals[-1]),
                 xytext=(max(ga_evals) * 0.6, (sa_vals[-1] + ga_vals[-1]) / 2),
                 fontsize=12, fontweight='bold', color='#333',
                 arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))

# 左下：模拟退火路径
ax = plt.subplot2grid((3, 2), (1, 0), rowspan=1)
ax.scatter(xs, ys, s=5, c='gray', alpha=0.4, zorder=1)
ax.plot([xs[i] for i in path_sa], [ys[i] for i in path_sa],
        linewidth=0.4, color='#d62728', alpha=0.4, zorder=2)
ax.set_title(f'模拟退火路径 (最优={sa_best:.0f}, {sa_time:.1f}s)', fontsize=13, fontweight='bold')
ax.set_aspect('equal')
ax.grid(True, alpha=0.2)

# 右下：遗传算法路径
ax = plt.subplot2grid((3, 2), (1, 1), rowspan=1)
ax.scatter(xs, ys, s=5, c='gray', alpha=0.4, zorder=1)
ax.plot([xs[i] for i in path_ga], [ys[i] for i in path_ga],
        linewidth=0.4, color='#2ca02c', alpha=0.4, zorder=2)
ax.set_title(f'遗传算法路径 (最优={ga_best:.0f}, {ga_time:.1f}s)', fontsize=13, fontweight='bold')
ax.set_aspect('equal')
ax.grid(True, alpha=0.2)

# 底部：柱状图对比（跨2列）
ax = plt.subplot2grid((3, 2), (2, 0), colspan=2, rowspan=1)
methods_sa_ga = ['模拟退火', '遗传算法']
values_sa_ga = [sa_best, ga_best]
times_sa_ga = [sa_time, ga_time]
colors_sa_ga = ['#d62728', '#2ca02c']
bars = ax.bar(methods_sa_ga, values_sa_ga, color=colors_sa_ga, width=0.3)
ax.set_ylabel('最优路径长度', fontsize=14)
ax.set_title('最终结果对比', fontsize=15, fontweight='bold')
for bar, val, t in zip(bars, values_sa_ga, times_sa_ga):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{val:.0f}', ha='center', fontsize=14, fontweight='bold')
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
            f'{t:.1f}s', ha='center', fontsize=12, color='white')
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle(f'模拟退火 vs 遗传算法 ({n} 城市, 总评估次数={ITERATIONS})',
             fontsize=16, fontweight='bold', y=0.99)
plt.tight_layout()
plt.savefig('output/sa_vs_ga.png', dpi=150, bbox_inches='tight')
print(f"模拟退火 vs 遗传算法对比图已保存至 output/sa_vs_ga.png")
