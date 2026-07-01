#!/usr/bin/env python3
"""模拟退火算法求解 TSP"""

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
from tsp_core import load_cities, build_distance_matrix, simulated_annealing_loop


def main():
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    log_interval = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    T_start = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0
    T_end = float(sys.argv[4]) if len(sys.argv) > 4 else 0.001
    alpha = float(sys.argv[5]) if len(sys.argv) > 5 else 0.9999

    print("加载城市数据...")
    cities = load_cities("data/TSP1.txt")
    n = len(cities)
    print(f"城市数量: {n}")

    print("构建距离矩阵...")
    dist = build_distance_matrix(cities)
    print("距离矩阵已构建")

    print(f"\n参数: 迭代={iterations}, T_start={T_start}, T_end={T_end}, alpha={alpha}")
    print(f"开始模拟退火...")
    start_time = time.time()

    best_len, best_path, history_iters, history_values = simulated_annealing_loop(
        dist, iterations, log_interval, T_start, T_end, alpha
    )

    elapsed = time.time() - start_time

    print(f"\n模拟退火完成!")
    print(f"耗时: {elapsed:.2f} 秒")
    print(f"最短路径长度: {best_len:.6f}")
    for it, val in zip(history_iters, history_values):
        print(f"  迭代 {it:>7d}: T≈{T_start * (alpha ** it):.4f}, 最优长度 = {val:.6f}")

    # 绘制收敛曲线
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(history_iters, history_values, linewidth=2, color='#d62728')
    axes[0].set_xlabel('迭代次数', fontsize=12)
    axes[0].set_ylabel('最优路径长度', fontsize=12)
    axes[0].set_title('模拟退火收敛曲线', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # 计算温度曲线
    temps = [max(T_start * (alpha ** it), T_end) for it in history_iters]
    axes[1].plot(history_iters, temps, linewidth=2, color='#1f77b4')
    axes[1].set_xlabel('迭代次数', fontsize=12)
    axes[1].set_ylabel('温度 T', fontsize=12)
    axes[1].set_title('温度衰减曲线', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/simulated_annealing.png', dpi=150)
    print(f"\n收敛曲线已保存至 output/simulated_annealing.png")


if __name__ == "__main__":
    main()
