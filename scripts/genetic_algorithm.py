import sys
import os
import time
import matplotlib.pyplot as plt

# 设置中文字体（macOS）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, '.')

from tsp_core import load_cities, build_distance_matrix, genetic_algorithm_loop


def main():
    generations = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    log_interval = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    pop_size = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    mutation_rate = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
    elite_size = int(sys.argv[5]) if len(sys.argv) > 5 else 10

    print("加载城市数据...")
    cities = load_cities("data/TSP1.txt")
    n = len(cities)
    print(f"城市数量: {n}")

    print("构建距离矩阵...")
    dist = build_distance_matrix(cities)
    print("距离矩阵已构建")

    print(f"\n参数: 代数={generations}, 种群={pop_size}, 变异率={mutation_rate}, 精英={elite_size}")
    print(f"开始遗传算法...")
    start_time = time.time()

    best_len, best_path, history_gens, history_values = genetic_algorithm_loop(
        dist, generations, log_interval, pop_size, mutation_rate, elite_size
    )

    elapsed = time.time() - start_time

    print(f"\n遗传算法完成!")
    print(f"耗时: {elapsed:.2f} 秒")
    print(f"最短路径长度: {best_len:.6f}")
    for gen, val in zip(history_gens, history_values):
        print(f"  代数 {gen:>5d}: 最优长度 = {val:.6f}")

    # 绘制收敛曲线
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(history_gens, history_values, linewidth=2, color='#2ca02c')
    axes[0].set_xlabel('代数', fontsize=12)
    axes[0].set_ylabel('最优路径长度', fontsize=12)
    axes[0].set_title('遗传算法收敛曲线', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # 绘制路径
    path = best_path + [best_path[0]]
    xs = [cities[i][0] for i in path]
    ys = [cities[i][1] for i in path]
    axes[1].plot(xs, ys, linewidth=0.5, color='#2ca02c', alpha=0.6)
    axes[1].scatter([c[0] for c in cities], [c[1] for c in cities], s=5, c='gray', alpha=0.5)
    axes[1].set_xlabel('X', fontsize=12)
    axes[1].set_ylabel('Y', fontsize=12)
    axes[1].set_title(f'最优路径 (长度={best_len:.2f})', fontsize=14)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/genetic_algorithm.png', dpi=150)
    print(f"\n收敛曲线已保存至 output/genetic_algorithm.png")


if __name__ == "__main__":
    main()
