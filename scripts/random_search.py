import sys
import os
import time

# 自动激活项目 .venv（无需手动 source activate）
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

# 设置中文字体（macOS）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, '.')

from tsp_core import load_cities, build_distance_matrix, random_search_loop


def main():
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 1000000
    log_interval = int(sys.argv[2]) if len(sys.argv) > 2 else 100000

    print("加载城市数据...")
    cities = load_cities("data/TSP1.txt")
    n = len(cities)
    print(f"城市数量: {n}")

    print("构建距离矩阵...")
    dist = build_distance_matrix(cities)
    print("距离矩阵已构建")

    print(f"开始随机采样 {iterations} 次 (每 {log_interval} 次记录一次)...")
    start_time = time.time()

    best_len, history_iters, history_values = random_search_loop(
        dist, iterations, log_interval
    )

    elapsed = time.time() - start_time

    print(f"\n采样完成!")
    print(f"耗时: {elapsed:.2f} 秒")
    print(f"最短路径长度: {best_len:.6f}")

    for it, val in zip(history_iters, history_values):
        print(f"  迭代 {it:>8d}: 最优路径长度 = {val:.6f}")

    # 绘制收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history_iters, history_values, linewidth=2)
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('最优路径长度', fontsize=12)
    plt.title('随机采样收敛曲线', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片到 output 目录
    import os
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/convergence_curve.png', dpi=150)
    print(f"\n收敛曲线已保存至 output/convergence_curve.png")


if __name__ == "__main__":
    main()
