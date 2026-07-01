# distutils: language = c++
# distutils: sources = src/tsp_utils.cpp

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.string cimport string
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.time cimport time, time_t
from libc.math cimport exp

cdef extern from "tsp_utils.h":
    cdef cppclass TSPUtils:
        @staticmethod
        vector[pair[double, double]] loadCities(const string& filepath)
        
        @staticmethod
        vector[vector[double]] buildDistanceMatrix(
            const vector[pair[double, double]]& cities)
        
        @staticmethod
        double calcPathLength(const vector[int]& path,
                             const vector[vector[double]]& dist)


def load_cities(str filepath):
    """加载城市坐标，返回 [(x, y), ...] 列表"""
    cdef bytes filepath_bytes = filepath.encode('utf-8')
    cdef vector[pair[double, double]] cities = TSPUtils.loadCities(filepath_bytes)
    return [(c.first, c.second) for c in cities]


def build_distance_matrix(list cities):
    """构建距离矩阵，返回二维列表"""
    cdef vector[pair[double, double]] cpp_cities
    cdef double x, y
    
    for city in cities:
        x, y = city
        cpp_cities.push_back(pair[double, double](x, y))
    
    cdef vector[vector[double]] dist = TSPUtils.buildDistanceMatrix(cpp_cities)
    
    result = []
    for row in dist:
        result.append([d for d in row])
    return result


def calc_path_length(list path, list dist):
    """计算路径长度"""
    cdef vector[int] cpp_path
    cdef vector[vector[double]] cpp_dist
    cdef int i
    
    for p in path:
        cpp_path.push_back(p)
    
    for row in dist:
        cpp_dist.push_back(vector[double]())
        for d in row:
            cpp_dist.back().push_back(d)
    
    return TSPUtils.calcPathLength(cpp_path, cpp_dist)


def random_search_loop(list dist, int iterations, int log_interval):
    """在 Cython 层执行整个随机采样循环，避免 Python/C++ 数据转换开销
    
    Returns:
        (best_length, history_iters, history_values)
    """
    cdef int n = len(dist)
    cdef int i, j, k
    
    # 构建 C++ 距离矩阵
    cdef vector[vector[double]] cpp_dist
    for row in dist:
        cpp_dist.push_back(vector[double]())
        for d in row:
            cpp_dist.back().push_back(d)
    
    # 初始化路径
    cdef vector[int] path
    for i in range(n):
        path.push_back(i)
    
    cdef double current_len = TSPUtils.calcPathLength(path, cpp_dist)
    cdef double best_len = current_len
    cdef vector[int] best_path = path
    
    # 记录历史
    cdef list history_iters = []
    cdef list history_values = []
    
    # Fisher-Yates shuffle in Cython
    cdef int tmp
    cdef int swap_idx
    
    srand(time(NULL))
    
    for i in range(iterations):
        # Fisher-Yates shuffle
        for j in range(n - 1, 0, -1):
            swap_idx = rand() % (j + 1)
            tmp = path[j]
            path[j] = path[swap_idx]
            path[swap_idx] = tmp
        
        current_len = TSPUtils.calcPathLength(path, cpp_dist)
        
        if current_len < best_len:
            best_len = current_len
            best_path = path
        
        if (i + 1) % log_interval == 0:
            history_iters.append(i + 1)
            history_values.append(best_len)
    
    return best_len, [best_path[idx] for idx in range(n)], history_iters, history_values


def simulated_annealing_loop(list dist, int iterations, int log_interval,
                              double T_start, double T_end, double alpha):
    """模拟退火主循环

    Args:
        dist: 距离矩阵（二维列表）
        iterations: 总迭代次数
        log_interval: 每隔多少次记录一次
        T_start: 初始温度
        T_end: 结束温度
        alpha: 降温速率 (0 < alpha < 1, 通常 0.95~0.999)

    Returns:
        (best_length, best_path, history_iters, history_values)
    """
    cdef int n = len(dist)
    cdef int i, j

    # 构建 C++ 距离矩阵
    cdef vector[vector[double]] cpp_dist
    for row in dist:
        cpp_dist.push_back(vector[double]())
        for d in row:
            cpp_dist.back().push_back(d)

    # 初始化路径
    cdef vector[int] path
    for i in range(n):
        path.push_back(i)

    # 随机打乱初始路径
    srand(time(NULL))
    cdef int swap_idx, tmp
    for i in range(n):
        swap_idx = rand() % n
        tmp = path[i]
        path[i] = path[swap_idx]
        path[swap_idx] = tmp

    cdef double current_len = TSPUtils.calcPathLength(path, cpp_dist)
    cdef double best_len = current_len
    cdef vector[int] best_path = path

    # 记录历史
    cdef list history_iters = []
    cdef list history_values = []

    cdef double T = T_start
    cdef double new_len, delta, r
    cdef int a, b

    for i in range(iterations):
        # 随机交换两个城市
        a = rand() % n
        b = rand() % n
        if a == b:
            if (i + 1) % log_interval == 0:
                history_iters.append(i + 1)
                history_values.append(best_len)
            continue

        # 执行交换
        tmp = path[a]
        path[a] = path[b]
        path[b] = tmp

        new_len = TSPUtils.calcPathLength(path, cpp_dist)
        delta = new_len - current_len

        if delta < 0:
            # 更优解，总是接受
            current_len = new_len
            if new_len < best_len:
                best_len = new_len
                best_path = path
        else:
            # 较差解，以概率 exp(-delta/T) 接受
            r = <double>rand() / <double>RAND_MAX
            if r < exp(-delta / T):
                current_len = new_len
            else:
                # 回退交换
                tmp = path[a]
                path[a] = path[b]
                path[b] = tmp

        # 降温
        if T > T_end:
            T *= alpha

        if (i + 1) % log_interval == 0:
            history_iters.append(i + 1)
            history_values.append(best_len)

    return best_len, [best_path[idx] for idx in range(n)], history_iters, history_values


def genetic_algorithm_loop(list dist, int generations, int log_interval,
                           int pop_size, double mutation_rate, int elite_size):
    """遗传算法主循环（优化版）"""
    cdef int n = len(dist)
    cdef int i, j, k, gen
    cdef double tmp_double
    cdef int tmp_int
    
    # 扁平距离矩阵（连续内存，cache 友好）
    cdef vector[double] flat_dist
    flat_dist.resize(n * n)
    for i in range(n):
        for j in range(n):
            flat_dist[i * n + j] = dist[i][j]
    
    srand(time(NULL))
    
    # 种群：扁平存储 pop_size * n
    cdef vector[int] pop_flat
    pop_flat.resize(pop_size * n)
    cdef int swap_idx
    
    for i in range(pop_size):
        for j in range(n):
            pop_flat[i * n + j] = j
        # Fisher-Yates shuffle
        for j in range(n - 1, 0, -1):
            swap_idx = rand() % (j + 1)
            tmp_int = pop_flat[i * n + j]
            pop_flat[i * n + j] = pop_flat[i * n + swap_idx]
            pop_flat[i * n + swap_idx] = tmp_int
    
    # 适应度
    cdef vector[double] fitnesses
    fitnesses.resize(pop_size)
    cdef double path_len
    cdef int base_i, p1, p2
    
    # 记录历史
    cdef list history_gens = []
    cdef list history_values = []
    
    cdef double best_len = 1e18
    cdef int best_idx = 0
    cdef int gen_best_idx
    
    # 锦标赛选择
    cdef int tournament_k = 5
    cdef int candidates[5]
    cdef double fit_candidates[5]
    cdef int best_cand_idx
    
    # 交叉参数
    cdef int start_pos, end_pos, pos, p2_pos, filled, frag_len
    cdef int a, b, c
    
    # 2-opt 局部搜索
    cdef int i1, i2, elite_idx, left, right, opt_iters
    cdef double old_dist, new_dist
    cdef int improved
    cdef int opt_limit = 3  # 只对前 3 个精英做 2-opt
    cdef int opt_interval = 50  # 每 50 代做一次 2-opt
    cdef int opt_max_iters = 5  # 最多迭代 5 轮
    cdef int c_i1, c_i1p, c_i2, c_i2p
    
    # 精英排序
    cdef vector[double] sorted_fitnesses
    cdef vector[int] sorted_indices
    sorted_fitnesses.resize(pop_size)
    sorted_indices.resize(pop_size)
    
    # 新种群
    cdef vector[int] new_pop_flat
    new_pop_flat.resize(pop_size * n)
    cdef int new_count
    
    # 子代（复用）
    cdef vector[int] child
    child.resize(n)
    cdef vector[int] in_child
    in_child.resize(n)
    cdef int stamp = 1
    
    # 父代指针（在扁平数组中的偏移）
    cdef int p1_base, p2_base
    
    for gen in range(generations):
        # 计算适应度（内联距离计算，避免函数调用）
        for i in range(pop_size):
            path_len = 0.0
            base_i = i * n
            for j in range(n - 1):
                p1 = pop_flat[base_i + j]
                p2 = pop_flat[base_i + j + 1]
                path_len += flat_dist[p1 * n + p2]
            # 回到起点
            p1 = pop_flat[base_i + n - 1]
            p2 = pop_flat[base_i]
            path_len += flat_dist[p1 * n + p2]
            fitnesses[i] = path_len
        
        # 找当前代最优
        gen_best_idx = 0
        for i in range(1, pop_size):
            if fitnesses[i] < fitnesses[gen_best_idx]:
                gen_best_idx = i
        
        if fitnesses[gen_best_idx] < best_len:
            best_len = fitnesses[gen_best_idx]
            best_idx = gen_best_idx
        
        # 记录历史
        if (gen + 1) % log_interval == 0:
            history_gens.append(gen + 1)
            history_values.append(best_len)
        
        # 精英排序（选择排序前 elite_size 个）
        for i in range(pop_size):
            sorted_fitnesses[i] = fitnesses[i]
            sorted_indices[i] = i
        
        for i in range(elite_size):
            for j in range(pop_size - 1, i, -1):
                if sorted_fitnesses[j] < sorted_fitnesses[j-1]:
                    tmp_double = sorted_fitnesses[j]
                    sorted_fitnesses[j] = sorted_fitnesses[j-1]
                    sorted_fitnesses[j-1] = tmp_double
                    tmp_int = sorted_indices[j]
                    sorted_indices[j] = sorted_indices[j-1]
                    sorted_indices[j-1] = tmp_int
        
        # 构建新种群
        new_count = 0
        
        # 精英直接复制
        for i in range(elite_size):
            src_base = sorted_indices[i] * n
            dst_base = new_count * n
            for j in range(n):
                new_pop_flat[dst_base + j] = pop_flat[src_base + j]
            new_count += 1

        # 每隔 opt_interval 代对前几个精英做 2-opt 局部搜索
        if gen % opt_interval == 0:
            for elite_idx in range(min(opt_limit, elite_size)):
                base_i = elite_idx * n
                improved = 1
                opt_iters = 0
                while improved and opt_iters < opt_max_iters:
                    improved = 0
                    opt_iters += 1
                    for i1 in range(n - 2):
                        for i2 in range(i1 + 2, n):
                            if i1 == 0 and i2 == n - 1:
                                continue
                            c_i1 = new_pop_flat[base_i + i1]
                            c_i1p = new_pop_flat[base_i + i1 + 1]
                            c_i2 = new_pop_flat[base_i + i2]
                            c_i2p = new_pop_flat[base_i + (i2 + 1) % n]
                            old_dist = flat_dist[c_i1 * n + c_i1p] + flat_dist[c_i2 * n + c_i2p]
                            new_dist = flat_dist[c_i1 * n + c_i2] + flat_dist[c_i1p * n + c_i2p]
                            if new_dist < old_dist - 1e-10:
                                left = i1 + 1
                                right = i2
                                while left < right:
                                    tmp_int = new_pop_flat[base_i + left]
                                    new_pop_flat[base_i + left] = new_pop_flat[base_i + right]
                                    new_pop_flat[base_i + right] = tmp_int
                                    left += 1
                                    right -= 1
                                improved = 1
        
        # 生成剩余个体
        while new_count < pop_size:
            # 锦标赛选择父代1
            for k in range(tournament_k):
                candidates[k] = rand() % pop_size
                fit_candidates[k] = fitnesses[candidates[k]]
            best_cand_idx = 0
            for k in range(1, tournament_k):
                if fit_candidates[k] < fit_candidates[best_cand_idx]:
                    best_cand_idx = k
            p1_base = candidates[best_cand_idx] * n
            
            # 锦标赛选择父代2
            for k in range(tournament_k):
                candidates[k] = rand() % pop_size
                fit_candidates[k] = fitnesses[candidates[k]]
            best_cand_idx = 0
            for k in range(1, tournament_k):
                if fit_candidates[k] < fit_candidates[best_cand_idx]:
                    best_cand_idx = k
            p2_base = candidates[best_cand_idx] * n
            
            # 顺序交叉 (OX)
            start_pos = rand() % n
            end_pos = start_pos + rand() % (n - start_pos)
            if end_pos >= n:
                end_pos = n - 1
            frag_len = end_pos - start_pos + 1
            
            # 复制父1的片段
            for i in range(start_pos, end_pos + 1):
                child[i] = pop_flat[p1_base + i]
                in_child[child[i]] = stamp
            
            # 从父2填充剩余
            pos = (end_pos + 1) % n
            p2_pos = (end_pos + 1) % n
            filled = 0
            while filled < n - frag_len:
                c = pop_flat[p2_base + p2_pos]
                if in_child[c] != stamp:
                    child[pos] = c
                    in_child[c] = stamp
                    filled += 1
                    pos = (pos + 1) % n
                p2_pos = (p2_pos + 1) % n
            
            # 交换变异
            if <double>rand() / <double>RAND_MAX < mutation_rate:
                a = rand() % n
                b = rand() % n
                if a != b:
                    tmp_int = child[a]
                    child[a] = child[b]
                    child[b] = tmp_int
            
            # 写入新种群
            dst_base = new_count * n
            for j in range(n):
                new_pop_flat[dst_base + j] = child[j]
            new_count += 1
            stamp += 1
        
        # 替换种群（O(1) 交换，避免整段复制）
        pop_flat.swap(new_pop_flat)
    
    # 返回最优路径
    cdef list best_path = []
    cdef int best_base = best_idx * n
    for i in range(n):
        best_path.append(pop_flat[best_base + i])
    
    return best_len, best_path, history_gens, history_values
