#pragma once

#include <string>
#include <vector>

class TSPUtils {
public:
    // 从文件加载城市坐标
    static std::vector<std::pair<double, double>> loadCities(const std::string& filepath);

    // 预计算距离矩阵 (n x n)
    static std::vector<std::vector<double>> buildDistanceMatrix(
        const std::vector<std::pair<double, double>>& cities);

    // 计算路径总长度（回到起点）
    static double calcPathLength(const std::vector<int>& path,
                                 const std::vector<std::vector<double>>& dist);
};
