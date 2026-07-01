#include "tsp_utils.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

std::vector<std::pair<double, double>> TSPUtils::loadCities(const std::string& filepath) {
    std::ifstream ifs(filepath);
    if (!ifs.is_open()) {
        throw std::runtime_error("无法打开文件: " + filepath);
    }

    std::vector<std::pair<double, double>> cities;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        double x, y;
        char comma;
        if (!(iss >> x >> comma >> y)) continue;
        cities.emplace_back(x, y);
    }
    return cities;
}

std::vector<std::vector<double>> TSPUtils::buildDistanceMatrix(
    const std::vector<std::pair<double, double>>& cities) {
    int n = static_cast<int>(cities.size());
    std::vector<std::vector<double>> dist(n, std::vector<double>(n, 0.0));
    
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double dx = cities[i].first - cities[j].first;
            double dy = cities[i].second - cities[j].second;
            double d = std::sqrt(dx * dx + dy * dy);
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }
    return dist;
}

double TSPUtils::calcPathLength(const std::vector<int>& path,
                                const std::vector<std::vector<double>>& dist) {
    double total = 0.0;
    int n = static_cast<int>(path.size());
    
    for (int i = 0; i < n - 1; ++i) {
        total += dist[path[i]][path[i + 1]];
    }
    // 回到起点
    total += dist[path[n - 1]][path[0]];
    return total;
}
