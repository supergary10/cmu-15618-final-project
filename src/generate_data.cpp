#include "../include/generate_data.hpp"
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <algorithm>

std::vector<std::vector<int>> loadTestData(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<int>> data;

    if (!file) {
        std::cerr << "Unable to open file: " << filename << "\n";
        return data;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int val;
        std::vector<int> vec;
        while (iss >> val) {
            vec.push_back(val);
        }
        std::sort(vec.begin(), vec.end());
        data.push_back(vec);
    }

    return data;
}