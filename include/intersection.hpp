#ifndef INTERSECTION_HPP
#define INTERSECTION_HPP

#include <vector>

std::vector<int> parallelLinearScanIntersection(const std::vector<std::vector<int>>& inputVectors, int numPartitions);

std::vector<int> rangePartition(const std::vector<std::vector<int>>& inputVectors, int numPartitions);

std::vector<int> parallelCriticalIntersection(const std::vector<std::vector<int>>& inputVectors, int numPartitions);

#endif
