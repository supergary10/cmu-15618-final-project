#ifndef ENHANCED_INTERSECTION_HPP
#define ENHANCED_INTERSECTION_HPP

#include <vector>

std::vector<int> binarySearchIntersection(const std::vector<std::vector<int>>& inputVectors, int numPartitions);

std::vector<int> adaptiveRangePartition(const std::vector<std::vector<int>>& inputVectors, int numPartitions);

std::vector<int> parallelLeapfrogJoin(const std::vector<std::vector<int>>& inputVectors, int numPartitions);

std::vector<int> workStealingIntersection(const std::vector<std::vector<int>>& inputVectors, int numPartitions);

#endif