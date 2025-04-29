#include "enhanced_intersection.hpp"
#include <numeric>
#include <algorithm>
#include <limits>
#include <omp.h>
#include <iostream>
#include <cmath>
#include <unordered_set>
#include <vector>
#include <atomic>  // Add this for std::atomic
#include "LeapfrogJoin.hpp"

/**
 * Binary search intersection implementation
 * More efficient when vectors are of significantly different sizes
 */
std::vector<int> binarySearchIntersection(const std::vector<std::vector<int>>& inputVectors, int numPartitions) {
    if (inputVectors.empty()) {
        return {};
    }
    if (inputVectors.size() == 1) {
        return inputVectors[0];
    }

    // Find the smallest vector to iterate over
    int smallestIndex = -1;
    size_t minSize = std::numeric_limits<size_t>::max();
    for (size_t i = 0; i < inputVectors.size(); ++i) {
        if (!inputVectors[i].empty()) {
            if (smallestIndex == -1 || inputVectors[i].size() < minSize) {
                minSize = inputVectors[i].size();
                smallestIndex = static_cast<int>(i);
            }
        } else {
            return {};
        }
    }
    if (smallestIndex == -1) {
        return {};
    }

    std::cout << "Using binary search intersection with smallest vector index: " << smallestIndex << std::endl;
    const std::vector<int>& base = inputVectors[smallestIndex];
    std::vector<std::vector<int>> threadResults;

    #pragma omp parallel
    {
        #pragma omp single
        {
            threadResults.resize(omp_get_num_threads());
        }

        int tid = omp_get_thread_num();

        #pragma omp for schedule(dynamic, 64)
        for (int idx = 0; idx < static_cast<int>(base.size()); ++idx) {
            int candidate = base[idx];
            bool in_all = true;

            // Binary search for this candidate in all other vectors
            for (size_t j = 0; j < inputVectors.size(); ++j) {
                if (j == static_cast<size_t>(smallestIndex)) {
                    continue;
                }

                // Use binary search instead of linear scan
                if (!std::binary_search(inputVectors[j].begin(), inputVectors[j].end(), candidate)) {
                    in_all = false;
                    break;
                }
            }

            if (in_all) {
                threadResults[tid].push_back(candidate);
            }
        }
    }
    
    // Combine results
    std::vector<int> result;
    for (const auto& localVec : threadResults) {
        result.insert(result.end(), localVec.begin(), localVec.end());
    }
    std::sort(result.begin(), result.end());
    return result;
}

/**
 * Advanced range partitioning that adjusts partition sizes based on data distribution
 */
std::vector<int> adaptiveRangePartition(const std::vector<std::vector<int>>& inputVectors, int numPartitions) {
    if (inputVectors.empty()) {
        return {};
    }
    if (inputVectors.size() == 1) {
        return inputVectors[0];
    }

    // Find global min and max
    int global_min = std::numeric_limits<int>::max();
    int global_max = std::numeric_limits<int>::min();
    bool data_found = false;
    
    for (const auto& vec : inputVectors) {
        if (!vec.empty()) {
            global_min = std::min(global_min, vec.front());
            global_max = std::max(global_max, vec.back());
            data_found = true;
        } else {
            return {};
        }
    }

    if (!data_found) return {};

    if (global_min == global_max) {
        bool in_all = true;
        for(const auto& vec : inputVectors) {
            if (!std::binary_search(vec.begin(), vec.end(), global_min)) {
                in_all = false;
                break;
            }
        }
        return in_all ? std::vector<int>{global_min} : std::vector<int>{};
    }

    // Determine partition boundaries using sampling
    std::vector<int> boundaries;
    const size_t sample_size = 1000; // Number of samples to take
    std::vector<int> samples;

    // Take samples from all vectors
    for (const auto& vec : inputVectors) {
        if (vec.size() <= sample_size) {
            // If vector is small, add all elements
            samples.insert(samples.end(), vec.begin(), vec.end());
        } else {
            // Otherwise, take evenly distributed samples
            const double step = static_cast<double>(vec.size()) / sample_size;
            for (size_t i = 0; i < sample_size; ++i) {
                size_t idx = static_cast<size_t>(i * step);
                if (idx < vec.size()) {
                    samples.push_back(vec[idx]);
                }
            }
        }
    }

    // Sort and remove duplicates
    std::sort(samples.begin(), samples.end());
    samples.erase(std::unique(samples.begin(), samples.end()), samples.end());

    // Create partition boundaries
    const size_t partitions = std::min(static_cast<size_t>(numPartitions), samples.size());
    boundaries.resize(partitions - 1);
    
    for (size_t i = 0; i < partitions - 1; ++i) {
        size_t idx = (i + 1) * samples.size() / partitions;
        if (idx < samples.size()) {
            boundaries.push_back(samples[idx]);
        }
    }
    
    // Sort and remove duplicates from boundaries
    std::sort(boundaries.begin(), boundaries.end());
    boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());

    // Add global min and max to have complete ranges
    boundaries.insert(boundaries.begin(), global_min);
    boundaries.push_back(global_max + 1); // +1 to include global_max

    std::cout << "Using adaptive range partitioning with " << boundaries.size() - 1 << " partitions" << std::endl;

    // Process each partition in parallel
    std::vector<std::vector<int>> threadResults;
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            threadResults.resize(omp_get_num_threads());
        }

        int tid = omp_get_thread_num();

        #pragma omp for schedule(dynamic)
        for (size_t part = 0; part < boundaries.size() - 1; ++part) {
            int start_val = boundaries[part];
            int end_val = boundaries[part + 1] - 1; // -1 because end is exclusive
            
            // Find ranges in each vector that fall within this partition
            std::vector<std::pair<std::vector<int>::const_iterator, std::vector<int>::const_iterator>> ranges;
            
            for (const auto& vec : inputVectors) {
                auto start_it = std::lower_bound(vec.begin(), vec.end(), start_val);
                auto end_it = std::upper_bound(start_it, vec.end(), end_val);
                ranges.emplace_back(start_it, end_it);
                
                // Early termination if any vector has no elements in this range
                if (start_it == end_it) {
                    ranges.clear(); // No intersection possible
                    break;
                }
            }

            // If all vectors have elements in this range, find intersection
            if (!ranges.empty()) {
                // Find smallest range for iteration
                size_t smallest_idx = 0;
                size_t min_range_size = std::distance(ranges[0].first, ranges[0].second);
                
                for (size_t i = 1; i < ranges.size(); ++i) {
                    size_t range_size = std::distance(ranges[i].first, ranges[i].second);
                    if (range_size < min_range_size) {
                        min_range_size = range_size;
                        smallest_idx = i;
                    }
                }

                // Iterate through smallest range and check others using binary search
                for (auto it = ranges[smallest_idx].first; it != ranges[smallest_idx].second; ++it) {
                    int candidate = *it;
                    bool in_all = true;
                    
                    for (size_t i = 0; i < ranges.size(); ++i) {
                        if (i == smallest_idx) continue;
                        
                        // Check if candidate exists in this range
                        auto start = ranges[i].first;
                        auto end = ranges[i].second;
                        if (!std::binary_search(start, end, candidate)) {
                            in_all = false;
                            break;
                        }
                    }
                    
                    if (in_all) {
                        threadResults[tid].push_back(candidate);
                    }
                }
            }
        }
    }
    
    // Combine results
    std::vector<int> result;
    size_t total_size = 0;
    for (const auto& vec : threadResults) {
        total_size += vec.size();
    }
    result.reserve(total_size);
    
    for (const auto& vec : threadResults) {
        result.insert(result.end(), vec.begin(), vec.end());
    }
    std::sort(result.begin(), result.end());
    return result;
}

/**
 * Implementation using the LeapfrogJoin algorithm for parallel processing
 */
std::vector<int> parallelLeapfrogJoin(const std::vector<std::vector<int>>& inputVectors, int numPartitions) {
    if (inputVectors.empty()) {
        return {};
    }
    if (inputVectors.size() == 1) {
        return inputVectors[0];
    }

    // Find global min and max
    int global_min = std::numeric_limits<int>::max();
    int global_max = std::numeric_limits<int>::min();
    
    for (const auto& vec : inputVectors) {
        if (!vec.empty()) {
            global_min = std::min(global_min, vec.front());
            global_max = std::max(global_max, vec.back());
        } else {
            return {};
        }
    }

    // Create a vector of partitioned ranges
    std::vector<std::pair<int, int>> ranges;
    int range_size = std::max(1, (global_max - global_min + 1) / numPartitions);
    
    for (int i = 0; i < numPartitions; ++i) {
        int start = global_min + i * range_size;
        int end = (i == numPartitions - 1) ? global_max + 1 : start + range_size;
        ranges.emplace_back(start, end);
    }

    std::cout << "Using parallel leapfrog join with " << ranges.size() << " partitions" << std::endl;
    
    std::vector<std::vector<int>> threadResults;
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            threadResults.resize(omp_get_num_threads());
        }

        int tid = omp_get_thread_num();

        #pragma omp for schedule(dynamic)
        for (size_t r = 0; r < ranges.size(); ++r) {
            int start_val = ranges[r].first;
            int end_val = ranges[r].second - 1; // -1 because end is exclusive
            
            // Create sub-vectors for this range
            std::vector<std::vector<int>> subVectors;
            bool skip_range = false;
            
            for (const auto& vec : inputVectors) {
                auto start_it = std::lower_bound(vec.begin(), vec.end(), start_val);
                auto end_it = std::upper_bound(start_it, vec.end(), end_val);
                
                // If any vector has no elements in this range, we can skip
                if (start_it == end_it) {
                    skip_range = true;
                    break;
                }
                
                // Create a copy of the sub-vector for this range
                std::vector<int> subVec(start_it, end_it);
                subVectors.push_back(subVec);
            }
            
            if (!skip_range) {
                // Use the LeapfrogJoin algorithm on this subset
                std::vector<int> rangeResult = leapfrogJoin(subVectors);
                threadResults[tid].insert(threadResults[tid].end(), rangeResult.begin(), rangeResult.end());
            }
        }
    }
    
    // Combine results
    std::vector<int> result;
    size_t total_size = 0;
    for (const auto& vec : threadResults) {
        total_size += vec.size();
    }
    result.reserve(total_size);
    
    for (const auto& vec : threadResults) {
        result.insert(result.end(), vec.begin(), vec.end());
    }
    
    // Results are already sorted within each thread's partition, 
    // but we need to merge them
    std::sort(result.begin(), result.end());
    return result;
}

/**
 * Implementation with work stealing for better load balancing
 */
std::vector<int> workStealingIntersection(const std::vector<std::vector<int>>& inputVectors, int numPartitions) {
    if (inputVectors.empty()) {
        return {};
    }
    if (inputVectors.size() == 1) {
        return inputVectors[0];
    }

    // Find the smallest vector to iterate over
    int smallestIndex = -1;
    size_t minSize = std::numeric_limits<size_t>::max();
    for (size_t i = 0; i < inputVectors.size(); ++i) {
        if (!inputVectors[i].empty()) {
            if (smallestIndex == -1 || inputVectors[i].size() < minSize) {
                minSize = inputVectors[i].size();
                smallestIndex = static_cast<int>(i);
            }
        } else {
            return {};
        }
    }
    if (smallestIndex == -1) {
        return {};
    }

    const std::vector<int>& base = inputVectors[smallestIndex];
    
    // Create task chunks for better load balancing
    const int chunk_size = std::max(1, static_cast<int>(base.size()) / (numPartitions * 4));
    std::vector<std::pair<int, int>> tasks; // (start, end) indices
    
    for (int start = 0; start < static_cast<int>(base.size()); start += chunk_size) {
        int end = std::min(start + chunk_size, static_cast<int>(base.size()));
        tasks.emplace_back(start, end);
    }
    
    std::cout << "Using work stealing with " << tasks.size() << " chunks (chunk size: " << chunk_size << ")" << std::endl;
    
    std::vector<std::vector<int>> threadResults;
    std::atomic<size_t> nextTaskIdx(0);
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        
        #pragma omp single
        {
            threadResults.resize(num_threads);
        }
        
        while (true) {
            // Get next task atomically
            size_t taskIdx = nextTaskIdx.fetch_add(1);
            if (taskIdx >= tasks.size()) {
                break;
            }
            
            int start = tasks[taskIdx].first;
            int end = tasks[taskIdx].second;
            
            // Process this chunk
            for (int idx = start; idx < end; ++idx) {
                int candidate = base[idx];
                bool in_all = true;
                
                for (size_t j = 0; j < inputVectors.size(); ++j) {
                    if (j == static_cast<size_t>(smallestIndex)) {
                        continue;
                    }
                    
                    if (!std::binary_search(inputVectors[j].begin(), inputVectors[j].end(), candidate)) {
                        in_all = false;
                        break;
                    }
                }
                
                if (in_all) {
                    threadResults[tid].push_back(candidate);
                }
            }
        }
    }
    
    // Combine results
    std::vector<int> result;
    size_t total_size = 0;
    for (const auto& vec : threadResults) {
        total_size += vec.size();
    }
    result.reserve(total_size);
    
    for (const auto& vec : threadResults) {
        result.insert(result.end(), vec.begin(), vec.end());
    }
    
    std::sort(result.begin(), result.end());
    return result;
}