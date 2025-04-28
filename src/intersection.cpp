#include "intersection.hpp"
#include <numeric>
#include <algorithm>
#include <limits>
#include <omp.h>
#include <iostream>
#include <cmath>

std::vector<int> parallelLinearScanIntersection(const std::vector<std::vector<int>>& inputVectors, int numPartitions) {
    if (inputVectors.empty()) {
        return {};
    }
    if (inputVectors.size() == 1) {
        return inputVectors[0];
    }

    // Find the index of the smallest non-empty vector
    int smallestIndex = -1;
    size_t minSize = std::numeric_limits<size_t>::max();
    for (size_t i = 0; i < inputVectors.size(); ++i) {
        if (!inputVectors[i].empty()) {
            if (smallestIndex == -1 || inputVectors[i].size() < minSize) {
                minSize = inputVectors[i].size();
                smallestIndex = static_cast<int>(i);
            }
        } else {
            return {}; // Intersection with empty set is empty
        }
    }
    if (smallestIndex == -1) {
        return {}; // All vectors were empty
    }

    std::cout << "Smallest vector index: " << smallestIndex << std::endl;
    const std::vector<int>& base = inputVectors[smallestIndex];
    std::vector<int> result;

    std::vector<std::vector<int>> threadResults;

    std::cout << "Starting parallel processing with " << numPartitions << " partitions." << std::endl;
    #pragma omp parallel // Start parallel region
    {
        // Use #pragma omp single to have only one thread resize the vector
        // based on the actual number of threads in this parallel region.
        #pragma omp single
        {
            threadResults.resize(omp_get_num_threads());
        } // Implicit barrier ensures all threads see the resized vector

        int tid = omp_get_thread_num(); // Get the actual thread ID

        // The 'for' directive distributes loop iterations among threads
        #pragma omp for schedule(dynamic)
        for (int idx = 0; idx < static_cast<int>(base.size()); ++idx) {
            int candidate = base[idx];
            bool in_all = true;

            // Check candidate against all *other* vectors
            for (size_t j = 0; j < inputVectors.size(); ++j) {
                if (j == static_cast<size_t>(smallestIndex)) {
                    continue;
                }

                bool found = false;
                for (int value : inputVectors[j]) {
                    if (value == candidate) {
                        found = true;
                        break;
                    }
                    if (value > candidate) {
                        break;
                    }
                }

                if (!found) {
                    in_all = false;
                    break;
                }
            }

            if (in_all) {
                threadResults[tid].push_back(candidate);
            }
        } // End parallel for loop

    } // End parallel region
    
    std::cout << "Finished parallel processing." << std::endl;
    for (const auto& localVec : threadResults) {
        result.insert(result.end(), localVec.begin(), localVec.end());
    }
    sort(result.begin(), result.end());
    return result;
}

std::vector<int> rangePartition(const std::vector<std::vector<int>>& inputVectors, int numPartitions) {
    if (inputVectors.empty()) {
        return {};
    }
    if (inputVectors.size() == 1) {
        return inputVectors[0];
    }

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

    std::vector<int> result;
    std::vector<std::vector<int>> threadResults;

    #pragma omp parallel
    {
        int actual_num_threads = omp_get_num_threads();
        #pragma omp single
        {
            threadResults.resize(actual_num_threads);
        }

        int tid = omp_get_thread_num();

        double range_size = static_cast<double>(global_max) - global_min + 1.0;
        double partition_size = std::max(1.0, std::ceil(range_size / actual_num_threads));

        long long partition_start_val_ll = static_cast<long long>(global_min) + static_cast<long long>(tid) * static_cast<long long>(partition_size);
        long long partition_end_val_ll = partition_start_val_ll + static_cast<long long>(partition_size);

        int partition_start_val = static_cast<int>(std::max((long long)std::numeric_limits<int>::min(), partition_start_val_ll));
        int partition_end_val_inclusive = static_cast<int>(std::min((long long)global_max, partition_end_val_ll - 1));

        if (partition_end_val_inclusive < partition_start_val) {
            partition_start_val = partition_end_val_inclusive + 1;
        }

        std::vector<std::vector<int>::const_iterator> starts(inputVectors.size());
        std::vector<std::vector<int>::const_iterator> ends(inputVectors.size());
        bool possible_in_partition = true;
        for (size_t i = 0; i < inputVectors.size(); ++i) {
            starts[i] = std::lower_bound(inputVectors[i].begin(), inputVectors[i].end(), partition_start_val);
            ends[i] = std::upper_bound(starts[i], inputVectors[i].end(), partition_end_val_inclusive);

            if (starts[i] == inputVectors[i].end() || *starts[i] > partition_end_val_inclusive) {
                possible_in_partition = false;
                break;
            }
        }

        if (possible_in_partition) {
            int local_smallest_idx = -1;
            size_t local_min_size = std::numeric_limits<size_t>::max();
            for(size_t i = 0; i < inputVectors.size(); ++i) {
                 size_t current_size = std::distance(starts[i], ends[i]);
                 if (current_size > 0) {
                      if(local_smallest_idx == -1 || current_size < local_min_size) {
                           local_min_size = current_size;
                           local_smallest_idx = static_cast<int>(i);
                      }
                 } else {
                      possible_in_partition = false;
                      break;
                 }
            }

            if (possible_in_partition && local_smallest_idx != -1) {
                 for (auto it_base = starts[local_smallest_idx]; it_base != ends[local_smallest_idx]; ++it_base) {
                     int candidate = *it_base;
                     bool in_all_local = true;

                     for (size_t j = 0; j < inputVectors.size(); ++j) {
                         if (static_cast<int>(j) == local_smallest_idx) continue;

                         bool found_local = false;
                         for (auto it_check = starts[j]; it_check != ends[j]; ++it_check) {
                             if (*it_check == candidate) {
                                 found_local = true;
                                 break;
                             }
                             if (*it_check > candidate) {
                                 break;
                             }
                         }
                         if (!found_local) {
                             in_all_local = false;
                             break;
                         }
                     }

                     if (in_all_local) {
                         threadResults[tid].push_back(candidate);
                     }
                 }
            }
        }
    }

    for (const auto& localVec : threadResults) {
        result.insert(result.end(), localVec.begin(), localVec.end());
    }
    sort(result.begin(), result.end());
    return result;
}

std::vector<int> parallelCriticalIntersection(const std::vector<std::vector<int>>& inputVectors, int numPartitions) {
    if (inputVectors.empty()) {
        return {};
    }
    if (inputVectors.size() == 1) {
        return inputVectors[0];
    }

    int smallestIndex = -1;
    size_t minSize = std::numeric_limits<size_t>::max();
    for (size_t i = 0; i < inputVectors.size(); ++i) {
        if (!inputVectors[i].empty()) {
            if (smallestIndex == -1 || inputVectors[i].size() < minSize) {
                minSize = inputVectors[i].size();
                smallestIndex = static_cast<int>(i);
            }
        } else {
            return {}; // Intersection with empty set is empty
        }
    }
    if (smallestIndex == -1) {
        return {}; // All vectors were empty
    }

    std::cout << "Smallest vector index: " << smallestIndex << std::endl;
    const std::vector<int>& base = inputVectors[smallestIndex];
    std::vector<int> result;
    
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic)
        for (int idx = 0; idx < static_cast<int>(base.size()); ++idx) {
            int candidate = base[idx];
            bool in_all = true;

            for (size_t j = 0; j < inputVectors.size(); ++j) {
                if (j == static_cast<size_t>(smallestIndex)) continue;

                bool found = false;
                for (int value : inputVectors[j]) {
                    if (value == candidate) {
                        found = true;
                        break;
                    }
                    if (value > candidate) {
                        break;
                    }
                }
                if (!found) {
                    in_all = false;
                    break;
                }
            }

            if (in_all) {
                #pragma omp critical
                {
                    result.push_back(candidate);
                }
            }
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}