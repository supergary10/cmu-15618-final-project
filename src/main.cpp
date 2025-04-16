#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <omp.h>

#include "LeapfrogJoin.hpp"   // (Not used in the naive join but kept for reference)
#include "generate_data.hpp"  // Data loader

using namespace std;

/**
 * Naive Parallel Join:
 *
 * This function chooses the smallest sorted vector as the base. For each element in the base,
 * it linearly scans every other input vector (which are also sorted) to check if the element exists.
 * The comparisons are done one-by-one (without any binary search skipping).
 *
 * The loop over the base elements is parallelized with OpenMP; each thread writes to its own local result.
 *
 * Complexity: In the worst-case, O(N_base * k), where k is the number of vectors.
 */
std::vector<int> naiveParallelJoin(const std::vector<std::vector<int>>& vectors) {
    if (vectors.empty())
        return {};

    // Select the smallest vector as the base for comparison.
    int smallestIndex = 0;
    for (size_t i = 1; i < vectors.size(); i++) {
        if (vectors[i].size() < vectors[smallestIndex].size())
            smallestIndex = i;
    }
    const std::vector<int>& base = vectors[smallestIndex];
    std::vector<int> result;

    // Prepare a local results container for each thread.
    int numThreads = omp_get_max_threads();
    std::vector<std::vector<int>> threadResults(numThreads);

    // Parallel loop over each element in the base vector.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(base.size()); i++) {
        int candidate = base[i];
        bool inAll = true;
        // Compare candidate with every other vector using a simple linear scan.
        for (size_t j = 0; j < vectors.size(); j++) {
            if (j == static_cast<size_t>(smallestIndex))
                continue;
            bool found = false;
            // Since the vectors are sorted, scan one by one.
            for (int value : vectors[j]) {
                if (value == candidate) {
                    found = true;
                    break;
                }
                else if (value > candidate) {
                    // Since the vector is sorted, no need to check further.
                    break;
                }
            }
            if (!found) {
                inAll = false;
                break;
            }
        }
        if (inAll) {
            int tid = omp_get_thread_num();
            threadResults[tid].push_back(candidate);
        }
    }

    // Merge all thread-local results.
    for (const auto& localVec : threadResults) {
        result.insert(result.end(), localVec.begin(), localVec.end());
    }
    sort(result.begin(), result.end());
    return result;
}

void printUsage(const string & exeName) {
    cerr << "Usage: " << exeName << " <datafile.txt> op [numPartitions]" << endl;
    cerr << "  op: Mode flag to run the optimized (naive one-by-one) parallel join." << endl;
    cerr << "  [numPartitions]: Optional. Number of partitions to use for splitting the key range." << endl;
    cerr << "                   (If omitted, defaults to the number of OpenMP threads.)" << endl;
}

/**
 * Naive Parallel Join via Range Partitioning:
 *
 * For demonstration, here is an alternative approach that partitions the global key range and then
 * performs the naive one-by-one comparison on each partition. (The idea is similar to the earlier leapfrog"
 * parallel version but without the leapfrog seek; instead, each partition uses the naive join on its sub-vector.)
 */
std::vector<int> naiveParallelJoinRangePartitioning(const std::vector<std::vector<int>>& inputVectors, int numPartitions) {
    if (inputVectors.empty())
         return {};

    // Determine the global minimum and maximum.
    int globalMin = inputVectors[0].front();
    int globalMax = inputVectors[0].back();
    for (const auto & vec : inputVectors) {
         if (!vec.empty()) {
             globalMin = std::min(globalMin, vec.front());
             globalMax = std::max(globalMax, vec.back());
         }
    }
    int range = globalMax - globalMin + 1;
    int partitionSize = range / numPartitions;
    if (range % numPartitions != 0)
         partitionSize++;  // Ensure full coverage.

    // Prepare a container for partition results.
    vector<vector<int>> partitionResults(numPartitions);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < numPartitions; i++) {
         int partitionStart = globalMin + i * partitionSize;
         int partitionEnd = (i == numPartitions - 1) ? globalMax : (partitionStart + partitionSize - 1);
         
         // For each input vector, extract the sub-vector falling into the partition.
         vector<vector<int>> subVectors;
         bool validPartition = true;
         for (const auto & vec : inputVectors) {
              auto lower = std::lower_bound(vec.begin(), vec.end(), partitionStart);
              auto upper = std::upper_bound(vec.begin(), vec.end(), partitionEnd);
              vector<int> subVec(lower, upper);
              // If any sub-vector is empty, then there is no intersection in this partition.
              if (subVec.empty()) {
                  validPartition = false;
                  break;
              }
              subVectors.push_back(subVec);
         }
         if (validPartition) {
             // Call the naive parallel join on the sub-vectors.
             partitionResults[i] = naiveParallelJoin(subVectors);
         } else {
             partitionResults[i] = vector<int>(); // Empty partition.
         }
    }

    // Merge partition results.
    vector<int> finalResult;
    for (const auto & part : partitionResults) {
         finalResult.insert(finalResult.end(), part.begin(), part.end());
    }
    sort(finalResult.begin(), finalResult.end());
    return finalResult;
}

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 4) {
         printUsage(argv[0]);
         return 1;
    }

    string filename = argv[1];
    string mode = argv[2];
    if (mode != "op") {
         printUsage(argv[0]);
         return 1;
    }

    int numPartitions = omp_get_max_threads();  // Default partitions equals available threads.
    if (argc == 4) {
         numPartitions = stoi(argv[3]);
         if (numPartitions <= 0) {
             cerr << "Invalid number of partitions provided. Must be > 0." << endl;
             return 1;
         }
    }

    // Load test data from file.
    vector<vector<int>> inputVectors = loadTestData(filename);
    if (inputVectors.empty()){
         cerr << "Data loading failed or file is empty!" << endl;
         return 1;
    }

    // We now call the alternative join. Choose one method:
    // Method 1: Naive parallel join on the base vector
    // vector<int> result = naiveParallelJoin(inputVectors);
    // Method 2: Naive parallel join via range partitioning
    auto start = chrono::steady_clock::now();
    vector<int> result = naiveParallelJoinRangePartitioning(inputVectors, numPartitions);
    auto end = chrono::steady_clock::now();
    chrono::duration<double> duration = end - start;
    
    cout << "Optimized (naive) Parallel Join found " << result.size() 
         << " elements in " << duration.count() << " seconds." << endl;
    cout << "Used " << numPartitions << " partitions." << endl;

    return 0;
}