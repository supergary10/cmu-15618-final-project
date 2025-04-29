#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <numeric>
#include <omp.h>
#include <algorithm>

#include "intersection.hpp"
#include "enhanced_intersection.hpp"
#include "generate_data.hpp"

using namespace std;

void printUsage(const string & exeName) {
    cerr << "Usage: " << exeName << " <datafile.txt> algorithm [numPartitions]" << endl;
    cerr << "  algorithm: Algorithm to use for intersection." << endl;
    cerr << "      Options: " << endl;
    cerr << "          op       - Basic parallel linear scan" << endl;
    cerr << "          range    - Range partitioning" << endl;
    cerr << "          critical - Critical section approach" << endl; 
    cerr << "          binary   - Binary search optimization" << endl;
    cerr << "          adaptive - Adaptive range partitioning" << endl;
    cerr << "          leapfrog - Parallel leapfrog join" << endl;
    cerr << "          worksteal - Work stealing approach" << endl;
    cerr << "  [numPartitions]: Optional. Number of partitions/threads to use." << endl;
    cerr << "                   (If omitted, defaults to the number of OpenMP threads.)" << endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 4) {
        printUsage(argv[0]);
        return 1;
    }

    string filename = argv[1];
    string algorithm = argv[2];
 
    int numPartitions = 0;
    if (argc == 4) {
        try {
            numPartitions = stoi(argv[3]);
            if (numPartitions <= 0) {
                cout << "Non-positive number of partitions provided. Defaulting to max threads." << endl;
                numPartitions = 0;
            }
        } catch (const std::exception& e) {
            cerr << "Error parsing numPartitions: " << e.what() << ". Defaulting to max threads." << endl;
            numPartitions = 0;
        }
    }

    if (numPartitions == 0) {
        numPartitions = omp_get_max_threads();
        cout << "Using default number of partitions (max threads): " << numPartitions << endl;
    } else {
        cout << "Requesting " << numPartitions << " partitions/threads." << endl;
    }

    omp_set_num_threads(numPartitions);

    cout << "Loading test data from: " << filename << "..." << endl;
    vector<vector<int>> inputVectors = loadTestData(filename);

    if (inputVectors.empty()){
        cerr << "Error: Data loading failed or returned empty vector container!" << endl;
        return 1;
    }
    cout << "Data loaded successfully. Number of vectors: " << inputVectors.size() << endl;
    bool anyVectorEmpty = false;
    for(size_t i = 0; i < inputVectors.size(); ++i) {
        cout << "  Vector " << i << " size: " << inputVectors[i].size() << endl;
        if (inputVectors[i].empty()) {
            cerr << "Warning: Vector " << i << " is empty. Intersection will be empty." << endl;
            anyVectorEmpty = true;
        }
    }
 
    cout << "\n--- Running Parallel Intersection ---" << endl;
    cout << "Algorithm: " << algorithm << endl;

    #pragma omp parallel
    {
        #pragma omp single
        {
            cout << "(Max hardware threads: " << omp_get_max_threads()
                << ", Actual threads for this run: " << omp_get_num_threads() << ")" << endl;
        }
    }

    auto start = chrono::steady_clock::now();
    vector<int> result;
 
    if (algorithm == "op") {
        result = parallelLinearScanIntersection(inputVectors, numPartitions);
    } else if (algorithm == "range") {
        result = rangePartition(inputVectors, numPartitions);
    } else if (algorithm == "critical") {
        result = parallelCriticalIntersection(inputVectors, numPartitions);
    } else if (algorithm == "binary") {
        result = binarySearchIntersection(inputVectors, numPartitions);
    } else if (algorithm == "adaptive") {
        result = adaptiveRangePartition(inputVectors, numPartitions);
    } else if (algorithm == "leapfrog") {
        result = parallelLeapfrogJoin(inputVectors, numPartitions);
    } else if (algorithm == "worksteal") {
        result = workStealingIntersection(inputVectors, numPartitions);
    } else {
        cerr << "Unknown algorithm: " << algorithm << endl;
        printUsage(argv[0]);
        return 1;
    }
 
    auto end = chrono::steady_clock::now();
    chrono::duration<double> duration = end - start;

    cout << "\n--- Results ---" << endl;
    cout << "Result size: " << result.size() << std::endl;
    cout << "Computation time: " << duration.count() << " seconds" << std::endl;
    cout << "Threads used (set by omp_set_num_threads): " << numPartitions << std::endl;
 
    return 0;
}