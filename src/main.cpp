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
#include "generate_data.hpp"

using namespace std;

void printUsage(const string & exeName) {
    cerr << "Usage: " << exeName << " <datafile.txt> op [numPartitions]" << endl;
    cerr << "  op: Mode flag to run the optimized (naive one-by-one) parallel join." << endl;
    cerr << "      Options: op, range, critical" << endl;
    cerr << "  [numPartitions]: Optional. Number of partitions to use for splitting the key range." << endl;
    cerr << "                   (If omitted, defaults to the number of OpenMP threads.)" << endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 4) {
        printUsage(argv[0]);
        return 1;
    }

    string filename = argv[1];
    string mode = argv[2];
 
    // Determine the number of threads/partitions
    int numPartitions = 0; // Default to 0, will be set to max threads if needed
    if (argc == 4) {
        try {
            numPartitions = stoi(argv[3]);
            if (numPartitions <= 0) {
                cout << "Non-positive number of partitions provided. Defaulting to max threads." << endl;
                numPartitions = 0; // Signal to use default max threads
            }
        } catch (const std::exception& e) {
            cerr << "Error parsing numPartitions: " << e.what() << ". Defaulting to max threads." << endl;
            numPartitions = 0; // Signal to use default max threads
        }
    }

    if (numPartitions == 0) {
        numPartitions = omp_get_max_threads(); // Use max threads if default or invalid input
        cout << "Using default number of partitions (max threads): " << numPartitions << endl;
    } else {
        cout << "Requesting " << numPartitions << " partitions/threads." << endl;
    }

    // --- Explicitly set the number of threads for OpenMP ---
    omp_set_num_threads(numPartitions);

    // --- Load Data ---
    cout << "Loading test data from: " << filename << "..." << endl;
    vector<vector<int>> inputVectors = loadTestData(filename); // Call the function from generate_data.hpp/cpp
 
    // --- Add Data Verification ---
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
        // Optional: Check if data is sorted
        /*
        if (!inputVectors[i].empty() && !std::is_sorted(inputVectors[i].begin(), inputVectors[i].end())) {
            cerr << "Error: Vector " << i << " is not sorted!" << endl;
        }
        */
    }
    // --- End Data Verification ---
 
 
    cout << "\n--- Running Parallel Intersection ---" << endl;
    cout << "Mode: " << mode << endl;
    // Verify actual number of threads being used in the upcoming parallel region
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
 
     // --- Select and run the chosen parallel method ---
    if (mode == "op") {
        result = parallelLinearScanIntersection(inputVectors, numPartitions);
    } else if (mode == "range") {
        result = rangePartition(inputVectors, numPartitions);
    } else if (mode == "critical") {
        result = parallelCriticalIntersection(inputVectors, numPartitions);
    }
 
    auto end = chrono::steady_clock::now();
    chrono::duration<double> duration = end - start;

    cout << "\n--- Results ---" << endl;
    cout << "Result size: " << result.size() << std::endl;
    cout << "Computation time: " << duration.count() << " seconds" << std::endl;
    // Report the number of partitions/threads actually used based on omp_set_num_threads
    cout << "Threads used (set by omp_set_num_threads): " << numPartitions << std::endl;
 
 
    return 0;
 }
 