#ifndef GENERATE_DATA_HPP
#define GENERATE_DATA_HPP

#include <vector>
#include <string>

/**
 * Loads test data from the specified file.
 * Each line in the file represents a sorted integer vector (space-separated).
 */
std::vector<std::vector<int>> loadTestData(const std::string& filename);

#endif // GENERATE_DATA_HPP