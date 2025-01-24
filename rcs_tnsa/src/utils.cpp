#include "utils.h"  // NOLINT(build/include)

#include <fstream>
#include <iostream>
#include <sstream>

std::vector<std::vector<size_t>> utils::ReadGroups(const std::string filename) {
  auto stream = std::ifstream(filename);
  std::string line;
  std::vector<std::vector<size_t>> grouped_edge_nums;
  while (std::getline(stream, line)) {
    auto line_stream = std::istringstream(line);
    std::vector<std::string> tokens;
    std::string token;
    while (getline(line_stream, token, ' ')) {
      tokens.push_back(token);
    }
    std::vector<size_t> group;
    for (size_t i = 0; i < tokens.size(); i++) {
      group.push_back(std::stoi(tokens[i]));
    }
    grouped_edge_nums.push_back(group);
  }

  return grouped_edge_nums;
}
