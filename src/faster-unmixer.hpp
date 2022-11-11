#pragma once

#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fastunmixer {

constexpr auto root_node_name = "##ROOT##";
constexpr auto unset_node_name = "##UNSET##";
constexpr auto NO_DOWNSTREAM_NEIGHBOUR = std::numeric_limits<size_t>::max();

// Each SampleNode correspond to a sample, specified by a name and (x,y)
// location, as well as a portion of the watershed. This portion of the
// watershed flows into a downstream node and receives flow from upstream nodes
struct SampleNode {
  using name_t = std::string;
  // Sample name
  name_t name;
  // Sample location
  int64_t x = std::numeric_limits<int64_t>::min();
  int64_t y = std::numeric_limits<int64_t>::min();
  // Sample's water flows downstream into this node
  name_t downstream_node;
  // Sample receives water from these upstream nodes
  std::vector<std::string> upstream_nodes;
  // Area that uniquely contributes to this sample
  int64_t area = 0;
  // Total upstream area including `area`
  int64_t total_upstream_area = 0;
  // Label used in the labels image output
  int64_t label = std::numeric_limits<size_t>::max();
};

using NamePair = std::pair<SampleNode::name_t, SampleNode::name_t>;

// Used to store graph edges
struct NamePairHash {
  std::size_t operator()(const NamePair& sp) const {
    auto a_hash = std::hash<SampleNode::name_t>()(sp.first);
    auto b_hash = std::hash<SampleNode::name_t>()(sp.second);
    // Boost hash combine function: https://stackoverflow.com/a/27952689/752843
    return a_hash ^ (b_hash + 0x9e3779b9 + (a_hash << 6) + (a_hash >> 2));
  }
};

using SampleGraph = std::unordered_map<SampleNode::name_t, SampleNode>;
using NeighborsToBorderLength = std::unordered_map<NamePair, int64_t, NamePairHash>;

std::pair<SampleGraph, NeighborsToBorderLength> faster_unmixer(const std::string& data_dir);

}