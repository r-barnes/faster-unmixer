#include "faster-unmixer.hpp"
#include "gdal/gdal_priv.h"

#include <richdem/common/Array2D.hpp>
#include <richdem/common/grid_cell.hpp>
#include <richdem/common/iterators.hpp>
#include <richdem/depressions/Barnes2014.hpp>
#include <richdem/flowmet/d8_flowdirs.hpp>
#include <richdem/methods/flow_accumulation.hpp>
#include <richdem/misc/conversion.hpp>
#include <iostream>

#include <fstream>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace rd = richdem;

namespace fastunmixer {

namespace detail {

struct PairHash {
  std::size_t operator()(const std::pair<uint32_t, uint32_t>& sp) const {
    // Boost hash combine function: https://stackoverflow.com/a/27952689/752843
    return sp.first ^ (sp.second + 0x9e3779b9 + (sp.first << 6) + (sp.first >> 2));
  }
};

struct SampleData {
  std::string name = unset_node_name;
  double x = std::numeric_limits<double>::quiet_NaN();
  double y = std::numeric_limits<double>::quiet_NaN();
};

// Each NativeSampleNode correspond to a sample, specified by a name and (x,y)
// location, as well as a portion of the watershed. This portion of the
// watershed flows into a downstream node and receives flow from upstream nodes
struct NativeSampleNode {
  using name_t = std::string;
  // Sample data
  SampleData data;
  // Sample's water flows downstream into this node
  size_t downstream_node = NO_DOWNSTREAM_NEIGHBOUR;
  // Sample receives water from these upstream nodes
  std::vector<std::string> upstream_nodes;
  // Area that uniquely contributes to this sample
  int64_t area = 0;
  // Total upstream area including `area`
  int64_t total_upstream_area = 0;

  // Makes a root node
  static NativeSampleNode make_root_node() {
    NativeSampleNode temp;
    temp.data.name = root_node_name;
    return temp;
  }

  static NativeSampleNode make_w_downstream_and_sample(
    const size_t &downstream_node,
    const SampleData &sample_data
  ){
    NativeSampleNode temp;
    temp.downstream_node = downstream_node;
    temp.data = sample_data;
    return temp;
  }
};

using NeighborsToBorderLength = std::unordered_map<std::pair<uint32_t, uint32_t>, int64_t, PairHash>;

}


std::vector<detail::SampleData> get_sample_data(const std::string &sample_filename){
  std::vector<detail::SampleData> sample_data;

  {
    std::ifstream fin(sample_filename);
    std::string temp;
    std::getline(fin, temp);
    if(!temp.starts_with("Sample.Code x_coordinate y_coordinate")){
      throw std::runtime_error("'" + sample_filename + "' header must start with 'Sample.Code x_coordinate y_coordinate'!");
    }
    while(std::getline(fin, temp)){
      std::stringstream ss(temp);
      std::string name;
      double sx;
      double sy;
      ss>>name>>sx>>sy;
      sample_data.push_back(detail::SampleData{name, sx, sy});
    }
  }

  return sample_data;
}

void calculate_total_upstream_areas(std::vector<detail::NativeSampleNode> &sample_graph){
  // Count how many upstream neighbours we're waiting on
  std::vector<size_t> deps(sample_graph.size());
  for(size_t i = 0; i < sample_graph.size(); i++){
    deps.at(i) = sample_graph.at(i).upstream_nodes.size();
  }

  // Find cells with no upstream neighbours
  std::queue<size_t> q;
  for(size_t i = 0; i < sample_graph.size(); i++){
    if(deps.at(i)==0){
      q.emplace(i);
    }
  }

  while(!q.empty()){
    const auto c = q.front();
    q.pop();

    auto& self = sample_graph.at(c);

    // Add my own area
    self.total_upstream_area += self.area;

    if(self.downstream_node==NO_DOWNSTREAM_NEIGHBOUR){
      continue;
    }

    // Add my area to downstream neighbour
    sample_graph.at(self.downstream_node).total_upstream_area += self.total_upstream_area;

    // My downstream node no longer depends on me
    deps.at(self.downstream_node)--;

    if(deps.at(self.downstream_node)==0){
      q.emplace(self.downstream_node);
    }
  }
}

std::pair<std::vector<detail::NativeSampleNode>, detail::NeighborsToBorderLength> faster_unmixer_internal(const std::string& flowdirs_filename, const std::string& sample_filename){
  // Load data
  rd::Array2D<rd::d8_flowdir_t> arc_flowdirs(flowdirs_filename);

  // Convert raw flowdirs to RichDEM flowdirs
  auto flowdirs = rd::Array2D<rd::d8_flowdir_t>::make_from_template(arc_flowdirs);
  rd::convert_arc_flowdirs_to_richdem_d8(arc_flowdirs, flowdirs);
  flowdirs.saveGDAL("rd_flowdirs.tif");

  // Check that boundary conditions are correct
  iterate_2d(flowdirs, [&](const auto x, const auto y){
    // if x,y is on the boundary and the node is not a sink raise an exception:
    if(
      (x==0 || y==0 || x==flowdirs.width()-1 || y==flowdirs.height()-1) &&
      flowdirs(x,y)!=rd::NO_FLOW
    ){
      throw std::runtime_error("Boundary condition violated at (" + std::to_string(x) + ", " + std::to_string(y) + ")! Expected a sink node, but got a flow direction!");
    }
  });

  // Get geotransform info from raster
  // Extract GDAL origin (upper left) + pixel widths
  const auto originX = flowdirs.geotransform[0];
  const auto originY = flowdirs.geotransform[3];
  const auto pixelWidth = flowdirs.geotransform[1];
  const auto pixelHeight = flowdirs.geotransform[5];

  // Get sample locations and put them in a set using flat-indexing for fast
  // look-up
  std::unordered_map<uint32_t, detail::SampleData> sample_locs;
  for(const auto &sample: get_sample_data(sample_filename)){
    // Get x, y indices relative to upper left
    const auto x_ul = static_cast<int64_t>(std::round((sample.x-originX)/pixelWidth));
    const auto y_ul = static_cast<int64_t>(std::round((sample.y-originY)/pixelHeight));

    sample_locs[flowdirs.xyToI(x_ul, y_ul)] = sample;
  }

  // Graph of how the samples are connected together.
  std::vector<detail::NativeSampleNode> sample_parent_graph;

  // Identify cells which do not flow to anywhere. These are the start of our
  // region-identifying procedure
  std::queue<rd::GridCell> q;
  iterate_2d(flowdirs, [&](const auto x, const auto y){
    if(flowdirs(x,y)==rd::NO_FLOW){
      q.emplace(x,y);
    }
  });

  // Indicates that the cell doesn't correspond to any label
  constexpr auto NO_LABEL = 0;
  auto sample_label = rd::Array2D<uint32_t>::make_from_template(flowdirs, NO_LABEL);
  sample_label.setNoData(NO_LABEL);
  sample_parent_graph.push_back(detail::NativeSampleNode::make_root_node());

  // Iterate in a wave from all the flow endpoints to the headwaters, labeling
  // cells as we go.
  while(!q.empty()){
    const auto c = q.front();
    q.pop();

    // We have identified a new station!
    if(sample_locs.count(flowdirs.xyToI(c.x,c.y))!=0){
      const auto &data = sample_locs.at(flowdirs.xyToI(c.x,c.y));
      // Generate a new label for the sample
      const auto my_new_label = sample_parent_graph.size();
      // The current label will become the parent label
      const auto my_current_label = sample_label(c.x,c.y);
      auto& parent = sample_parent_graph.at(my_current_label);
      parent.upstream_nodes.push_back(data.name);
      sample_parent_graph.push_back(
        detail::NativeSampleNode::make_w_downstream_and_sample(my_current_label, data)
      );
      // Update the sample's label
      sample_label(c.x,c.y) = my_new_label;
    }

    const auto my_label = sample_label(c.x,c.y);
    sample_parent_graph.at(my_label).area++;

    // Loop over all my neighbours
    for(int n=1;n<=8;n++){
      const int nx = c.x + rd::d8x[n];
      const int ny = c.y + rd::d8y[n];
      // If cell flows into me, it gets my label. If the cell happens to have a
      // sample, then this label will be overwritten when the cell is popped
      // from the queue. D8 means that upstream cells are only added to the
      // queue once.
      if(flowdirs.inGrid(nx,ny) && flowdirs(nx,ny)==rd::d8_inverse[n]){
        sample_label(nx,ny) = sample_label(c.x,c.y);
        q.emplace(nx,ny);
      }
    }
  }

  // Save regions output
  sample_label.saveGDAL("labels.tif");
  {
    std::ofstream fout("labels.csv");
    for(size_t i = 0; i < sample_parent_graph.size(); i++){
      fout<<i<<",\""<<sample_parent_graph.at(i).data.name<<"\""<<std::endl;
    }
  }

  // Sanity check
  {
    size_t total_area=0;
    for(const auto &x: sample_parent_graph){
      total_area+=x.area;
    }
    if(total_area!=flowdirs.size()){
      throw std::runtime_error("Total area in graph " + std::to_string(total_area) + " does not equal total area in DEM " + std::to_string(flowdirs.size()) + "!");
    }
  }

  detail::NeighborsToBorderLength adjacency_graph;

  // Get border lengths between adjacent sample regions
  iterate_2d(sample_label, [&](const auto x, const auto y){
    const auto my_label = sample_label(x, y);
    if(my_label == sample_label.noData()){
      return;
    }
    for(int n=1;n<=8;n++){
      const int nx = x + rd::d8x[n];
      const int ny = y + rd::d8y[n];
      if(sample_label.inGrid(nx,ny)){
        const auto n_label = sample_label(nx, ny);
        // Less-than comparison creates a preferred ordering to prevent
        // double-counting and also prevents self-loops
        if(n_label != sample_label.noData() && my_label < n_label){
          adjacency_graph[{my_label,n_label}]++;
        }
      }
    }
  });

  calculate_total_upstream_areas(sample_parent_graph);

  std::ofstream fout_sg("sample_graph.dot");
  fout_sg<<"# dot -Tpng sample_graph.dot -o sample_graph.png"<<std::endl;
  fout_sg<<"digraph sample_graph {"<<std::endl;
  for(size_t i=0;i<sample_parent_graph.size();i++){
    const auto& self = sample_parent_graph.at(i);
    fout_sg<<"    "<<i<<" [label=\""<<self.data.name<<"\\n"<<self.area<<"\"];"<<std::endl;
    fout_sg<<"    "<<i<<" -> "<<self.downstream_node<<";"<<std::endl;
  }
  fout_sg<<"}"<<std::endl;

  return {sample_parent_graph, adjacency_graph};
}

std::pair<SampleGraph, NeighborsToBorderLength> faster_unmixer(const std::string& flowdirs_filename, const std::string& sample_filename){
  const auto& [sample_parent_graph, adjacency_graph_internal] = faster_unmixer_internal(flowdirs_filename, sample_filename);

  NeighborsToBorderLength adjacency_graph_external;
  for(auto &x: adjacency_graph_internal){
    auto nodea = sample_parent_graph.at(x.first.first).data.name;
    auto nodeb = sample_parent_graph.at(x.first.second).data.name;
    if(nodea > nodeb){
      std::swap(nodea, nodeb);
    }
    adjacency_graph_external[{nodea, nodeb}] = x.second;
  }

  SampleGraph nodes;
  for(size_t i = 0; i < sample_parent_graph.size(); i++){
    const auto node = sample_parent_graph.at(i);
    const std::string downstream_node_name = (node.downstream_node == NO_DOWNSTREAM_NEIGHBOUR) ? root_node_name : sample_parent_graph.at(node.downstream_node).data.name;
    nodes[node.data.name] = NativeSampleNode{
      .name = node.data.name,
      .x = node.data.x,
      .y = node.data.y,
      .downstream_node = downstream_node_name,
      .upstream_nodes = node.upstream_nodes,
      .area = node.area,
      .total_upstream_area = node.total_upstream_area,
      .label = static_cast<int64_t>(i)
    };
  }

  return {nodes, adjacency_graph_external};
}

}