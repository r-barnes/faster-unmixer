import cvxpy as cp
from typing import Dict, List, Tuple

SampleAdjacency = Dict[Tuple[str,str], int]

root_node_name: str

class NativeSampleNode:
  name: str
  x: int
  y: int
  downstream_node: str
  upstream_nodes: List[str]
  area: int
  total_upstream_area: int
  label: int

def fastunmix(flowdirs_filename: str, sample_data_filename: str) -> Tuple[Dict[str, NativeSampleNode], SampleAdjacency]: ...
