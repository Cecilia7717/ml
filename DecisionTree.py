"""
Decision tree data structure (recursive).
Author: Adam Poliak + Cecilia Chen
Date:
"""
from typing import List, Dict
from collections import Counter
from Partition import *

class DecisionTree:

  def __init__(self, partition: Partition, depth=float('inf')) -> None:
    self.feature = None
    if depth != 1:
      self.label = None
      self._isleaf = False
    else:
      # then do nothing just find the major label
      self._isleaf = True
      label_count = Counter(example.label for example in partition.data)
      if label_count[-1] >= label_count[1]:
        self.label = -1
      else:
        self.label = 1

    # store the child nodes as a directory for mapping feature values to subtrees  
    self.children_node = {}

    self.depth = depth
    self.partition = partition
    
    # initialize the root first
    self.feature = self.best_feature(partition)
    
    # recursively begin to split the partition
    self.split_partition()

    raise NotImplementedError

  def split_partition(self):
    
  def best_feature(self, partition):
    """find the features with the maximum information gain"""
    best_ig = -float('inf')
    best_feature = None
    for feature in partition.F:
      ig = partition.infor_gain(feature)
      if ig > best_ig:
        best_feature = feature
        best_ig = ig
    return best_feature
  
  def print_self(self, tabs=0) -> None:
    raise NotImplementedError

  def _stop(self, partition, current_depth) -> bool:
    """Check the stopping criteria."""
    # 1. All examples have the same label
    if len(set(ex.label for ex in partition.data)) == 1:
      return True
    # 2. No features remain to split on
    if len(partition.F) == 0: 
      return True
    # 3. Maximum depth reached
    if current_depth >= self.depth:
      return True
    # 4. No examples in the partition
    if not partition.data:
      return True
    return False


  def classify(self, test_features) -> int:

  def subtree(self, partition, features, depth) -> None:
    if(self._stop(partition, depth, depth)):
