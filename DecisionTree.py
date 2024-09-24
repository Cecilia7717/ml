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
    self.positive_count = 0
    self.negative_count = 0

    # depth here is depth left
    if not (self._stop(partition, depth)):
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

  def find_best_feature(self):
        # Placeholder function to find the best feature to split on
        return next(iter(self.partition.F))
  
  def split_partition(self):
    """split the partition here"""
    remaining_features = {f: values for f, values in self.partition.F.items() if f != self.feature}
    if self._stop(self.partition, self.depth):
      return  # Stop further splitting
    
    
    # Determine remaining features for child nodes
    remaining_features = {
        f: values for f, values in self.partition.F.items() if f != self.feature
    }

    # Iterate over each possible value of the best feature
    for value in self.partition.F[self.feature]:
        # Create a subset of examples where this feature has the specific value
        examples_feature_f = [
            example for example in self.partition.data if example.features[self.feature] == value
        ]

        # Create a new partition for these examples
        new_partition = Partition(examples_feature_f, remaining_features)

        # Create a child decision tree node, decreasing depth
        if examples_feature_f:
            self.children_node[value] = DecisionTree(new_partition, depth=self.depth - 1)
        else:
            # If no examples, create a leaf node with the majority label
            self.children_node[value] = DecisionTree(
                Partition([], remaining_features), depth=0
            )

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

  def _stop(self, partition, current_depth) -> bool:
    """Check the stopping criteria."""
    # 1. All examples have the same label
    if len(set(ex.label for ex in partition.data)) == 1:
      return True
    # 2. No features remain to split on
    if len(partition.F) == 0: 
      return True
    # 3. Maximum depth reached
    if current_depth == 0:
      return True
    # 4. No examples in the partition
    if not partition.data:
      return True
    return False

  # test
  def classify(self, test_features) -> int:
    """Classify a test example by traversing the tree."""
    if self._isleaf:
        return self.label
    feature_value = test_features[self.feature]
    if feature_value in self.children_node:
        return self.children_node[feature_value].classify(test_features)
    else:
        return self.label

  def print_self(self, tabs=0):
    indent = "    " * tabs  # Create indentation based on the level of depth
    label_count = Counter(ex.label for ex in self.partition.data)
    pos_count = label_count[1]
    neg_count = label_count[-1]
    print(f"[{neg_count}, {pos_count}]", end ="")
    # If the current node is a leaf, print its label
    if self._isleaf:
      print(f": {self.label}")
    else:
      # Print the current feature being split at this node
      print(f"")
        
      # Iterate over each child node and recursively print their structure
      for value, child in self.children_node.items():
        # Count examples in the child's partition
        child_label_count = Counter(ex.label for ex in child.partition.data)
        child_pos_count = child_label_count[1]
        child_neg_count = child_label_count[-1]
        if tabs == 0:
           print(f"{self.feature}={value} ", end="")
        else:   
          # Print the feature value along with the example counts
          print(f"|{indent}{self.feature}={value} ", end ="")
          # Recursively print each child node with increased indentation
        child.print_self(tabs + 1)
