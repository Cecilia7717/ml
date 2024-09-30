"""
Author: Cecilia Chen
Date: 9/23/2024
"""
from typing import List, Dict
from collections import Counter
from Partition import *

class DecisionTree:

    def __init__(self, partition: Partition, depth=float('inf')) -> None:
        self.feature = None
        self.positive_count = 0
        self.negative_count = 0
        self.split_value = 0

        if not (self._stop(partition, depth)):
            self.label = None
            self._isleaf = False
        else:
            self._isleaf = True
            label_count = Counter(example.label for example in partition.data)
            self.label = -1 if label_count[-1] >= label_count[1] else 1

        self.children_node = {}
        self.depth = depth
        self.partition = partition

        self.feature = self.best_feature(partition)
        self.split_partition()

    def best_feature(self, partition):
        """Find the feature with the maximum information gain."""
        return partition.best_feature()
    
    def best_threshold(self, feature_name):
      """Find the threshold that maximizes information gain for a continuous feature."""
      
      # Ensure feature_values are scalar for continuous features
      feature_values = [example.features[feature_name] for example in self.partition.data]
      
      # Flatten feature values if they are lists or multi-dimensional (assuming you want to handle this)
      if isinstance(feature_values[0], list):
          feature_values = [item for sublist in feature_values for item in sublist]  # Flattening the lists

      sorted_values = sorted(set(feature_values))  # Sorting unique scalar values

      best_threshold = None
      max_info_gain = float('-inf')

      for i in range(1, len(sorted_values)):
          threshold = (sorted_values[i - 1] + sorted_values[i]) / 2
          info_gain = self.partition._info_gain_thre(threshold, feature_name)

          if info_gain > max_info_gain:
              max_info_gain = info_gain
              best_threshold = threshold

      return best_threshold


    def split_partition(self):
        """Split the partition based on the best feature."""
        if self._stop(self.partition, self.depth):
            return

        remaining_features = {f: values for f, values in self.partition.F.items() if f != self.feature}

        if self.partition.is_continuous(self.feature):
            self.split_value = self.best_threshold(self.feature)

            
            examples_below_threshold = [ex for ex in self.partition.data if ex.features[self.feature] <= self.split_value]
            examples_above_threshold = [ex for ex in self.partition.data if ex.features[self.feature] > self.split_value]

            below_partition = Partition(examples_below_threshold, remaining_features)
            above_partition = Partition(examples_above_threshold, remaining_features)

            self.children_node[f'<= {self.split_value}'] = DecisionTree(below_partition, depth=self.depth - 1)
            self.children_node[f'> {self.split_value}'] = DecisionTree(above_partition, depth=self.depth - 1)
        else:
            for value in self.partition.F[self.feature]:
                examples_feature_f = [ex for ex in self.partition.data if ex.features[self.feature] == value]
                new_partition = Partition(examples_feature_f, remaining_features)
                if examples_feature_f:
                    self.children_node[value] = DecisionTree(new_partition, depth=self.depth - 1)
                else:
                    self.children_node[value] = DecisionTree(Partition([], remaining_features), depth=0)

    def _stop(self, partition, current_depth) -> bool:
        """Check the stopping criteria."""
        if len(set(ex.label for ex in partition.data)) == 1:
            return True
        if len(partition.F) == 0: 
            return True
        if current_depth == 0:
            return True
        if not partition.data:
            return True
        return False

    def classify(self, example: Dict):
        #Classify an example by traversing the tree.
        if self._isleaf:
            return self.label
        feature_value = example[self.feature]

        if self.partition.is_continuous(self.feature):
            if feature_value <= self.split_value:
                return self.children_node[f'<= {self.split_value}'].classify(example)
            else:
                return self.children_node[f'> {self.split_value}'].classify(example)
        else:
            if feature_value in self.children_node:
                return self.children_node[feature_value].classify(example)
            return None  # Handle unseen values
    """  
    def classify(self, example: Example):
        #Classify an example by traversing the tree.
        if self._isleaf:
            return self.label
        feature_value = example.features[self.split_partition]

        if self.partition.is_continuous(self.split_partition):
            if feature_value <= self.split_value:
                return self.children_node[f'<= {self.split_value}'].classify(example)
            else:
                return self.children_node[f'> {self.split_value}'].classify(example)
        else:
            if feature_value in self.children_node:
                return self.children_node[feature_value].classify(example)
            return None  # Handle unseen values
    """
    def print_self(self, tabs=0):
        """Print the structure of the tree."""
        indent = "    " * tabs
        label_count = Counter(ex.label for ex in self.partition.data)
        pos_count = label_count[1]
        neg_count = label_count[-1]
        print(f"[{neg_count}, {pos_count}]", end="")
        if self._isleaf:
            print(f": {self.label}")
        else:
            print(f"")
            for value, child in self.children_node.items():
                child_label_count = Counter(ex.label for ex in child.partition.data)
                child_pos_count = child_label_count[1]
                child_neg_count = child_label_count[-1]
                if tabs == 0:
                    print(f"{self.feature}={value} ", end="")
                else:
                    print(f"|{indent}{self.feature}={value} ", end="")
                child.print_self(tabs + 1)
