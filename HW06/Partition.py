"""
Example and Partition data structures.
Authors: Sara Mathieson + Adam Poliak + <your names>
Date:
"""
from collections import Counter
import math
from collections import defaultdict
import numpy as np
class Example:

    def __init__(self, features, label, weight):
        """
        Class to hold an individual example (data point) and its weight.
        features -- dictionary of features for this example (key: feature name,
            value: feature value for this example)
        label -- label of the example (-1 or 1)
        weight -- weight of the example (starts out as 1/n)
        """
        self.features = features
        self.label = label
        self.weight = weight

    def set_weight(self, new):
        """Change the weight on an example (used for AdaBoost)."""
        self.weight = new

class Partition:

    def __init__(self, data, F):
        """
        Class to hold a set of examples and their associated features, as well
        as compute information about this partition (i.e. entropy).
        data -- list of Examples
        F -- dictionary (key: feature name, value: list of feature values)
        """
        self.data = data
        self.F = F
        self.n = len(self.data)

    """TODO complete this class.""" 

    def prob_pos(self) -> float:
        """Based on the weights, compute the probability of a positive label."""
        # print(len(example.weight) for example in self.data)
        total_weight = sum(example.weight for example in self.data)
        pos_weight = sum(example.weight for example in self.data if example.label == 1)
        return pos_weight / total_weight if total_weight > 0 else 0.0

    def _cond_prob(self, f: str, val: str, c: int) -> float:
        """Compute P(Y=c|X=val)."""
        total_count = sum(1 for example in self.data if example.features == val)
        class_count = sum(1 for example in self.data if example.label == c )
        return class_count / total_count if total_count > 0 else 0


    def _prob(self, c: int) -> float:
        """Compute P(Y=c)."""
        

    def _entropy(self) -> float:
        """Compute H(Y)."""
        pass

    def _cond_entropy(self, f: str, val: str) -> float:
        """Compute H(Y|X=val)."""
        pass

    def _probX(self, f: str, val: str) -> float:
        """Compute P(X=val)."""
        pass

    def _full_cond_entropy(self, f: str) -> float:
        """Compute H(Y|X)."""
        pass
  
    def _info_gain(self, f: str, val: str) -> float:
        """Compute information gain."""
        pass

    def best_feature(self, f: str, val: str) -> float:
        """Compute information gain of each feature to find the best one:."""
        pass

