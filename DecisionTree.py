"""
Decision tree data structure (recursive).
Author: Adam Poliak + <your name here>
Date:
"""

class DecisionTree:

  def __init__(self, partition, depth=float('inf')) -> None:
    raise NotImplementedError

  def print_self(self, tabs=0) -> None:
    raise NotImplementedError

  def classify(self, test_features) -> int:
    raise NotImplementedError
