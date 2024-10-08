"""
Author: Cecilia Chen
Date: 10/7/2024
"""

from typing import List, Dict
from collections import Counter
from util import *

class NaiveBayes:

    def __init__(self, partition: Partition) -> None:
        self.feature = None
        self.positive_count = 0
        self.negative_count = 0
        self.split_value = 0


        self.feature = self.best_feature(partition)
        self.split_partition()

    def best_feature(self, partition):
        """Find the feature with the maximum information gain."""
        return partition.best_feature()