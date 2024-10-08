"""
Utils for naive bayes
Author: Sara Mathieson + Adam Poliak
Date: 10/7/2024
"""

# my file imports
from typing import List, Dict
from collections import Counter, OrderedDict, defaultdict
import math


class Example:

    def __init__(self, features: Dict, label: int) -> None:
        """Helper class (like a struct) that stores info about each example."""
        # dictionary. key=feature name: value=feature value for this example
        self.features = features
        self.label = label # in {-1, 1}

class Partition:

    def __init__(self, data: List[Example], F: Dict, K: int) -> None:
        """Store information about a dataset"""
        self.data = data # list of examples
        # dictionary. key=feature name: value=set of possible values
        self.F = F
        self.n = len(self.data)
        self.K = K

    def get(self, index):
        #print(f"herehere{self.F[index].features}")
        """Get the features of a specific example in the partition."""
        return self.F[index].features 
    
    def _prob(self, c: int) -> float:
        """Compute P(Y=c)."""
        if self.n == 0:
            return 0
        label_num = Counter(example.label for example in self.data)
        prob = label_num[c] / self.n
        return prob


def read_arff(filename: str, train: bool) -> Partition:
    """
    Read arff file into Partition format, assuming all features are discrete.
    Converts labels from arbitrary strings to integers {0, 1, 2, ..., K-1}.
    """
    arff_file = open(filename, 'r')
    data = []  # list of Examples
    F = OrderedDict()  # key: feature name, value: list of feature values
    label_map = {}  # map labels to integers {0, 1, 2, ..., K-1}
    label_counter = 0

    header = arff_file.readline()
    line = arff_file.readline().strip()

    # read the attributes
    while line != "@data":
        clean = line.replace('{', '').replace('}', '').replace(',', '')
        tokens = clean.split()
        name = tokens[1][1:-1]

        # all features are discrete now
        feature_values = tokens[2:]

        # record features or label
        if name != "class":
            F[name] = feature_values
        else:
            class_values = feature_values  # capture the class labels
        line = arff_file.readline().strip()

    # read the examples
    for line in arff_file:
        tokens = line.strip().split(",")
        X_dict = {}
        i = 0
        for key in F:
            X_dict[key] = tokens[i]
            i += 1

        # map the label to an integer in {0, 1, 2, ..., K-1}
        label_str = tokens[-1]
        if label_str not in label_map:
            label_map[label_str] = label_counter
            label_counter += 1

        label = label_map[label_str]
        data.append(Example(X_dict, label))

    arff_file.close()
    #print(len(label_map))
    partition = Partition(data, F, K = len(label_map))
    return partition

