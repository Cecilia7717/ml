"""
Utils for ensemble methods.
Authors: Sara Mathieson + Adam Poliak +  <your names>
Date:
"""

from collections import OrderedDict
from typing import List

# my files
from Partition import *

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Random Forest and AdaBoost classifier arguments")
    
    # Define the command line arguments
    parser.add_argument("-r", "--train_filename", type=str, required=True,
                        help="Path to the ARFF file of training data")
    parser.add_argument("-e", "--test_filename", type=str, required=True,
                        help="Path to the ARFF file of testing data")
    parser.add_argument("-T", "--num_classifiers", type=int, default=10,
                        help="Number of classifiers to use in the ensemble (default=10)")
    parser.add_argument("-p", "--thresh", type=float, default=0.5,
                        help="Probability threshold to classify a test example as positive (default=0.5)")
    
    # Parse and return the arguments
    return parser.parse_args()

def boostrap(data: List[Example]) -> List[Example]:
    new_example = []
    """Create a new bootstrap dataset by sampling with replacement."""
    pass

def read_arff(filename):
    """Read arff file into Partition format."""
    arff_file = open(filename,'r')
    data = [] # list of Examples
    F = OrderedDict() # dictionary

    header = arff_file.readline()
    line = arff_file.readline().strip()

    # read the attributes
    while line != "@data":
        line = line.replace('{','').replace('}','').replace(',','')
        tokens = line.split()
        name = tokens[1][1:-1]
        features = tokens[2:]

        # label
        if name != "class":
            F[name] = features
        else:
            first = tokens[2]
        line = arff_file.readline().strip()

    # read the examples
    for line in arff_file:
        tokens = line.strip().split(",")
        X_dict = {}
        i = 0
        for key in F:
            val = tokens[i]
            X_dict[key] = val
            i += 1
        label = -1 if tokens[-1] == first else 1
        # set weight to None for now
        data.append(Example(X_dict,label,None))

    arff_file.close()

    # set weights on each example
    n = len(data)
    for i in range(n):
        data[i].set_weight(1/n)

    partition = Partition(data, F)
    return partition
