"""
TODO: high level comment, author, and date
"""

import util
import argparse
from DecisionTree import *

def get_args():
    """Parse command line arguments (train and test arff files)."""
    argparser = argparse.ArgumentParser(description='run decision tree method')

    argparser.add_argument("-r", "--train_filename", help="path to train arff file",
                           type=str, default='data/',
                           required=True)
    argparser.add_argument("-e", "--test_filename", help="path to test arff file",
                           type=str, default='data/',
                           required=True)
    argparser.add_argument("-d", "--depth", help="max depth (optional)",
                           type=int, default=-1)

    args = argparser.parse_args()
    return args


def main():

    args = get_args()
    train_partition = util.read_arff(args.train_filename, True)
    test_partition  = util.read_arff(args.test_filename, False)

    # TODO:create an instance of the DecisionTree class from the train_partition
    if args.depth != -1:
        decision_tree = DecisionTree(train_partition, depth=args.depth)
    else:
        decision_tree = DecisionTree(train_partition)
    # depth from args

    # TODO: print text representation of the DecisionTree
    decision_tree.print_self()

    # TODO: evaluate the decision tree on the test_partition
    correct_predictions = 0
    total_examples = len(test_partition.data)
    for example in test_partition.data:
        # Classify each test example and compare it to the true label
        prediction = decision_tree.classify(example.features)
        if prediction == example.label:
            correct_predictions += 1

    # Calculate accuracy
    accuracy = correct_predictions / total_examples
    print(f"{correct_predictions} out of {total_examples} correct")
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    main()
