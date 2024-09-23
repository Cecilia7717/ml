"""
TODO: high level comment, author, and date
"""

import util
import argparse

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

    # TODO: print text representation of the DecisionTree

    # TODO: evaluate the decision tree on the test_partition

if __name__ == '__main__':
    main()
