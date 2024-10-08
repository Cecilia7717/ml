"""
Author: Cecilia Chen
Date: 10/7/2024
"""

import util
import argparse
from NaiveBayes import *

def get_args():
    """Parse command line arguments (train and test arff files)."""
    argparser = argparse.ArgumentParser(description='run naive bayes method')

    argparser.add_argument("-r", "--train_filename", help="path to train arff file",
                           type=str, default='data/',
                           required=True)
    argparser.add_argument("-e", "--test_filename", help="path to test arff file",
                           type=str, default='data/',
                           required=True)
    args = argparser.parse_args()
    return args



def main():
    opts = get_args()
    #opts = util.parse_args()
    train_partition = util.read_arff(opts.train_filename, True)
    test_partition  = util.read_arff(opts.test_filename, False)

    # sanity check
    print("num train =", train_partition.n, ", num classes =", train_partition.K)
    print("num test  =", test_partition.n, ", num classes =", test_partition.K)

    nb_model = NaiveBayes(train_partition)
    

if __name__ == '__main__':
    main()
