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
    train_partition = util.read_arff(opts.train_filename, True)
    test_partition  = util.read_arff(opts.test_filename, False)

    # Sanity check
    # print("num train =", train_partition.n, ", num classes =", train_partition.K)
    # print("num test  =", test_partition.n, ", num classes =", test_partition.K)

    # Initialize Naive Bayes model
    nb_model = NaiveBayes(train_partition)

    # Evaluate the model on the test set
    accuracy, confusion_matrix = nb_model.evaluate(test_partition)

    # Output results
    correct = int(accuracy * test_partition.n)  # Calculate correct count based on accuracy
    total = test_partition.n
    print(f"Accuracy: {accuracy:.6f} ({correct} out of {total} correct)")
    print("\n      prediction")
    print("   " + "  ".join(str(i) for i in range(nb_model.num_classes)))
    print(" ---------------------")
    for i in range(nb_model.num_classes):
        print(f"{i}| " + "  ".join(str(confusion_matrix[i][j]) for j in range(nb_model.num_classes)))

if __name__ == '__main__':
    main()


