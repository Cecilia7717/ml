
import sys
import numpy as np
from LogisticRegression import *
import optparse

import matplotlib.pyplot as plt

def parse_args():
    """Parse command line arguments (train and test arff files)."""
    parser = optparse.OptionParser(description='run decision tree method')

    parser.add_option('-r', '--train_filename', type='string', help='path to' +\
        ' train arff file')
    parser.add_option('-e', '--test_filename', type='string', help='path to' +\
        ' test arff file')
    parser.add_option('-a', '--alpha', type='float', default=0.1, help='alpha (optional)')
    parser.add_option('-l', '--lmda', type='float', default=0.1, help='lmda (optional)')

    (opts, args) = parser.parse_args()

    mandatories = ['train_filename', 'test_filename',]
    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()

    return opts
    

def parse_csv(filename):
    """Parse CSV file to separate features and labels."""
    with open(filename, 'r') as f:
        X = []
        y = []
        for line in f:
            tokens = line.strip().split(",")
            X.append([float(x) for x in tokens[0:-1]])
            y_val = int(tokens[-1])  # Assuming labels are already 0 or 1
            y.append(y_val)
        X = np.array(X)
        y = np.array(y)
    return X, y

def confusion_matrix(y_true, y_pred):
    """Calculate confusion matrix values."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])

def main():
    # Parse command-line arguments
    opts = parse_args()
    
    # Parse training and testing data
    X_train, y_train = parse_csv(opts.train_filename)
    X_test, y_test = parse_csv(opts.test_filename)
    
    # Train logistic regression model
    coef = fit_SGD(X_train, y_train, alpha=opts.alpha)
    
    y_pred_prob = prediction(X_test, coef)
    y_pred = threshold(y_pred_prob)
    
    acc = accuracy(X_test, coef, y_test)
    confu = confusion_matrix(y_test, y_pred)
    
    # Calculate correct predictions count
    correct_count = np.sum(y_pred == y_test)
    total_count = len(y_test)
    
    # Display results
    print(f'Accuracy: {acc:.6f} ({correct_count} out of {total_count} correct)\n')
    print("   prediction")
    print("      0  1")
    print("    ------")
    print(f" 0 | {confu[0, 0]:2} {confu[0, 1]:2}")
    print(f" 1 | {confu[1, 0]:2} {confu[1, 1]:2}")

if __name__ == '__main__':
    main()
