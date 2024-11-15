"""
Implements AdaBoost algorithm with decision stumps.
Authors:
Date:
"""

import util

def main():

    # read in data (y in {-1,1})
    opts = util.parse_args()
    train_partition = util.read_arff(opts.train_filename)
    test_partition  = util.read_arff(opts.test_filename)

if __name__ == "__main__":
    main()
