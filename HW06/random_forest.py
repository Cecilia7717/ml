"""
Implements Random Forests with decision stumps.
Authors:
Date:
"""

from typing import List, Tuple
from DecisionStump import DecisionStump
from Partition import Partition
import util

def rf_train(train_partition: Partition, T: int) -> List[DecisionStump]:
    """Random Forest algorithm training (T is the number of stumps)."""
    pass

def rf_test(test_partition: Partition, thresh: float, stump_lst: List[DecisionStump]) -> Tuple[float, float]:
    """Random Forest testing."""
    pass

def main():

    # read in data (y in {-1,1})
    args = util.parse_args()
    train = util.read_arff(args.train_filename)
    # print(f"Train examples, n={train.n}")

    test = util.read_arff(args.test_filename)
    # print(f"Test examples, n={test.n}")
    
    #train_partition = util.read_arff(args.train_filename)
    #test_partition  = util.read_arff(args.test_filename)

if __name__ == "__main__":
    main()
