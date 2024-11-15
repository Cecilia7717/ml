"""
Decision stump data structure (i.e. tree with depth=1), non-recursive.
Authors: Sara Mathieson + Adam Poliak + <your names>
Date:
"""

import util

class DecisionStump:

    def __init__(self, partition):
        """
        Create a DecisionStump from the given partition of data by chosing one
        best feature and splitting the data into leaves based on this feature.
        """
        # key: edge, value: probability of positive (i.e. 1)
        self.children = {}

        # use weighted conditional entropy to select the best feature
        feature = partition.best_feature()
        self.name = feature

        # divide data into separate partitions based on this feature
        values = partition.F[feature]
        groups = {}
        for v in values:
            groups[v] = []
        for i in range(partition.n):
            v = partition.data[i].features[feature]
            groups[v].append(partition.data[i])

        # add a child for each possible value of the feature
        for v in values:
            new_partition = Partition(groups[v], partition.F)
            # weighted probability of a positive result
            prob_pos = new_partition.prob_pos()
            self.add_child(v,prob_pos)

    def get_name(self):
        """Getter for the name of the best feature (root)."""
        return self.name

    def add_child(self, edge, prob):
        """
        Add a child with edge as the feature value, and prob as the probability
        of a positive result.
        """
        self.children[edge] = prob

    def get_child(self, edge):
        """Return the probability of a positive result, given feature value."""
        return self.children[edge]

    def __str__(self):
        """Returns a string representation of the decision stump."""
        s = self.name + " =\n"
        for v in self.children:
            s += "  " + v + ", " + str(self.children[v]) + "\n"
        return s

    def classify(self, test_features, thresh=0.5) -> int:
        """
        Classify the test example (using features only) as +1 (positive) or -1
        (negative), using the provided threshold.
        """
        feature_value = test_features[self.name]
    
        if feature_value in self.children:
            prob_pos = self.children[feature_value]
        else:
            # If the feature value isn't in children, return -1
            return -1
        
        # Return 1 if probability >= threshold, otherwise -1
        return 1 if prob_pos >= thresh else -1



def main():
    train_partition = util.read_arff("data/tennis_train.arff")
    test_partition = util.read_arff("data/tennis_test.arff")
    pass

if __name__ == "__main__":
   main()
