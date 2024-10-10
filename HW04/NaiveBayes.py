"""
Author: Cecilia Chen
Date: 10/7/2024
"""

from typing import List, Dict
from util import *

import numpy as np
from collections import defaultdict, Counter

class NaiveBayes:
    def __init__(self, partition: Partition):
        self.class_probs = {}
        self.fe_prob = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.class_counts = Counter()
        self.feature_counts = defaultdict(lambda: defaultdict(Counter))
        self.cont_fe_stats = defaultdict(lambda: defaultdict(lambda: (0, 0)))  # (mean, variance)
        self.tol = 0
        self.feature_names = []
        self.num_classes = 0

        self.train(partition)  # Call train method to initialize model with the given partition

    def train(self, partition: Partition):
        self.tol = partition.n
        self.num_classes = partition.K

        for example in partition.data:
            label = example.label
            self.class_counts[label] += 1
            for feature, value in example.features.items():
                if isinstance(value, str):  # Assuming discrete features are strings
                    self.feature_counts[feature][label][value] += 1
                else:  # Continuous feature
                    self.cont_fe_stats[feature][label][0] += value  # Sum for mean
                    self.cont_fe_stats[feature][label][1] += value ** 2  # Sum for variance

        self.feature_names = list(partition.F.keys())

        # Calculate probabilities for discrete features
        for label, count in self.class_counts.items():
            self.class_probs[label] = (count + 1) / (self.tol + self.num_classes)

        for feature in self.feature_names:
            if feature in self.cont_fe_stats:  # Check if continuous
                for label in self.class_counts:
                    total_count = self.class_counts[label]
                    mean = self.cont_fe_stats[feature][label][0] / total_count
                    variance = (self.cont_fe_stats[feature][label][1] / total_count) - (mean ** 2)
                    self.cont_fe_stats[feature][label] = (mean, variance)
            else:  # Discrete feature
                for label in self.class_counts:
                    total_count = sum(self.feature_counts[feature][label].values())
                    unique_values = len(self.feature_counts[feature][label])
                    for value, count in self.feature_counts[feature][label].items():
                        self.fe_prob[feature][label][value] = (count + 1) / (total_count + unique_values)

    def gaussian_probability(self, x, mean, variance):
        if variance == 0:
            return 0
        return (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-((x - mean) ** 2) / (2 * variance))

    def classify(self, features):
        log_probs = defaultdict(float)
        for label in self.class_probs:
            log_probs[label] = np.log(self.class_probs[label])
            for feature, value in features.items():
                if feature in self.cont_fe_stats:  # Continuous feature
                    mean, variance = self.cont_fe_stats[feature][label]
                    log_probs[label] += np.log(self.gaussian_probability(value, mean, variance))
                else:  # Discrete feature
                    if value in self.fe_prob[feature][label]:
                        log_probs[label] += np.log(self.fe_prob[feature][label][value])
                    else:
                        log_probs[label] += np.log(1 / (sum(self.feature_counts[feature][label].values()) + len(self.feature_counts[feature][label])))

        predicted_label = max(log_probs, key=log_probs.get)
        return predicted_label

    def evaluate(self, test_partition):
        correct = 0
        total = len(test_partition.data)
        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)

        for example in test_partition.data:
            y_hat = self.classify(example.features)
            if y_hat == example.label:
                correct += 1
            confusion_matrix[example.label][y_hat] += 1

        accuracy = correct / total
        return accuracy, confusion_matrix
