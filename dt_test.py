"""
Unit tests for Decision Tree implementation.
Author: Adam Poliak"
Date: 07/02/2024
"""

import unittest
from math import log

import numpy as np
from Partition import Partition, Example
from DecisionTree import DecisionTree
import util

class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        self.data = util.read_arff("data/movies_train.arff", 0)
        return

    def test_prob(self):
        self.assertAlmostEqual(2/3, self.data._prob(1))
        self.assertAlmostEqual(1/3, self.data._prob(-1))
        self.assertAlmostEqual(0, self.data._prob(0))

    def test_best_feature(self):
        self.assertEqual(self.data.best_feature(), 'Director')

    def test_info_gain(self):
        self.assertAlmostEqual(0.3060986113514965, self.data._info_gain('Length'))
        self.assertAlmostEqual(0.5577277787393194, self.data._info_gain('Director'))
        
    def test_cond_entropy(self):
        self.assertAlmostEqual(0.9182958340544896, self.data._cond_entropy('Type', 'Comedy'))
        self.assertAlmostEqual(0.9182958340544896, self.data._cond_entropy('Type', 'Animated'))
        self.assertAlmostEqual(0.0, self.data._cond_entropy('Type', 'Drama'))
        self.assertAlmostEqual(0.7219280948873623, self.data._cond_entropy('Famous_actors', 'No'))

    def test_full_cond_entropy(self):
        self.assertAlmostEqual(0.8455156082707569, self.data._full_cond_entropy('Famous_actors'))
        self.assertAlmostEqual(0.612197222702993,  self.data._full_cond_entropy('Type'))

    def test_entropy(self):
        self.assertAlmostEqual(0.9182958340544896,self.data._entropy())

    def _test_dataset(self, category, overfit=False, max_depth=-1):
        train = None
        if overfit:
          train = util.read_arff(f"data/{category}_test.arff", True)
        else:
          train = util.read_arff(f"data/{category}_train.arff", True)
        test = util.read_arff(f"data/{category}_test.arff", False)
        dtree = None
        if max_depth == -1:
          dtree = DecisionTree(train)
        else:
          dtree = DecisionTree(train, max_depth)

        correct = 0.
        for ex in test.data:
          pred = dtree.classify(ex.features)
          if pred == ex.label:
            correct += 1
        acc = correct / len(test.data) 
        return acc 

if __name__ == '__main__':
    unittest.main()
