"""
Partition class (holds feature information, feature values, and labels for a
dataset). Includes helper class Example.
Author: Sara Mathieson + Adam Poliak + <your name here>
Date:
"""

from typing import List, Dict

class Example:

    def __init__(self, features: Dict, label: int) -> None:
        """Helper class (like a struct) that stores info about each example."""
        # dictionary. key=feature name: value=feature value for this example
        self.features = features
        self.label = label # in {-1, 1}

class Partition:

    def __init__(self, data: List[Example], F: Dict) -> None:
        """Store information about a dataset"""
        self.data = data # list of examples
        # dictionary. key=feature name: value=set of possible values
        self.F = F
        self.n = len(self.data)

    # TODO: implement entropy and information gain methods here!
    def _prob(self, c: int) -> float:
        """Compute P(Y=c)."""
        raise NotImplementedError

