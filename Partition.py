"""
Partition class (holds feature information, feature values, and labels for a
dataset). Includes helper class Example.
Author: Sara Mathieson + Adam Poliak + Cecilia Chen
Date: 9/23/2024
"""

from typing import List, Dict
from collections import Counter
import math
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


    def _prob(self, c: int) -> float:
        """Compute P(Y=c)."""
        if self.n == 0:
            return 0
        label_num = Counter(example.label for example in self.data)
        prob = label_num[c] / self.n
        return prob

    def entropy(self) -> float:
        """compute entropy"""
        sum = 0
        labels = [-1, 1]
        for label in labels:
            prob = self._prob(label)
            if prob > 0:
                single = prob * math.log2(prob)
                sum = sum - single
        return sum
        
    def cond_entropy(self, data_wihtout_itself: List[Example]) -> float:
        """compute conditional entropy"""
        """H(Y|X = v) here is the entropy of the subset of the data where feature X's value is v"""
        if not data_wihtout_itself:
            return 0.0
        # here should has either only 1 or only -1, since the full dataset only has {1, -1}
        label_left = Counter(example.label for example in data_wihtout_itself)
        entropy = 0.0
        for label in label_left.values():
            # P(y_i|X=v)
            prob = label/len(data_wihtout_itself)
            if prob > 0:
                entropy = entropy - prob * math.log2(prob)
        return entropy

    def infor_gain(self, feature: str) -> float:
        """compute information gain here"""
        # H(Y)
        entropy_total = self.entropy()
        entropy_cond = 0.0

        # summation of sum_i^c
        for value in self.F[feature]:
            if self.n != 0:
                data_if_x_is_v = [example for example in self.data if example.features[feature] == value]
                entropy_cond = entropy_cond + (len(data_if_x_is_v)/self.n) * self.cond_entropy(data_wihtout_itself=data_if_x_is_v)
        return entropy_total - entropy_cond
    
'''
# below is the checking for tennis, where information gain is right as below:
#Overall Entropy: 0.9402859586706311
#Information Gain for Outlook: 0.24674981977443933
#Information Gain for Temperature: 0.02922256565895487
#Information Gain for Humidity: 0.15183550136234159
#Information Gain for Wind: 0.04812703040826949

# Data provided in the prompt
data_strings = [
    ("Sunny", "Hot", "High", "Weak", 0),
    ("Sunny", "Hot", "High", "Strong", 0),
    ("Overcast", "Hot", "High", "Weak", 1),
    ("Rain", "Mild", "High", "Weak", 1),
    ("Rain", "Cool", "Normal", "Weak", 1),
    ("Rain", "Cool", "Normal", "Strong", 0),
    ("Overcast", "Cool", "Normal", "Strong", 1),
    ("Sunny", "Mild", "High", "Weak", 0),
    ("Sunny", "Cool", "Normal", "Weak", 1),
    ("Rain", "Mild", "Normal", "Weak", 1),
    ("Sunny", "Mild", "Normal", "Strong", 1),
    ("Overcast", "Mild", "High", "Strong", 1),
    ("Overcast", "Hot", "Normal", "Weak", 1),
    ("Rain", "Mild", "High", "Strong", 0)
]

# Convert data strings into Example instances
examples = [Example(features={'Outlook': outlook, 'Temperature': temp, 'Humidity': humidity, 'Wind': wind}, label=(1 if label == 0 else -1)) for outlook, temp, humidity, wind, label in data_strings]

# Define the feature values
features_dict = {
    'Outlook': {'Sunny', 'Overcast', 'Rain'},
    'Temperature': {'Hot', 'Mild', 'Cool'},
    'Humidity': {'High', 'Normal'},
    'Wind': {'Strong', 'Weak'}
}

# Create Partition
partition = Partition(data=examples, F=features_dict)

# Calculate Entropy and Information Gain for each feature
print(f"Overall Entropy: {partition.entropy()}")
for feature in features_dict.keys():
    print(f"Information Gain for {feature}: {partition.infor_gain(feature)}")
'''
'''
# below is the checking for movie, where information gain is right as below:
Overall Entropy: 0.9182958340544896
Information Gain for Type: 0.3060986113514965
Information Gain for Length: 0.3060986113514965
Information Gain for Director: 0.5577277787393194
Information Gain for Famous_actors: 0.07278022578373267
# Data provided in the prompt
data_strings = [
    ("Comedy", "Short", "Adamson", "No", "Yes"),
    ("Animated", "Short", "Lasseter", "No", "No"),
    ("Drama", "Medium", "Adamson", "No", "Yes"),
    ("Animated", "Long", "Lasseter", "Yes", "No"),
    ("Comedy", "Long", "Lasseter", "Yes", "No"),
    ("Drama", "Medium", "Singer", "Yes", "Yes"),
    ("Animated", "Short", "Singer", "No", "Yes"),
    ("Comedy", "Long", "Adamson", "Yes", "Yes"),
    ("Drama", "Medium", "Lasseter", "No", "Yes")
]

# Convert data strings into Example instances with adjusted labels
examples = [
    Example(
        features={
            'Type': type_,
            'Length': length,
            'Director': director,
            'Famous_actors': famous_actors
        },
        label=(-1 if label == "No" else 1)  # Transform label: "No" -> -1, "Yes" -> 1
    )
    for type_, length, director, famous_actors, label in data_strings
]

# Define the feature values
features_dict = {
    'Type': {'Comedy', 'Animated', 'Drama'},
    'Length': {'Short', 'Medium', 'Long'},
    'Director': {'Adamson', 'Lasseter', 'Singer'},
    'Famous_actors': {'Yes', 'No'}
}

# Create Partition
partition = Partition(data=examples, F=features_dict)

# Testing: Calculate Entropy and Information Gain for each feature
print(f"Overall Entropy: {partition.entropy()}")
for feature in features_dict.keys():
    print(f"Information Gain for {feature}: {partition.infor_gain(feature)}")'''