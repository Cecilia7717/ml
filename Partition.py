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
        self.entropy = self._entropy
    
    def is_continuous(self, feature: str) -> bool:
        """Determine if a feature is continuous."""
        return all(isinstance(example.features[feature], (int, float)) for example in self.data)
    
    def get(self, index):
        print(f"herehere{self.F[index].features}")
        """Get the features of a specific example in the partition."""
        return self.F[index].features 
    
    def _prob(self, c: int) -> float:
        """Compute P(Y=c)."""
        if self.n == 0:
            return 0
        label_num = Counter(example.label for example in self.data)
        prob = label_num[c] / self.n
        return prob

    def _entropy(self) -> float:
        """compute entropy"""
        sum = 0
        labels = [-1, 1]
        for label in labels:
            prob = self._prob(label)
            if prob > 0:
                single = prob * math.log2(prob)
                sum = sum - single
        return sum
        
    def _cond_entropy(self, feature_name:str, feature_value) -> float:
        """compute conditional entropy"""
        """H(Y|X = v) here is the entropy of the subset of the data where feature X's value is v"""
        subset = [example for example in self.data if example.features[feature_name] == feature_value]
        if not subset:
            return 0.0
        
        label_counts = Counter(example.label for example in subset)
        entropy = 0.0
        for count in label_counts.values():
            # P(y_i|X=v)
            prob = count / len(subset)
            if prob > 0:
                entropy -= prob * math.log2(prob)
        return entropy

    def _full_cond_entropy(self, feature_name) -> float:
        """H(Y|X)"""
        cond_entropy = 0.0
        for feature_value in self.F[feature_name]:
            subset = [example for example in self.data if example.features[feature_name] == feature_value]
            prob = len(subset) / self.n
            if len(subset) > 0:
                cond_entropy += prob * self._cond_entropy(feature_name, feature_value)
        return cond_entropy

    def best_threshould(self, feature_name):
        """find the features with the maximum information gain"""
        feature_values = [example.features[feature_name] for example in self.data]
        sorted_values = sorted(set(feature_values))
        
        best_threshold = None
        max_info_gain = float('-inf')

        # Try each midpoint between sorted feature values as a potential threshold
        for i in range(1, len(sorted_values)):
            threshold = (sorted_values[i - 1] + sorted_values[i]) / 2
            info_gain = self._info_gain_thre(threshold, feature_name)

            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_threshold = threshold

        return best_threshold
    
    def _info_gain_thre(self, threshold, feature):
        examples_below = [ex for ex in self.partition.data if ex.features[feature] <= threshold]
        examples_above = [ex for ex in self.partition.data if ex.features[feature] > threshold]

        infor_below = self._info_gain(examples_below)
        infor_above = self._info_gain(examples_above)

        return max(infor_below, infor_above)
    
    def best_feature(self):
        """find the features with the maximum information gain"""
        best_ig = -float('inf')
        best_feature = None
        for feature in self.F:
            ig = self._info_gain(feature)
            if ig > best_ig:
                best_feature = feature
                best_ig = ig
        return best_feature
    
    def _info_gain(self, feature: str) -> float:
        """compute information gain here"""
        # H(Y)
        entropy_total = self._entropy()
        # H(Y|X)
        entropy_cond = self._full_cond_entropy(feature)
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