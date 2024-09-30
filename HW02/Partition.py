"""
Partition class (holds feature information, feature values, and labels for a
dataset). Includes helper class Example.
Author: Sara Mathieson + Adam Poliak + Cecilia Chen
Date: 9/23/2024
"""

from typing import List, Dict
from collections import Counter
import math
from collections import defaultdict

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
    
    def is_continuous(self, feature) -> bool:
        """Determine if a feature is continuous."""
        #for example in self.data:
            # Print the feature value and its type for debugging
            #print(f"Feature value: {example.features[feature]}, Type: {type(example.features[feature])}")
        #print(all(isinstance(example.features[feature], (int, float)) for example in self.data))
        return all(isinstance(example.features[feature], (int, float)) for example in self.data)


    def get(self, index):
        #print(f"herehere{self.F[index].features}")
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
        
    def _cond_entropy(self, feature_name, feature_value) -> float:
        """Compute conditional entropy H(Y|X = v) for a given feature value."""
        
        # Group examples by their feature values using defaultdict
        grouped_by_feature = defaultdict(list)
        for example in self.data:
            #print(f"{feature_name}")
            feature_name_1 = feature_name
            if type(feature_name) == list:
                break
            grouped_by_feature[example.features[feature_name_1]].append(example)
        #feature_value = value[feature_name] if type(value) == list else feature_name 
        # Get the subset of examples for the specific feature value
        subset = grouped_by_feature.get(feature_value, [])
        
        if not subset:
            return 0.0
        
        # Compute the label counts in the subset
        label_counts = Counter(example.label for example in subset)
        
        # Compute the entropy
        entropy = 0.0
        for count in label_counts.values():
            prob = count / len(subset)  # P(y_i | X = v)
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        return entropy

    '''def _full_cond_entropy(self, feature_name) -> float:
        """H(Y|X)"""
        cond_entropy = 0.0
        # Ensure that the feature values are hashable (e.g., using set)
        feature_values = set(self.F)  # Convert to set if not already
        #print(feature_values.s)
        for feature_value in feature_values:
            subset = [example for example in self.data if example.features[feature_name] == feature_value]
            if self.n != 0:
                prob = len(subset) / self.n
                if len(subset) > 0:
                    cond_entropy += prob * self._cond_entropy(feature_name, feature_value)
        return cond_entropy

    def _full_cond_entropy(self, feature_name) -> float:
        """H(Y|X)"""
        cond_entropy = 0.0
        feature_values = set()

        for example in self.data:
            value = example.features
            #print(f"Processing example with features: {value}")  # Debugging print
            #print(f"Feature name: {feature_name}, Feature value type: {type(value[feature_name])}")

            # Access the feature by name, and handle unhashable types like lists
            feature_value = value[feature_name] if type(value) == list else feature_name 
            if isinstance(feature_value, list):
                feature_value = tuple(feature_value)  # Convert lists to tuples (hashable)

            print(f"Processed feature value: {feature_value}, Type: {type(feature_value)}")  # Debugging print

            feature_values.add(feature_value)

        # Continue with calculating conditional entropy
        for feature_value in feature_values:
            print(f"1{type(example)}")
            print(f"2{type(example.features)}")
            print(f"3{type(feature_value)}")
            print(f"4{type(self.data)}")

            subset = [example for example in self.data if (example.features[feature_name] == feature_value)]
            #subset = [example for example in self.data if (example.features[feature_name] == feature_value if type(example.features) == list )]
            if self.n != 0:
                prob = len(subset) / self.n
                if len(subset) > 0:
                    cond_entropy += prob * self._cond_entropy(feature_name, feature_value)

        return cond_entropy'''

    def _full_cond_entropy(self, feature_name) -> float:
        """H(Y|X)"""
        cond_entropy = 0.0
        feature_groups = defaultdict(list)

        # Pre-group the examples by feature values
        for example in self.data:
            value = example.features

            # Access the feature by name, and handle unhashable types like lists
            feature_value = value[feature_name] if type(value) == list else feature_name 
            if isinstance(feature_value, list):
                feature_value = tuple(feature_value)  # Convert lists to tuples (hashable)

            # Group examples by their feature value
            feature_groups[feature_value].append(example)

        # Calculate conditional entropy based on pre-grouped examples
        for feature_value, subset in feature_groups.items():
            if self.n != 0:
                prob = len(subset) / self.n
                if len(subset) > 0:
                    cond_entropy += prob * self._cond_entropy(feature_name, feature_value)

        return cond_entropy
    
    def _info_gain_thre(self, threshold, feature):
        examples_below = [ex for ex in self.data if ex.features[feature] <= threshold]
        examples_above = [ex for ex in self.data if ex.features[feature] > threshold]

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