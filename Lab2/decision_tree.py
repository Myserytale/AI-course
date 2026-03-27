"""
Decision Tree Implementation using Entropy and Information Gain

This implementation demonstrates:
1. How entropy measures uncertainty in a dataset
2. How information gain helps select the best attribute to split on
3. How divide and conquer recursively builds the tree
"""

import pandas as pd
import numpy as np
from collections import Counter
import math


class DecisionTreeNode:
    """Node in the decision tree"""
    def __init__(self, attribute=None, value=None, label=None):
        self.attribute = attribute  # The attribute to split on
        self.value = value  # The value for this branch (from parent)
        self.label = label  # Class label if this is a leaf node
        self.children = {}  # Dictionary of child nodes
        
    def is_leaf(self):
        return self.label is not None


class DecisionTree:
    """Decision Tree Classifier using ID3 algorithm"""
    
    def __init__(self):
        self.root = None
        self.target_attribute = None
        
    def calculate_entropy(self, data, target_attribute):
        """
        Calculate entropy of a dataset.
        
        Entropy measures the impurity/uncertainty in the data:
        H(S) = -Σ p(i) * log2(p(i))
        
        Where:
        - S is the dataset
        - p(i) is the proportion of class i in S
        - Lower entropy = more homogeneous (certain)
        - Higher entropy = more heterogeneous (uncertain)
        
        Args:
            data: DataFrame containing the dataset
            target_attribute: The target column name
            
        Returns:
            Entropy value (0 to log2(num_classes))
        """
        if len(data) == 0:
            return 0
        
        # Count occurrences of each class
        value_counts = data[target_attribute].value_counts()
        total = len(data)
        
        # Calculate entropy
        entropy = 0
        for count in value_counts:
            probability = count / total
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def calculate_information_gain(self, data, attribute, target_attribute):
        """
        Calculate information gain for an attribute.
        
        Information Gain measures how much information an attribute provides
        about the target class. It's the reduction in entropy.
        
        IG(S, A) = H(S) - Σ (|Sv| / |S|) * H(Sv)
        
        Where:
        - S is the dataset
        - A is the attribute
        - Sv is the subset of S where attribute A has value v
        - H(S) is the entropy of S
        
        Higher information gain = better attribute for splitting
        
        Args:
            data: DataFrame containing the dataset
            attribute: The attribute to evaluate
            target_attribute: The target column name
            
        Returns:
            Information gain value
        """
        # Calculate entropy of the entire dataset
        total_entropy = self.calculate_entropy(data, target_attribute)
        
        # Calculate weighted average entropy after splitting by attribute
        values = data[attribute].unique()
        weighted_entropy = 0
        
        for value in values:
            subset = data[data[attribute] == value]
            weight = len(subset) / len(data)
            weighted_entropy += weight * self.calculate_entropy(subset, target_attribute)
        
        # Information gain is the reduction in entropy
        information_gain = total_entropy - weighted_entropy
        
        return information_gain
    
    def get_best_attribute(self, data, attributes, target_attribute):
        """
        Select the attribute with the highest information gain.
        
        This is the key to the divide and conquer strategy:
        - We evaluate each attribute
        - Choose the one that best separates the data
        - Split on that attribute (divide)
        - Recursively build subtrees (conquer)
        
        Args:
            data: DataFrame containing the dataset
            attributes: List of available attributes
            target_attribute: The target column name
            
        Returns:
            Tuple of (best_attribute, information_gains_dict)
        """
        information_gains = {}
        
        for attribute in attributes:
            ig = self.calculate_information_gain(data, attribute, target_attribute)
            information_gains[attribute] = ig
        
        # Select attribute with maximum information gain
        best_attribute = max(information_gains, key=information_gains.get)
        
        return best_attribute, information_gains
    
    def build_tree(self, data, attributes, target_attribute, depth=0, parent_value=None):
        """
        Build decision tree using divide and conquer approach.
        
        DIVIDE AND CONQUER STRATEGY:
        1. BASE CASES (stop dividing):
           - All instances have same class → leaf node with that class
           - No more attributes → leaf node with majority class
           - No data → leaf node with parent's majority class
        
        2. DIVIDE:
           - Select best attribute using information gain
           - Split data into subsets based on attribute values
        
        3. CONQUER:
           - Recursively build subtrees for each subset
           - Each subtree solves a smaller version of the problem
        
        Args:
            data: DataFrame containing the dataset
            attributes: List of available attributes
            target_attribute: The target column name
            depth: Current tree depth (for visualization)
            parent_value: Value from parent split (for node labeling)
            
        Returns:
            DecisionTreeNode representing the tree/subtree
        """
        # BASE CASE 1: If all examples have same class, return leaf
        if len(data[target_attribute].unique()) == 1:
            label = data[target_attribute].iloc[0]
            return DecisionTreeNode(value=parent_value, label=label)
        
        # BASE CASE 2: If no attributes left, return leaf with majority class
        if len(attributes) == 0:
            majority_class = data[target_attribute].mode()[0]
            return DecisionTreeNode(value=parent_value, label=majority_class)
        
        # BASE CASE 3: If no data, return leaf with majority class from parent
        if len(data) == 0:
            return DecisionTreeNode(value=parent_value, label=None)
        
        # DIVIDE: Select best attribute to split on
        best_attribute, info_gains = self.get_best_attribute(data, attributes, target_attribute)
        
        # Create node for this attribute
        node = DecisionTreeNode(attribute=best_attribute, value=parent_value)
        
        # Print information for understanding
        indent = "  " * depth
        print(f"{indent}[Depth {depth}] Splitting on '{best_attribute}'")
        print(f"{indent}  Information Gains: {info_gains}")
        print(f"{indent}  Selected: {best_attribute} (IG={info_gains[best_attribute]:.4f})")
        
        # Get unique values of the best attribute
        attribute_values = data[best_attribute].unique()
        
        # CONQUER: Recursively build subtrees for each value
        remaining_attributes = [attr for attr in attributes if attr != best_attribute]
        
        for value in attribute_values:
            # Create subset where attribute has this value
            subset = data[data[best_attribute] == value]
            
            print(f"{indent}  Branch '{value}': {len(subset)} samples")
            
            # Recursively build subtree
            child_node = self.build_tree(
                subset, 
                remaining_attributes, 
                target_attribute, 
                depth + 1,
                value
            )
            
            node.children[value] = child_node
        
        return node
    
    def fit(self, data, target_attribute):
        """
        Train the decision tree on the dataset.
        
        Args:
            data: DataFrame containing the dataset
            target_attribute: The target column name
        """
        self.target_attribute = target_attribute
        attributes = [col for col in data.columns if col != target_attribute]
        
        print("="*70)
        print("BUILDING DECISION TREE USING ENTROPY AND INFORMATION GAIN")
        print("="*70)
        print(f"\nDataset size: {len(data)} samples")
        print(f"Attributes: {attributes}")
        print(f"Target: {target_attribute}")
        print(f"\nInitial entropy: {self.calculate_entropy(data, target_attribute):.4f}")
        print("\n" + "="*70)
        print("TREE CONSTRUCTION (Divide and Conquer)")
        print("="*70 + "\n")
        
        self.root = self.build_tree(data, attributes, target_attribute)
        
        print("\n" + "="*70)
        print("TREE CONSTRUCTION COMPLETE")
        print("="*70)
    
    def predict_sample(self, sample, node=None):
        """Predict class for a single sample"""
        if node is None:
            node = self.root
        
        # If leaf node, return the label
        if node.is_leaf():
            return node.label
        
        # Get the value of the splitting attribute for this sample
        attribute_value = sample[node.attribute]
        
        # Traverse to the appropriate child
        if attribute_value in node.children:
            return self.predict_sample(sample, node.children[attribute_value])
        else:
            # If value not seen during training, return None or majority class
            return None
    
    def predict(self, data):
        """Predict classes for a dataset"""
        predictions = []
        for _, sample in data.iterrows():
            prediction = self.predict_sample(sample)
            predictions.append(prediction)
        return predictions
    
    def print_tree(self, node=None, depth=0, value_from_parent="Root"):
        """Print the decision tree structure"""
        if node is None:
            node = self.root
        
        indent = "  " * depth
        
        if node.is_leaf():
            print(f"{indent}└─ [{value_from_parent}] → Predict: {node.label}")
        else:
            if depth == 0:
                print(f"{indent}[{node.attribute}]")
            else:
                print(f"{indent}└─ [{value_from_parent}] → Split on: {node.attribute}")
            
            for value, child in node.children.items():
                self.print_tree(child, depth + 1, value)


def demonstrate_concepts(tree, data, target_attribute):
    """
    Demonstrate key concepts with examples
    """
    print("\n" + "="*70)
    print("UNDERSTANDING THE CONCEPTS")
    print("="*70)
    
    print("\n1. ENTROPY - Measure of Uncertainty")
    print("-" * 70)
    print("Entropy quantifies how mixed/impure a dataset is:")
    print("  - Entropy = 0: All samples belong to one class (perfectly pure)")
    print("  - Entropy = 1: Samples equally distributed among classes (maximum uncertainty)")
    print("\nFormula: H(S) = -Σ p(i) * log2(p(i))")
    
    # Calculate entropy for different subsets
    print("\nExamples from PlayTennis dataset:")
    
    # Overall entropy
    overall_entropy = tree.calculate_entropy(data, target_attribute)
    yes_count = len(data[data[target_attribute] == 'Yes'])
    no_count = len(data[data[target_attribute] == 'No'])
    print(f"  Entire dataset: {yes_count} Yes, {no_count} No")
    print(f"  → Entropy = {overall_entropy:.4f}")
    
    # Entropy for Outlook=Sunny
    sunny_data = data[data['Outlook'] == 'Sunny']
    if len(sunny_data) > 0:
        sunny_entropy = tree.calculate_entropy(sunny_data, target_attribute)
        sunny_yes = len(sunny_data[sunny_data[target_attribute] == 'Yes'])
        sunny_no = len(sunny_data[sunny_data[target_attribute] == 'No'])
        print(f"\n  When Outlook=Sunny: {sunny_yes} Yes, {sunny_no} No")
        print(f"  → Entropy = {sunny_entropy:.4f}")
    
    # Entropy for Outlook=Overcast
    overcast_data = data[data['Outlook'] == 'Overcast']
    if len(overcast_data) > 0:
        overcast_entropy = tree.calculate_entropy(overcast_data, target_attribute)
        overcast_yes = len(overcast_data[overcast_data[target_attribute] == 'Yes'])
        overcast_no = len(overcast_data[overcast_data[target_attribute] == 'No'])
        print(f"\n  When Outlook=Overcast: {overcast_yes} Yes, {overcast_no} No")
        print(f"  → Entropy = {overcast_entropy:.4f} (Pure! All Yes)")
    
    print("\n\n2. INFORMATION GAIN - Reduction in Entropy")
    print("-" * 70)
    print("Information Gain measures how much an attribute reduces uncertainty:")
    print("  - Higher IG = better attribute for classification")
    print("  - We choose the attribute with maximum IG at each step")
    print("\nFormula: IG(S,A) = H(S) - Σ (|Sv|/|S|) * H(Sv)")
    print("  where Sv = subset of S where attribute A has value v")
    
    attributes = [col for col in data.columns if col != target_attribute]
    print("\nInformation Gain for each attribute:")
    for attr in attributes:
        ig = tree.calculate_information_gain(data, attr, target_attribute)
        print(f"  {attr:15s}: {ig:.4f}")
    
    print("\n\n3. DIVIDE AND CONQUER - Building the Tree")
    print("-" * 70)
    print("The algorithm recursively partitions the data:")
    print("\n  DIVIDE:")
    print("    1. Calculate information gain for all attributes")
    print("    2. Choose attribute with highest IG")
    print("    3. Split dataset based on attribute values")
    print("\n  CONQUER:")
    print("    4. Recursively build subtrees for each subset")
    print("    5. Stop when subset is pure or no attributes left")
    print("\n  This greedy approach builds the tree top-down,")
    print("  making locally optimal choices at each node.")
    
    print("\n\n4. FINAL DECISION TREE STRUCTURE")
    print("-" * 70)
    tree.print_tree()


def main():
    """Main function to run the decision tree implementation"""
    
    # Load the PlayTennis dataset
    print("Loading PlayTennis dataset...")
    data = pd.read_csv('PlayTennis.csv')
    
    print("\nDataset Preview:")
    print(data)
    print(f"\nShape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    # Create and train decision tree
    tree = DecisionTree()
    tree.fit(data, 'PlayTennis')
    
    # Demonstrate concepts
    demonstrate_concepts(tree, data, 'PlayTennis')
    
    # Test predictions
    print("\n\n5. PREDICTIONS")
    print("-" * 70)
    print("Testing the decision tree on training data:\n")
    
    predictions = tree.predict(data)
    correct = sum(1 for i, pred in enumerate(predictions) if pred == data['PlayTennis'].iloc[i])
    accuracy = correct / len(data) * 100
    
    print(f"Accuracy on training data: {correct}/{len(data)} ({accuracy:.1f}%)")
    
    # Show some example predictions
    print("\nExample predictions:")
    for i in range(min(5, len(data))):
        sample = data.iloc[i]
        pred = predictions[i]
        actual = sample['PlayTennis']
        status = "✓" if pred == actual else "✗"
        print(f"  {status} Sample {i+1}: Outlook={sample['Outlook']}, Temp={sample['Temperature']}, "
              f"Humidity={sample['Humidity']}, Wind={sample['Wind']}")
        print(f"      → Predicted: {pred}, Actual: {actual}")
    
    # Test with new sample
    print("\n\nTest with a new sample:")
    new_sample = pd.DataFrame([{
        'Outlook': 'Sunny',
        'Temperature': 'Cool',
        'Humidity': 'High',
        'Wind': 'Strong'
    }])
    prediction = tree.predict(new_sample)[0]
    print(f"  New sample: {new_sample.iloc[0].to_dict()}")
    print(f"  → Prediction: {prediction}")
    
    print("\n" + "="*70)
    print("HOW ENTROPY ADVANTAGES PROBLEM SOLVING")
    print("="*70)
    print("""
1. QUANTITATIVE MEASURE: Entropy provides a mathematical way to measure
   uncertainty, allowing us to make data-driven decisions.

2. OPTIMAL SPLITTING: By maximizing information gain (reducing entropy),
   we ensure each split provides maximum classification power.

3. GENERALIZATION: The greedy approach avoids overfitting by choosing
   attributes that best separate the overall data, not just specific cases.

4. EFFICIENCY: Divide and conquer reduces a complex problem (classify all data)
   into simpler subproblems (classify homogeneous subsets).

5. INTERPRETABILITY: The resulting tree is human-readable and shows
   the decision-making logic clearly.
""")
    print("="*70)


if __name__ == "__main__":
    main()
