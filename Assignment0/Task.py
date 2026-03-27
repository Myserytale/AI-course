"""
Fuzzy Decision Tree Implementation for Continuous Data

This implementation transforms the categorical decision tree (ID3) to work with
continuous/numerical data using fuzzy logic concepts.

Key differences from crisp decision tree:
1. Uses fuzzy membership functions to categorize continuous values
2. Calculates fuzzy entropy using membership degrees as weights
3. Samples can partially belong to multiple branches
4. Final prediction aggregates memberships across all paths

Dataset: Iris (4 continuous features, 3 classes)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class FuzzySet:
    """
    Defines a fuzzy set with triangular membership function.
    
    Parameters:
    - name: Name of the fuzzy set (e.g., 'Low', 'Medium', 'High')
    - a, b, c: Parameters for triangular membership function
      a = left foot, b = peak, c = right foot
    """
    def __init__(self, name, a, b, c):
        self.name = name
        self.a = a  # left foot
        self.b = b  # peak
        self.c = c  # right foot
    
    def membership(self, x):
        """Calculate membership degree for value x"""
        if x <= self.a or x >= self.c:
            return 0.0
        elif self.a < x < self.b:
            return (x - self.a) / (self.b - self.a)
        elif x == self.b:
            return 1.0
        else:  # self.b < x < self.c
            return (self.c - x) / (self.c - self.b)


class FuzzyVariable:
    """
    A fuzzy linguistic variable with multiple fuzzy sets.
    
    Creates Low, Medium, High fuzzy sets based on data range.
    """
    def __init__(self, name, data_min, data_max):
        self.name = name
        self.data_min = data_min
        self.data_max = data_max
        
        # Create overlapping triangular fuzzy sets
        range_val = data_max - data_min
        
        # Low: covers lower third with overlap
        self.low = FuzzySet('Low', 
                           data_min - range_val * 0.1,
                           data_min,
                           data_min + range_val * 0.5)
        
        # Medium: covers middle with overlap
        self.medium = FuzzySet('Medium',
                              data_min + range_val * 0.25,
                              data_min + range_val * 0.5,
                              data_min + range_val * 0.75)
        
        # High: covers upper third with overlap
        self.high = FuzzySet('High',
                            data_min + range_val * 0.5,
                            data_max,
                            data_max + range_val * 0.1)
        
        self.fuzzy_sets = {'Low': self.low, 'Medium': self.medium, 'High': self.high}
    
    def fuzzify(self, value):
        """Return membership degrees for all fuzzy sets"""
        return {name: fs.membership(value) for name, fs in self.fuzzy_sets.items()}


class FuzzyDecisionNode:
    """Node in the fuzzy decision tree"""
    def __init__(self, attribute=None, fuzzy_value=None, class_distribution=None):
        self.attribute = attribute          # Attribute to split on
        self.fuzzy_value = fuzzy_value      # Fuzzy set name that led to this node
        self.class_distribution = class_distribution  # For leaf nodes: {class: weight}
        self.children = {}                  # Dictionary: fuzzy_value -> child node
        self.is_leaf_node = False
        
    def is_leaf(self):
        return self.is_leaf_node
    
    def add_child(self, fuzzy_value, child_node):
        self.children[fuzzy_value] = child_node


def calc_fuzzy_entropy(memberships, labels, classes):
    """
    Calculate fuzzy entropy for a dataset.
    
    Uses membership degrees as weights:
    H_fuzzy = -sum(p_i * log2(p_i))
    where p_i = sum(membership for class i) / sum(all memberships)
    """
    total_membership = np.sum(memberships)
    if total_membership == 0:
        return 0
    
    entropy = 0
    for c in classes:
        # Sum of memberships for this class
        class_membership = np.sum(memberships[labels == c])
        p = class_membership / total_membership
        
        if p > 0:
            entropy -= p * np.log2(p)
    
    return entropy


def calc_fuzzy_information_gain(data, memberships, attribute, fuzzy_var, target_col, classes):
    """
    Calculate fuzzy information gain for a given attribute.
    
    FIG(S, A) = H_fuzzy(S) - sum((|S_v|_fuzzy / |S|_fuzzy) * H_fuzzy(S_v))
    
    where |S_v|_fuzzy is the sum of membership degrees for fuzzy set v
    """
    labels = data[target_col].values
    total_membership = np.sum(memberships)
    
    if total_membership == 0:
        return 0
    
    # Calculate current fuzzy entropy
    total_entropy = calc_fuzzy_entropy(memberships, labels, classes)
    
    # Calculate weighted entropy for each fuzzy set
    weighted_entropy = 0
    
    for fuzzy_name, fuzzy_set in fuzzy_var.fuzzy_sets.items():
        # Get membership degrees for this fuzzy set
        fuzzy_memberships = np.array([fuzzy_set.membership(v) for v in data[attribute]])
        
        # Combined membership (intersection with current memberships)
        combined_memberships = memberships * fuzzy_memberships
        subset_membership = np.sum(combined_memberships)
        
        if subset_membership > 0:
            weight = subset_membership / total_membership
            subset_entropy = calc_fuzzy_entropy(combined_memberships, labels, classes)
            weighted_entropy += weight * subset_entropy
    
    return total_entropy - weighted_entropy


def get_class_distribution(memberships, labels, classes):
    """Get weighted class distribution based on memberships"""
    distribution = {}
    total = np.sum(memberships)
    
    for c in classes:
        class_membership = np.sum(memberships[labels == c])
        distribution[c] = class_membership / total if total > 0 else 0
    
    return distribution


def build_fuzzy_decision_tree(data, memberships, fuzzy_variables, attributes, 
                              target_col, classes, depth=0, max_depth=5, 
                              min_membership=0.1, parent_value=None):
    """
    Build fuzzy decision tree using divide and conquer.
    
    Parameters:
    - data: DataFrame with the data
    - memberships: Array of membership degrees for each sample
    - fuzzy_variables: Dict mapping attribute names to FuzzyVariable objects
    - attributes: List of remaining attributes to consider
    - target_col: Name of target column
    - classes: List of class labels
    - depth: Current depth
    - max_depth: Maximum tree depth
    - min_membership: Minimum total membership to continue splitting
    - parent_value: Fuzzy value from parent (for printing)
    """
    indent = "  " * depth
    labels = data[target_col].values
    total_membership = np.sum(memberships)
    
    # Create node
    node = FuzzyDecisionNode(fuzzy_value=parent_value)
    
    # Base case 1: Max depth reached or too few samples
    if depth >= max_depth or total_membership < min_membership:
        distribution = get_class_distribution(memberships, labels, classes)
        node.class_distribution = distribution
        node.is_leaf_node = True
        winner = max(distribution, key=distribution.get)
        print(f"{indent}→ Leaf: {winner} (depth limit or low membership, total={total_membership:.2f})")
        return node
    
    # Base case 2: No more attributes
    if len(attributes) == 0:
        distribution = get_class_distribution(memberships, labels, classes)
        node.class_distribution = distribution
        node.is_leaf_node = True
        winner = max(distribution, key=distribution.get)
        print(f"{indent}→ Leaf: {winner} (no attributes)")
        return node
    
    # Base case 3: Check if one class dominates (>95%)
    distribution = get_class_distribution(memberships, labels, classes)
    max_prob = max(distribution.values())
    if max_prob > 0.95:
        node.class_distribution = distribution
        node.is_leaf_node = True
        winner = max(distribution, key=distribution.get)
        print(f"{indent}→ Leaf: {winner} (class dominates with {max_prob*100:.1f}%)")
        return node
    
    # Find best attribute based on fuzzy information gain
    best_attribute = None
    best_gain = -1
    
    for attr in attributes:
        gain = calc_fuzzy_information_gain(data, memberships, attr, 
                                          fuzzy_variables[attr], target_col, classes)
        if gain > best_gain:
            best_gain = gain
            best_attribute = attr
    
    # If no good split found, make leaf
    if best_gain <= 0.001:
        node.class_distribution = distribution
        node.is_leaf_node = True
        winner = max(distribution, key=distribution.get)
        print(f"{indent}→ Leaf: {winner} (no good split)")
        return node
    
    print(f"{indent}Split on: {best_attribute} (FIG={best_gain:.4f}, membership={total_membership:.2f})")
    
    node.attribute = best_attribute
    remaining_attributes = [a for a in attributes if a != best_attribute]
    fuzzy_var = fuzzy_variables[best_attribute]
    
    # Create child for each fuzzy set
    for fuzzy_name, fuzzy_set in fuzzy_var.fuzzy_sets.items():
        # Calculate memberships for this fuzzy set
        fuzzy_memberships = np.array([fuzzy_set.membership(v) for v in data[best_attribute]])
        combined_memberships = memberships * fuzzy_memberships
        
        subset_total = np.sum(combined_memberships)
        print(f"{indent}  Branch: {best_attribute} is {fuzzy_name} (membership={subset_total:.2f})")
        
        if subset_total > min_membership:
            child = build_fuzzy_decision_tree(
                data, combined_memberships, fuzzy_variables,
                remaining_attributes, target_col, classes,
                depth + 1, max_depth, min_membership, fuzzy_name
            )
            node.add_child(fuzzy_name, child)
        else:
            # Create leaf with current distribution
            dist = get_class_distribution(combined_memberships, labels, classes) if subset_total > 0 else distribution
            leaf = FuzzyDecisionNode(fuzzy_value=fuzzy_name, class_distribution=dist)
            leaf.is_leaf_node = True
            winner = max(dist, key=dist.get) if dist else "Unknown"
            print(f"{indent}    → Leaf: {winner} (low membership)")
            node.add_child(fuzzy_name, leaf)
    
    return node


def fuzzy_predict(tree, instance, fuzzy_variables, membership=1.0):
    """
    Predict class using fuzzy decision tree.
    
    Returns weighted class distribution by traversing ALL paths
    with their membership degrees.
    """
    if tree.is_leaf():
        # Return class distribution weighted by path membership
        return {c: p * membership for c, p in tree.class_distribution.items()}
    
    # Get fuzzy memberships for the splitting attribute
    attr = tree.attribute
    value = instance[attr]
    fuzzy_var = fuzzy_variables[attr]
    fuzzified = fuzzy_var.fuzzify(value)
    
    # Aggregate predictions from all children weighted by membership
    aggregated = {}
    
    for fuzzy_name, child in tree.children.items():
        child_membership = fuzzified[fuzzy_name]
        if child_membership > 0:
            child_pred = fuzzy_predict(child, instance, fuzzy_variables, 
                                      membership * child_membership)
            for c, prob in child_pred.items():
                aggregated[c] = aggregated.get(c, 0) + prob
    
    return aggregated


def predict_class(tree, instance, fuzzy_variables):
    """Get the predicted class label"""
    distribution = fuzzy_predict(tree, instance, fuzzy_variables)
    if not distribution:
        return None
    return max(distribution, key=distribution.get)


def print_fuzzy_tree(node, depth=0, branch_value=None):
    """Print the fuzzy decision tree"""
    indent = "  " * depth
    
    if branch_value:
        print(f"{indent}[{branch_value}]", end=" ")
    
    if node.is_leaf():
        dist_str = ", ".join([f"{c}: {p:.2f}" for c, p in node.class_distribution.items()])
        print(f"→ ({dist_str})")
    else:
        if depth == 0:
            print(f"{node.attribute}?")
        else:
            print(f"→ {node.attribute}?")
        
        for value, child in sorted(node.children.items()):
            print_fuzzy_tree(child, depth + 1, value)


def test_fuzzy_tree(tree, data, fuzzy_variables, target_col):
    """Test the fuzzy decision tree on a dataset"""
    correct = 0
    total = len(data)
    
    for idx, row in data.iterrows():
        instance = row.to_dict()
        predicted = predict_class(tree, instance, fuzzy_variables)
        actual = instance[target_col]
        
        if predicted == actual:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total


# Main execution
if __name__ == "__main__":
    print("=" * 70)
    print("FUZZY DECISION TREE - For Continuous/Numerical Data")
    print("=" * 70)
    
    # Load Iris dataset (continuous numerical data)
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 
                                          'petal_length', 'petal_width'])
    df['species'] = [iris.target_names[i] for i in iris.target]
    
    print("\nDataset: Iris (150 samples, 4 continuous features, 3 classes)")
    print(df.head(10))
    print(f"\nFeature statistics:")
    print(df.describe())
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['species'])
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    print(f"\nTraining set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    # Create fuzzy variables for each attribute
    attributes = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    fuzzy_variables = {}
    
    print("\n" + "=" * 70)
    print("FUZZY VARIABLES (Triangular Membership Functions):")
    print("=" * 70)
    
    for attr in attributes:
        data_min = df[attr].min()
        data_max = df[attr].max()
        fuzzy_variables[attr] = FuzzyVariable(attr, data_min, data_max)
        print(f"\n{attr} (range: {data_min:.2f} - {data_max:.2f}):")
        fv = fuzzy_variables[attr]
        print(f"  Low:    [{fv.low.a:.2f}, {fv.low.b:.2f}, {fv.low.c:.2f}]")
        print(f"  Medium: [{fv.medium.a:.2f}, {fv.medium.b:.2f}, {fv.medium.c:.2f}]")
        print(f"  High:   [{fv.high.a:.2f}, {fv.high.b:.2f}, {fv.high.c:.2f}]")
    
    # Get classes
    classes = df['species'].unique()
    target_col = 'species'
    
    # Initial memberships (all samples have full membership)
    initial_memberships = np.ones(len(train_df))
    
    print("\n" + "=" * 70)
    print("BUILDING FUZZY DECISION TREE:")
    print("=" * 70 + "\n")
    
    # Build the fuzzy decision tree
    tree = build_fuzzy_decision_tree(
        train_df, initial_memberships, fuzzy_variables,
        attributes, target_col, classes, max_depth=4
    )
    
    print("\n" + "=" * 70)
    print("FINAL FUZZY DECISION TREE:")
    print("=" * 70 + "\n")
    
    print_fuzzy_tree(tree)
    
    print("\n" + "=" * 70)
    print("TESTING:")
    print("=" * 70)
    
    # Test on training data
    train_acc, train_correct, train_total = test_fuzzy_tree(tree, train_df, fuzzy_variables, target_col)
    print(f"\nTraining Accuracy: {train_correct}/{train_total} ({train_acc*100:.1f}%)")
    
    # Test on test data
    test_acc, test_correct, test_total = test_fuzzy_tree(tree, test_df, fuzzy_variables, target_col)
    print(f"Test Accuracy: {test_correct}/{test_total} ({test_acc*100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTIONS WITH FUZZY MEMBERSHIPS:")
    print("=" * 70 + "\n")
    
    # Show a few example predictions with membership details
    for idx in [0, 50, 100]:
        if idx < len(df):
            instance = df.iloc[idx].to_dict()
            print(f"Sample {idx}: {instance['species']}")
            print(f"  Features: sepal_length={instance['sepal_length']:.1f}, "
                  f"sepal_width={instance['sepal_width']:.1f}, "
                  f"petal_length={instance['petal_length']:.1f}, "
                  f"petal_width={instance['petal_width']:.1f}")
            
            # Show fuzzification
            print(f"  Fuzzified values:")
            for attr in attributes:
                fuzz = fuzzy_variables[attr].fuzzify(instance[attr])
                non_zero = [(k, v) for k, v in fuzz.items() if v > 0.01]
                print(f"    {attr}: {non_zero}")
            
            # Get prediction
            distribution = fuzzy_predict(tree, instance, fuzzy_variables)
            predicted = max(distribution, key=distribution.get)
            print(f"  Prediction distribution: {distribution}")
            print(f"  Predicted: {predicted} (Actual: {instance['species']})")
            print()
