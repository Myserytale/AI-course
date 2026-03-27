import numpy as np
import pandas as pd

# Load PlayTennis dataset
df = pd.read_csv('PlayTennis.csv')

class DecisionNode:
    """Node in the decision tree"""
    def __init__(self, attribute=None, value=None, label=None):
        self.attribute = attribute  # Attribute to split on (for internal nodes)
        self.value = value          # Value that led to this node (from parent)
        self.label = label          # Class label (for leaf nodes)
        self.children = {}          # Dictionary: value -> child node
        
    def is_leaf(self):
        return self.label is not None
    
    def add_child(self, value, child_node):
        self.children[value] = child_node

def calc_entropy(data, target_column):
    """
    Calculate entropy for a dataset
    S = -sum(p_i * log2(p_i))
    """
    if len(data) == 0:
        return 0
    
    values, counts = np.unique(data[target_column], return_counts=True)
    probabilities = counts / len(data)
    
    # Filter out zero probabilities to avoid log(0)
    probabilities = probabilities[probabilities > 0]
    
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def calc_information_gain(data, attribute, target_column):
    """
    Calculate information gain for a given attribute
    IG(S, A) = H(S) - sum((|S_v| / |S|) * H(S_v))
    """
    total_entropy = calc_entropy(data, target_column)
    
    values = data[attribute].unique()
    weighted_entropy = 0
    
    for value in values:
        subset = data[data[attribute] == value]
        weight = len(subset) / len(data)
        weighted_entropy += weight * calc_entropy(subset, target_column)
    
    information_gain = total_entropy - weighted_entropy
    return information_gain

def get_most_common_label(data, target_column):
    """Return the most common class label in the data"""
    return data[target_column].mode()[0]

def build_decision_tree(data, attributes, target_column, parent_value=None, depth=0):
    """
    Build decision tree using divide and conquer (ID3 algorithm)
    
    Base cases:
    1. If all instances have the same class -> return leaf with that class
    2. If no more attributes to split on -> return leaf with most common class
    3. If no instances -> return leaf with most common class from parent
    
    Recursive case:
    - Find best attribute (highest information gain)
    - Create node for that attribute
    - For each value of the attribute:
        - Create subset of data with that value
        - Recursively build subtree
    """
    indent = "  " * depth
    
    # Base case 1: All instances have the same class
    if len(data[target_column].unique()) == 1:
        label = data[target_column].iloc[0]
        print(f"{indent}→ Leaf: {label} (all same class, {len(data)} instances)")
        return DecisionNode(label=label, value=parent_value)
    
    # Base case 2: No more attributes to split on
    if len(attributes) == 0:
        label = get_most_common_label(data, target_column)
        print(f"{indent}→ Leaf: {label} (no more attributes, {len(data)} instances)")
        return DecisionNode(label=label, value=parent_value)
    
    # Base case 3: No instances (shouldn't happen, but handle it)
    if len(data) == 0:
        print(f"{indent}→ Leaf: No data")
        return DecisionNode(label="Unknown", value=parent_value)
    
    # Recursive case: Find best attribute and split
    best_attribute = max(attributes, 
                        key=lambda attr: calc_information_gain(data, attr, target_column))
    best_ig = calc_information_gain(data, best_attribute, target_column)
    
    print(f"{indent}Split on: {best_attribute} (IG={best_ig:.4f}, {len(data)} instances)")
    
    # Create node for this attribute
    node = DecisionNode(attribute=best_attribute, value=parent_value)
    
    # Remove the best attribute from the list for recursive calls
    remaining_attributes = [attr for attr in attributes if attr != best_attribute]
    
    # For each value of the best attribute, create a subtree
    for value in sorted(data[best_attribute].unique()):
        print(f"{indent}  Branch: {best_attribute} = {value}")
        subset = data[data[best_attribute] == value]
        
        # Recursively build subtree
        child = build_decision_tree(subset, remaining_attributes, target_column, 
                                   value, depth + 1)
        node.add_child(value, child)
    
    return node

def print_tree(node, depth=0, branch_value=None):
    """Print the decision tree in a readable format"""
    indent = "  " * depth
    
    if branch_value:
        print(f"{indent}[{branch_value}]", end=" ")
    
    if node.is_leaf():
        print(f"→ {node.label}")
    else:
        if depth == 0:
            print(f"{node.attribute}?")
        else:
            print(f"→ {node.attribute}?")
        
        for value, child in sorted(node.children.items()):
            print_tree(child, depth + 1, value)

def predict(tree, instance):
    """Predict the class for a single instance"""
    node = tree
    
    while not node.is_leaf():
        attribute = node.attribute
        value = instance.get(attribute)
        
        if value not in node.children:
            # Value not seen in training, return None
            return None
        
        node = node.children[value]
    
    return node.label

def test_tree(tree, data, target_column):
    """Test the decision tree on a dataset"""
    correct = 0
    total = len(data)
    
    for idx, row in data.iterrows():
        instance = row.to_dict()
        predicted = predict(tree, instance)
        actual = instance[target_column]
        
        if predicted == actual:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total

# Main execution
if __name__ == "__main__":
    print("="*60)
    print("PLAYTENNIS DECISION TREE - Divide and Conquer Algorithm")
    print("="*60)
    print("\nDataset:")
    print(df)
    print("\n" + "="*60)
    
    # Initial entropy
    target_entropy = calc_entropy(df, 'PlayTennis')
    print(f"\nInitial Entropy: {target_entropy:.4f}")
    
    # Show information gains
    attributes = ['Outlook', 'Temperature', 'Humidity', 'Wind']
    print("\nInformation Gain for each attribute:")
    for attribute in attributes:
        ig = calc_information_gain(df, attribute, 'PlayTennis')
        print(f"  {attribute}: {ig:.4f}")
    
    print("\n" + "="*60)
    print("BUILDING DECISION TREE (Divide and Conquer):")
    print("="*60 + "\n")
    
    # Build the decision tree
    tree = build_decision_tree(df, attributes, 'PlayTennis')
    
    print("\n" + "="*60)
    print("FINAL DECISION TREE:")
    print("="*60 + "\n")
    
    # Print the tree
    print_tree(tree)
    
    print("\n" + "="*60)
    print("TESTING ON TRAINING DATA:")
    print("="*60 + "\n")
    
    # Test the tree
    accuracy, correct, total = test_tree(tree, df, 'PlayTennis')
    print(f"Accuracy: {correct}/{total} ({accuracy*100:.1f}%)")
    
    print("\n" + "="*60)
    print("EXAMPLE PREDICTIONS:")
    print("="*60 + "\n")
    
    # Test with a few examples
    test_instances = [
        {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak'},
        {'Outlook': 'Overcast', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Wind': 'Strong'},
        {'Outlook': 'Rainy', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Weak'},
    ]
    
    for instance in test_instances:
        prediction = predict(tree, instance)
        print(f"Instance: {instance}")
        print(f"Prediction: {prediction}\n")

