import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


"""
███╗   ███╗███████╗████████╗██╗  ██╗ ██████╗ ██████╗ ███████╗
████╗ ████║██╔════╝╚══██╔══╝██║  ██║██╔═══██╗██╔══██╗██╔════╝
██╔████╔██║█████╗     ██║   ███████║██║   ██║██║  ██║███████╗
██║╚██╔╝██║██╔══╝     ██║   ██╔══██║██║   ██║██║  ██║╚════██║
██║ ╚═╝ ██║███████╗   ██║   ██║  ██║╚██████╔╝██████╔╝███████║
╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝
"""


def generate_splitpoints(df: pd.DataFrame, col: str) -> np.ndarray:
    """
    Generate split points for a numeric column in a pandas DataFrame.
    Computes possible split points by taking midpoints between every consecutive unique point.

    Parameters:
    ---
        df: pd.DataFrame
            Input DataFrame
    
        col: str
            Name of the column to generate splitpoints for

    Returns:
    ---
        ret: np.ndarray
            An array of n-1 floats storing the splitpoints, where n is the number of unique values in the column
    """
    assert df is not None, "Dataframe does not exist"
    assert col in list(df.columns), "The column is not in the dataframe"
    assert pd.api.types.is_numeric_dtype(df[col]), "The column is not numeric"

    vals = df[col].unique()
    vals.sort()
    n = len(vals)

    ret = np.zeros(n - 1)
    
    for i in range(n - 1):
        ret[i] = 1/2 * (vals[i] + vals[i+1])

    return ret

def entropy(counts: tuple) -> float:
    """
    Calculate entropy for a node with given class counts

    Parameters:
    ---
        counts: tuple
            A tuple representing the counts of classes in the node. For this homework (#T, #F) is used

    Returns:
    ---
        entropy_val: float
            The entropy of the node
    """
    
    total = sum(counts)
    if total == 0:
        return 0
    
    entropy_val = 0
    for count in counts:
        prob = count / total
        if prob > 0:  # Avoid log(0)
            entropy_val -= prob * np.log2(prob)
    
    return entropy_val

def gini(counts: tuple) -> float:
    """
    Calculate gini index for a node with given class counts

    Parameters:
    ---
        counts: tuple
            A tuple representing the counts of classes in the node. For this homework (#T, #F) is used

    Returns:
    ---
        gini_idx: float
            The gini index of the node        
    """
    total = sum(counts)
    if total == 0:
        return 0

    probs = [c/total for c in counts]
    gini_idx = 1 - sum(p ** 2 for p in probs)
    return gini_idx

def impurity_reduction(left: tuple, right: tuple, parent: tuple, criterion: str = "entropy") -> float:
    """
    Calculate impurity reduction from a split. Can use entropy or gini index as criterion
    
    Parameters:
    ---
        left: tuple
            A tuple of class counts in left child (#T, #F)
        right: tuple
            A tuple of class counts in right child (#T, #F)
        parent: tuple
            A tuple of class counts in parent node (#T, #F)
        criterio: str
            A string denoting what criterion to use to calculate impurity. Options are ["entropy", "gini"]
            
    Returns:
    ---
        IR: float
            Impurity reduction from the split
    """
    assert criterion in ["entropy", "gini"], "Invalid purity criterion"
    impurity_fxn = entropy if criterion == "entropy" else gini
    
    n_parent = sum(parent)
    n_left = sum(left)
    n_right = sum(right)
    
    if n_parent == 0:
        return 0
    
    # Parent impurity
    parent_impurity = impurity_fxn(parent)
    
    # Weighted average of children impurity
    left_weight = n_left / n_parent
    right_weight = n_right / n_parent
    
    children_impurity = left_weight * impurity_fxn(left) + right_weight * impurity_fxn(right)

    IR = parent_impurity - children_impurity
    return IR

def find_split(df: pd.DataFrame, split_col: str, label_col: str, criterion: str = "entropy"):
    # Cannot split
    if df.empty or df[split_col].nunique() <= 1:
        return None, 0.0
    
    splitpoints = generate_splitpoints(df, split_col)
    if len(splitpoints) == 0:
        return None, 0.0
    
    # Get subset for calculation
    data = df[[split_col, label_col]].to_numpy()

    # (#T, #F)
    parent_counts = (
        len(data[data[:, 1] == 1]),
        len(data[data[:, 1] == 0])
    )
    
    # Pure node already
    if parent_counts[0] == 0 or parent_counts[1] == 0:
        return None, 0.0

    best_reduction = float('-inf')
    best_split = None

    for sp in splitpoints:
        lt = data[data[:, 0] < sp]
        gt = data[data[:, 0] >= sp]
        
        # Skip splits that result in empty nodes
        if lt.shape[0] == 0 or gt.shape[0] == 0:
            continue

        split_lt = (len(lt[lt[:, 1] == 1]), len(lt[lt[:, 1] == 0]))
        split_gt = (len(gt[gt[:, 1] == 1]), len(gt[gt[:, 1] == 0]))

        # # Debug output
        # print(sp)
        # print(f"Left size: {lt.shape[0]}\t | \t(1, 0): {split_lt}")
        # print(f"Right size: {gt.shape[0]}\t | \t(1, 0): {split_gt}")
        # print("---"*30)

        reduction = impurity_reduction(split_lt, split_gt, parent_counts, criterion=criterion)

        if reduction > best_reduction:
            best_reduction = reduction
            best_split = sp

    # If no valid split was found
    if best_split is None:
        return None, 0.0
        
    return best_split, best_reduction


"""
 ██████╗██╗      █████╗ ███████╗███████╗███████╗███████╗
██╔════╝██║     ██╔══██╗██╔════╝██╔════╝██╔════╝██╔════╝
██║     ██║     ███████║███████╗███████╗█████╗  ███████╗
██║     ██║     ██╔══██║╚════██║╚════██║██╔══╝  ╚════██║
╚██████╗███████╗██║  ██║███████║███████║███████╗███████║
 ╚═════╝╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚══════╝
"""


class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.feature = None
        self.threshold = None
        self.is_leaf = False
        self.prediction = None


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, criteron="entropy"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.criterion = criteron
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        
        self.root = self._build_tree(X, y)
        return self
    
    def _build_tree(self, X: pd.DataFrame, y: pd.DataFrame, depth=0):
        n_samples = len(y)

        # Don't modify input
        data = X.copy()
        data["target"] = y
        
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
            n_samples < self.min_samples_split or \
            len(np.unique(y)) == 1:
                # Create leaf node
                leaf = Node()
                leaf.is_leaf = True
                leaf.prediction = np.argmax(np.bincount(y))
                return leaf
        
        # Find the best split across features
        best_feature = None
        best_threshold = None
        best_gain = -float('inf')
        
        for feature in X.columns:
            threshold, gain = find_split(data, feature, "target", criterion=self.criterion)
            
            if gain > best_gain and threshold is not None:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
        
        # If no good split is found, create a leaf node
        if best_feature is None:
            leaf = Node()
            leaf.is_leaf = True
            leaf.prediction = np.argmax(np.bincount(y))
            return leaf
        
        # Create decision node
        node = Node()
        node.feature = best_feature
        node.threshold = best_threshold
        
        # Split the data (get indices)
        left_mask = X[best_feature] < best_threshold
        right_mask = ~left_mask
        
        # Recursively build left and right subtrees
        node.left = self._build_tree(X[left_mask], y[left_mask], "target", depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], "target", depth + 1)
        
        return node
    
    def _predict_sample(self, x, node):
        """Predict class for a single sample"""
        if node.is_leaf:
            return node.prediction
        
        if x[node.feature] < node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X):
        """Predict classes for samples in X"""
        # Convert X to DataFrame if it's not already
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        return np.array([self._predict_sample(x, self.root) for _, x in X.iterrows()])
        


if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    print(df.head())


    # Preprocessing: Label Encoding
    label_mappings = {}
    categorical_cols = [
        "gender",
        "chest pain type", 
        "fasting blood sugar > 120 mg/dl", 
        "resting electrocardiographic results",
        "exercise induced angina",
        "Has heart disease? (Prediction Target)"
    ]
    encoded_df = df.copy()

    for col in categorical_cols:
        encoder = LabelEncoder()
        encoded_df[col] = encoder.fit_transform(encoded_df[col])
        label_mappings[col] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))