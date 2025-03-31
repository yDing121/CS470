import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, fbeta_score
import argparse
import matplotlib.pyplot as plt
import shap


"""
███╗   ███╗███████╗████████╗██╗  ██╗ ██████╗ ██████╗ ███████╗
████╗ ████║██╔════╝╚══██╔══╝██║  ██║██╔═══██╗██╔══██╗██╔════╝
██╔████╔██║█████╗     ██║   ███████║██║   ██║██║  ██║███████╗
██║╚██╔╝██║██╔══╝     ██║   ██╔══██║██║   ██║██║  ██║╚════██║
██║ ╚═╝ ██║███████╗   ██║   ██║  ██║╚██████╔╝██████╔╝███████║
╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═════╝ ╚═════╝ ╚══════╝
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


def impurity_reduction(left: tuple, right: tuple, parent: tuple, criterion: str = "gain_ratio") -> float:
    """
    Calculate impurity reduction from a split. Can use entropy, gini, or gain_ratio as criterion.
    
    Parameters:
    ---
        left: tuple
            A tuple of class counts in left child (#T, #F)
        right: tuple
            A tuple of class counts in right child (#T, #F)
        parent: tuple
            A tuple of class counts in parent node (#T, #F)
        criterion: str
            A string denoting what criterion to use to calculate impurity.
            Options are ["entropy", "gini", "gain_ratio"].
            
    Returns:
    ---
        IR: float
            Impurity reduction (or gain ratio) from the split
    """
    assert criterion in ["entropy", "gini", "gain_ratio"], "Invalid purity criterion"
    n_parent = sum(parent)
    n_left = sum(left)
    n_right = sum(right)
    
    if n_parent == 0:
        return 0
    
    if criterion == "gain_ratio":
        # Use entropy as base impurity measure
        parent_impurity = entropy(parent)
        left_weight = n_left / n_parent
        right_weight = n_right / n_parent
        children_impurity = left_weight * entropy(left) + right_weight * entropy(right)
        info_gain = parent_impurity - children_impurity
        
        # Calculate split information
        def safe_log(p):
            return np.log2(p) if p > 0 else 0
        split_info = - (left_weight * safe_log(left_weight) + right_weight * safe_log(right_weight))
        if split_info == 0:
            return 0
        return info_gain / split_info
    else:
        impurity_fxn = entropy if criterion == "entropy" else gini
        parent_impurity = impurity_fxn(parent)
        left_weight = n_left / n_parent
        right_weight = n_right / n_parent
        children_impurity = left_weight * impurity_fxn(left) + right_weight * impurity_fxn(right)
        return parent_impurity - children_impurity


def find_split(df: pd.DataFrame, split_col: str, label_col: str, criterion: str = "entropy"):
    """
    Find the best split for a given column in a DataFrame based on impurity reduction.

    Parameters:
    ---
        df: pd.DataFrame
            The DataFrame containing the data to be split.

        split_col: str
            The column name to split on.
        
        label_col: str
            The column name of the target variable used for calculating class counts. 

        criterion: str
            The criterion to use for impurity calculation. Options are "entropy" or "gini". Default is "entropy".

    Returns:
    ---
        best_split: float or None
            The best split point found for the given column. Returns None if no valid split exists.

        best_reduction: float
            The impurity reduction achieved by the best split. Returns 0.0 if no valid split exists.
    """

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

        reduction = impurity_reduction(split_lt, split_gt, parent_counts, criterion=criterion)

        if reduction > best_reduction:
            best_reduction = reduction
            best_split = sp

    # If no valid split was found
    if best_split is None:
        return None, 0.0
        
    return best_split, best_reduction
        

def compute_metrics(y_true, y_pred):
    """
    Compute evaluation metrics for classification results.

    Parameters:
    ---
        y_true: np.ndarray or pd.Series
            True labels of the data.
        y_pred: np.ndarray or pd.Series
            Predicted labels from the model.

    Returns:
    ---
        accuracy: float
            Accuracy of the predictions.
        precision: float
            Precision of the predictions.
        recall: float
            Recall of the predictions.
        specificity: float
            Specificity of the predictions (True Negative Rate).
        f1: float
            F1 score of the predictions.
        f2: float
            F2 score of the predictions (weighted to prioritize recall).
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    f2 = fbeta_score(y_true, y_pred, beta=2)
    
    # Specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)

    return accuracy, precision, recall, specificity, f1, f2


def k_fold_cross_validation(model, X: pd.DataFrame, y: pd.Series, k: int = 5):
    """
    Perform stratified k-fold cross-validation on the given model and dataset.

    Parameters:
    ---
        model: DecisionTree
            The decision tree model to evaluate.
        X: pd.DataFrame
            Feature matrix.
        y: pd.Series
            Target vector.
        k: int
            Number of folds for cross-validation.

    Returns:
    ---
        metrics: dict
            A dictionary containing lists of evaluation metrics for each fold:
            - accuracy: List of accuracy scores.
            - precision: List of precision scores.
            - recall: List of recall scores.
            - specificity: List of specificity scores.
            - f1: List of F1 scores.
            - f2: List of F2 scores (weighted to prioritize recall).
    """
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "specificity": [],
        "f1": [],
        "f2": []
    }

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        acc, prec, rec, spec, f1, f2 = compute_metrics(y_test, predictions)

        metrics["accuracy"].append(acc)
        metrics["precision"].append(prec)
        metrics["recall"].append(rec)
        metrics["specificity"].append(spec)
        metrics["f1"].append(f1)
        metrics["f2"].append(f2)

    return metrics


"""
 ██████╗██╗      █████╗ ███████╗███████╗███████╗███████╗
██╔════╝██║     ██╔══██╗██╔════╝██╔════╝██╔════╝██╔════╝
██║     ██║     ███████║███████╗███████╗█████╗  ███████╗
██║     ██║     ██╔══██║╚════██║╚════██║██╔══╝  ╚════██║
╚██████╗███████╗██║  ██║███████║███████║███████╗███████║
 ╚═════╝╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚══════╝
"""


class Node:
    """Node class for the decision tree"""
    def __init__(self):
        self.left = None
        self.right = None
        self.feature = None
        self.threshold = None
        self.is_leaf = False
        self.prediction = None
    
    def __str__(self):
        """
        String representation of the Node for debugging purposes.
        If it's a leaf, it shows the prediction; otherwise, it shows the feature and threshold.
        """
        if self.is_leaf:
            return f"Leaf(prediction={self.prediction})"
        else:
            return f"Node(feature={self.feature}, threshold={self.threshold})"


class DecisionTree:
    """Decision Tree Classifier"""
    def __init__(self, max_depth=None, min_samples_split=2, criterion="entropy"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.criterion = criterion
        self.feature_names = None
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """Fit the decision tree"""
        self.feature_names = X.columns.tolist()
        self.root = self._build_tree(X, y)
    
    def _build_tree(self, X: pd.DataFrame, y: pd.DataFrame, depth=0):
        """Recursively build tree node by node, preorder"""
        n_samples = len(y)

        # Don't modify input
        data = X.copy()
        data["target"] = y
        
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
            n_samples < self.min_samples_split or \
            len(np.unique(y)) == 1:
                # Stop, create leaf node
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
        
        # If no good split is found, return majority class as prediction
        if best_feature is None:
            leaf = Node()
            leaf.is_leaf = True
            leaf.prediction = np.argmax(np.bincount(y))
            return leaf
        
        # Create branching node
        node = Node()
        node.feature = best_feature
        node.threshold = best_threshold
        
        # Split the data (boolean indexing)
        left_mask = X[best_feature] < best_threshold
        right_mask = ~left_mask
        
        # Recursively build left and right subtrees
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return node
    
    def _predict_sample(self, x, node):
        """Predict class for a single sample"""
        # Base Case
        if node.is_leaf:
            return node.prediction
        
        # Traverse
        if x[node.feature] < node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict classes for samples in X"""            
        return np.array([self._predict_sample(x, self.root) for _, x in X.iterrows()])
    
    def shap_values(self, X: pd.DataFrame) -> tuple:
        """
        Calculate SHAP values for the decision tree.
        
        Parameters:
        ---
            X: pd.DataFrame
                Feature matrix to explain
                
        Returns:
        ---
            shap_values: np.ndarray
                SHAP values for each prediction
            expected_value: float
                The base value that would be predicted if we didn't know any features
        """
        # Create background dataset (training data)
        background = X.copy()
        
        def tree_predict(X):
            return self.predict(pd.DataFrame(X, columns=self.feature_names))
        
        # Create the explainer
        explainer = shap.KernelExplainer(tree_predict, background)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X)
        
        return shap_values, explainer.expected_value


if __name__ == "__main__":
    # Add argparse to handle input arguments with defaults
    parser = argparse.ArgumentParser(description="Decision Tree Classifier")
    parser.add_argument("--input_path", default="data.csv", help="Path to input dataset file")
    parser.add_argument("--p2_path", default="para2_file.txt", help="Path to para2_file.txt (training set person IDs)")
    parser.add_argument("--p3_path", default="para3_file.txt", help="Path to para3_file.txt (test set person IDs)")
    parser.add_argument("--p4_path", default="para4_file.txt", help="Path to para4_file.txt (test set IDs and predictions)")

    args = parser.parse_args()

    # Read input dataset
    df = pd.read_csv(args.input_path)

    # Encode categorical columns
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

    encoded_df = pd.get_dummies(encoded_df, columns=["chest pain type", "resting electrocardiographic results"])
    dummy_columns = [col for col in encoded_df.columns if col.startswith("chest pain type_") or col.startswith("resting electrocardiographic results_")]
    encoded_df[dummy_columns] = encoded_df[dummy_columns].astype(int)

    # Get p2 and p3
    with open(args.p2_path, "r") as f:
        train_ids = set(line.strip() for line in f)

    with open(args.p3_path, "r") as f:
        test_ids = set(line.strip() for line in f)

    # Split dataset into training and testing sets
    train_df = encoded_df[encoded_df["person ID"].astype(str).isin(train_ids)]
    test_df = encoded_df[encoded_df["person ID"].astype(str).isin(test_ids)]

    print("Training set size:", len(train_df))
    print("Testing set size:", len(test_df))

    # Train
    X_train = train_df.drop(columns=["Has heart disease? (Prediction Target)", "person ID"])
    y_train = train_df["Has heart disease? (Prediction Target)"]
    X_test = test_df.drop(columns=["Has heart disease? (Prediction Target)", "person ID"])
    y_test = test_df["Has heart disease? (Prediction Target)"]

    model = DecisionTree(max_depth=8, min_samples_split=2, criterion="gain_ratio")
    model.fit(X_train, y_train)

    # Predict on the test set
    predictions = model.predict(X_test)

    # Write predictions to para4_file.txt
    with open(args.p4_path, "w") as f:
        for PID, prediction in zip(test_df["person ID"], predictions):
            f.write(f"{PID} {'yes' if prediction == 1 else 'no'}\n")

    print("Predictions written to", args.p4_path)

    # Evaluate predictions
    accuracy, precision, recall, specificity, f1, f2 = compute_metrics(y_test, predictions)
    print(f"--------\nEvaluation Metrics:\n"
          f"Accuracy: {accuracy:.4f}\n"
          f"Precision: {precision:.4f}\n"
          f"Recall: {recall:.4f}\n"
          f"Specificity: {specificity:.4f}\n"
          f"F1 Score: {f1:.4f}\n"
          f"F2 Score: {f2:.4f}")