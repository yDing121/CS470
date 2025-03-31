import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, recall_score
import matplotlib.pyplot as plt
import time

from decision_tree import DecisionTree

def run_comparison_tests():
    # Test 1: Heart Disease Dataset
    print("=== Test 1: Heart Disease Dataset ===")
    df = pd.read_csv("data.csv")
    
    # Prepare data
    categorical_cols = [
        "gender", "chest pain type", "fasting blood sugar > 120 mg/dl",
        "resting electrocardiographic results", "exercise induced angina"
    ]
    X = df.drop("Has heart disease? (Prediction Target)", axis=1)
    y = df["Has heart disease? (Prediction Target)"].map({"Presence": 1, "Absence": 0})
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate trees
    custom_tree = DecisionTree(max_depth=5)
    sklearn_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    
    # Custom tree
    start_time = time.time()
    custom_tree.fit(X_train, y_train)
    custom_train_time = time.time() - start_time
    custom_pred = custom_tree.predict(X_test)
    custom_accuracy = accuracy_score(y_test, custom_pred)
    
    # Sklearn tree
    start_time = time.time()
    sklearn_tree.fit(X_train, y_train)
    sklearn_train_time = time.time() - start_time
    sklearn_pred = sklearn_tree.predict(X_test)
    sklearn_accuracy = accuracy_score(y_test, sklearn_pred)
    
    print("\nHeart Disease Dataset Results:")
    print(f"Custom Tree Accuracy: {custom_accuracy:.4f} (Training time: {custom_train_time:.4f}s)")
    print(f"Sklearn Tree Accuracy: {sklearn_accuracy:.4f} (Training time: {sklearn_train_time:.4f}s)")
    
    # Test 2: Breast Cancer Dataset
    print("\n=== Test 2: Breast Cancer Dataset ===")
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to DataFrame for custom tree
    X_train_df = pd.DataFrame(X_train, columns=cancer.feature_names)
    X_test_df = pd.DataFrame(X_test, columns=cancer.feature_names)
    
    # Train and evaluate
    custom_tree = DecisionTree(max_depth=5)
    sklearn_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    
    custom_tree.fit(X_train_df, y_train)
    custom_pred = custom_tree.predict(X_test_df)
    custom_accuracy = accuracy_score(y_test, custom_pred)
    
    sklearn_tree.fit(X_train, y_train)
    sklearn_pred = sklearn_tree.predict(X_test)
    sklearn_accuracy = accuracy_score(y_test, sklearn_pred)
    
    print("\nBreast Cancer Dataset Results:")
    print(f"Custom Tree Accuracy: {custom_accuracy:.4f}")
    print(f"Sklearn Tree Accuracy: {sklearn_accuracy:.4f}")
    print("\nCustom Tree Classification Report:")
    print(classification_report(y_test, custom_pred))
    
    # Test 3: Moons Dataset (Nonlinear Decision Boundary)
    print("\n=== Test 3: Moons Dataset ===")
    X, y = make_moons(n_samples=1000, noise=0.3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to DataFrame
    X_train_df = pd.DataFrame(X_train, columns=['Feature 1', 'Feature 2'])
    X_test_df = pd.DataFrame(X_test, columns=['Feature 1', 'Feature 2'])
    
    # Train and evaluate
    custom_tree = DecisionTree(max_depth=5)
    sklearn_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    
    custom_tree.fit(X_train_df, y_train)
    custom_pred = custom_tree.predict(X_test_df)
    custom_accuracy = accuracy_score(y_test, custom_pred)
    
    sklearn_tree.fit(X_train, y_train)
    sklearn_pred = sklearn_tree.predict(X_test)
    sklearn_accuracy = accuracy_score(y_test, sklearn_pred)
    
    print("\nMoons Dataset Results:")
    print(f"Custom Tree Accuracy: {custom_accuracy:.4f}")
    print(f"Sklearn Tree Accuracy: {sklearn_accuracy:.4f}")
    
    # Visualize decision boundaries for Moons dataset
    plt.figure(figsize=(12, 5))
    
    # Plot custom tree decision boundary
    plt.subplot(121)
    plot_decision_boundary(custom_tree, X_test_df, y_test, "Custom Tree")
    
    # Plot sklearn tree decision boundary
    plt.subplot(122)
    plot_decision_boundary(sklearn_tree, X_test, y_test, "Sklearn Tree")
    
    plt.tight_layout()
    plt.savefig('decision_boundaries.png')
    plt.show()

def plot_decision_boundary(model, X, y, title):
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    if isinstance(X, pd.DataFrame):
        mesh_df = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=X.columns)
        Z = model.predict(mesh_df)
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.title(title)

if __name__ == "__main__":
    # run_comparison_tests()
    df = pd.read_csv("data.csv")

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

    X_ID = encoded_df.drop(["Has heart disease? (Prediction Target)"], axis=1)
    y = encoded_df["Has heart disease? (Prediction Target)"]

    X_train, X_test, y_train, y_test = train_test_split(X_ID, y, test_size=0.2, random_state=42)

    train_IDs = X_train["person ID"]
    test_IDs = X_test["person ID"]

    X_train = X_train.drop(["person ID"], axis=1)
    X_test = X_test.drop(["person ID"], axis=1)

    train_IDs.to_csv("para2_file.txt")
    test_IDs.to_csv("para3_file.txt")

    print(X_train.shape)
    X_train.head()

    myTree = DecisionTree(
        max_depth=5,
        min_samples_split=2,
        criteron="entropy"
    )

    myTree.fit(X_train, y_train)

    myPreds = myTree.predict(X_test)
    myAccuracy = accuracy_score(y_test, myPreds)
    print(f"My accuracy: {myAccuracy}")

    myRecall = recall_score(y_test, myPreds)
    print(f"My recall: {myRecall}")