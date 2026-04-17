import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from torch import nn
import os
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
torch.manual_seed(25)
torch.mps.manual_seed(25)
from torchvision.io import read_image
device = "mps" if torch.mps.is_available() else "cpu"

#################################################
# Basic Utilities #
#################################################

def train_loop(train_loader, val_loader, model, loss_fn, optimizer, max_epoch):

    for iter in max_epoch:
        model.train()
        train_loss = 0
        for batch, (X,y) in enumerate(train_loader):
            X,y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch, (X,y) in enumerate(val_loader):
                X,y = X.to(device), y.to(device)
                pred = model(X)
                loss = loss_fn(pred, y)
                val_loss += loss.item()
            
        val_loss /= len(val_loader)

        print(f"Epoch {iter+1}/{max_epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

def test_loop(test_loader, model, loss_fn):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch, (X,y) in enumerate(test_loader):
            X,y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            test_loss += loss.item()
        
    test_loss /= len(test_loader)

    print(f"Test Loss: {test_loss:.4f}")



#################################################
# Phase 1 #
#################################################

class RandomForest():

    def __init__(self, n_trees = 100, max_depth = 3):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
    
    def gini(self, y):
        if len(y) == 0:
            return 0
        p = np.bincount(y) / len(y)
        return 1 - np.sum(p**2)
    
    def best_split_for_single_feature(self, X_col, y, number_of_thresholds=500):
        """Best split for a single continuous feature column X_col and local node labels y"""
        best_gini = float('inf')
        best_threshold = None

        thresholds = np.linspace(np.min(X_col), np.max(X_col), num=number_of_thresholds)
        for threshold in thresholds:
            left_idx = np.where(X_col < threshold)[0]
            right_idx = np.where(X_col >= threshold)[0]

            gini_left = self.gini(y[left_idx])
            gini_right = self.gini(y[right_idx])
            gini_split = (len(left_idx) * gini_left + len(right_idx) * gini_right) / len(y)

            if gini_split < best_gini:
                best_gini = gini_split
                best_threshold = threshold

        parent_gini = self.gini(y)
        return best_threshold, (parent_gini - best_gini)

    def best_split(self, X, y, available_features, number_of_thresholds=500):
        """Find the best feature and threshold among available_features only."""

        best_gini_gain = float('-inf')
        best_feature_idx = None
        best_threshold = None

        for feature_idx in available_features:
            threshold, gini_gain = self.best_split_for_single_feature(X[:, feature_idx], y, number_of_thresholds)
            if gini_gain > best_gini_gain:
                best_gini_gain = gini_gain
                best_threshold = threshold
                best_feature_idx = feature_idx

        return best_threshold, best_gini_gain, best_feature_idx

    def build_tree(self, X, y, depth=0, available_features=None):
        """Recursively build a decision tree. Once a feature is used anywhere, it is
        globally removed and cannot be reused by any other node in the tree."""

        if available_features is None:
            available_features = list(range(X.shape[1]))

        majority_class = int(np.bincount(y, minlength=2).argmax())

        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) == 0 or len(available_features) == 0:
            return {'leaf': True, 'value': majority_class}

        threshold, gini_gain, feature_idx = self.best_split(X, y, available_features)


        if gini_gain <= 0 or feature_idx is None:
            return {'leaf': True, 'value': majority_class}

        left_idx  = np.where(X[:, feature_idx] <  threshold)[0]
        right_idx = np.where(X[:, feature_idx] >= threshold)[0]

    
        if len(left_idx) == 0 or len(right_idx) == 0:
            return {'leaf': True, 'value': majority_class}

        available_features.remove(feature_idx)

        return {
            'leaf'        : False,
            'feature_idx' : feature_idx,
            'threshold'   : threshold,
            'left'        : self.build_tree(X[left_idx],  y[left_idx],  depth + 1, available_features),
            'right'       : self.build_tree(X[right_idx], y[right_idx], depth + 1, available_features),
        }

    def predict_single(self, node, x):
        """Traverse the tree for one sample x."""
        if node['leaf']:
            return node['value']
        if x[node['feature_idx']] < node['threshold']:
            return self.predict_single(node['left'], x)
        else:
            return self.predict_single(node['right'], x)

    def predict(self, tree, X):
        """Predict labels for all rows in X using a built tree."""
        return np.array([self.predict_single(tree, x) for x in X])

    def fit(self, X, y, n_features_subset = 4):
        """Fit the random forest model on the training data."""

        n_features = X.shape[1]

        for _ in range(self.n_trees):

            feature_subset = np.random.choice(n_features, size=n_features_subset, replace=False).tolist()

            tree = self.build_tree(X,y, available_features=feature_subset)
            self.trees.append(tree)

    def predict_forest(self, X):
        """Predict labels for all rows in X by aggregating predictions from all trees."""
        
        tree_preds = np.array([self.predict(tree, X) for tree in self.trees])
        
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=tree_preds)


class adaboost():

    def __init__(self, n_trees = 50, max_depth = 2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
        self.alphas = []

    def gini(self, y):
        if len(y) == 0:
            return 0
        p = np.bincount(y) / len(y)
        return 1 - np.sum(p**2)
    
    def best_split_for_single_feature(self, X_col, y, number_of_thresholds=500):
        """Best split for a single continuous feature column X_col and local node labels y"""
        best_gini = float('inf')
        best_threshold = None

        thresholds = np.linspace(np.min(X_col), np.max(X_col), num=number_of_thresholds)
        for threshold in thresholds:
            left_idx = np.where(X_col < threshold)[0]
            right_idx = np.where(X_col >= threshold)[0]

            gini_left = self.gini(y[left_idx])
            gini_right = self.gini(y[right_idx])
            gini_split = (len(left_idx) * gini_left + len(right_idx) * gini_right) / len(y)

            if gini_split < best_gini:
                best_gini = gini_split
                best_threshold = threshold

        parent_gini = self.gini(y)
        return best_threshold, (parent_gini - best_gini)

    def best_split(self, X, y, available_features, number_of_thresholds=500):
        """Find the best feature and threshold among available_features only."""

        best_gini_gain = float('-inf')
        best_feature_idx = None
        best_threshold = None

        for feature_idx in available_features:
            threshold, gini_gain = self.best_split_for_single_feature(X[:, feature_idx], y, number_of_thresholds)
            if gini_gain > best_gini_gain:
                best_gini_gain = gini_gain
                best_threshold = threshold
                best_feature_idx = feature_idx

        return best_threshold, best_gini_gain, best_feature_idx

    def build_tree(self, X, y, depth=0, available_features=None):
        """Recursively build a decision tree. Once a feature is used anywhere, it is
        globally removed and cannot be reused by any other node in the tree."""

        if available_features is None:
            available_features = list(range(X.shape[1]))  

        majority_class = int(np.bincount(y, minlength=2).argmax())


        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) == 0 or len(available_features) == 0:
            return {'leaf': True, 'value': majority_class}

        threshold, gini_gain, feature_idx = self.best_split(X, y, available_features)

        
        if gini_gain <= 0 or feature_idx is None:
            return {'leaf': True, 'value': majority_class}

        left_idx  = np.where(X[:, feature_idx] <  threshold)[0]
        right_idx = np.where(X[:, feature_idx] >= threshold)[0]


        if len(left_idx) == 0 or len(right_idx) == 0:
            return {'leaf': True, 'value': majority_class}

        available_features.remove(feature_idx)

        return {
            'leaf'        : False,
            'feature_idx' : feature_idx,
            'threshold'   : threshold,
            'left'        : self.build_tree(X[left_idx],  y[left_idx],  depth + 1, available_features),
            'right'       : self.build_tree(X[right_idx], y[right_idx], depth + 1, available_features),
        }

    def predict_single(self, node, x):
        """Traverse the tree for one sample x."""
        if node['leaf']:
            return node['value']
        if x[node['feature_idx']] < node['threshold']:
            return self.predict_single(node['left'], x)
        else:
            return self.predict_single(node['right'], x)

    def predict(self, tree, X):
        """Predict labels for all rows in X using a built tree."""
        return np.array([self.predict_single(tree, x) for x in X])
    
    def fit(self, X, y):
        N = len(y)
        weight = (1/N)*np.ones(N)
        idx = np.arange(N)
        weight_track = []
        misclassification_track_idx = []

        y_signed = np.where(y == 0, -1, 1).astype(float)

        for m in range(self.n_trees):
            data_subset_idx = np.random.choice(idx, size=N, replace=True, p=weight)
            X_subset = X[data_subset_idx]
            y_subset = y[data_subset_idx]

            tree = self.build_tree(X_subset, y_subset)
            self.trees.append(tree)

            y_pred = self.predict(tree, X)
            error = np.average(y != y_pred, weights=weight)

            alpha = 0.5 * np.log((1 - error) / error)
            self.alphas.append(alpha)

            weight_track.append(weight.copy())

            y_pred_signed = np.where(y_pred == 0, -1, 1).astype(float)
            weight *= np.exp(-alpha * y_signed * y_pred_signed)
            weight /= weight.sum() 

            misclassification_track_idx.append(np.where(y != y_pred)[0])

        return weight_track, misclassification_track_idx

    def predict_adaboost(self, X):
        """Predict labels for all rows in X by aggregating predictions from all trees."""

        tree_preds = np.array([np.where(self.predict(tree, X) == 0, -1, 1) for tree in self.trees])
        weighted_sum = np.array(self.alphas) @ tree_preds

        return np.where(weighted_sum >= 0, 1, 0)






if __name__ == "__main__":
    DATA_DIR = "/Users/dwaipayanhaldar/Downloads/Notes and Books/IISc Coding Assignments and Project/Data/PRNN_2026_A2_data"
    LABEL_COL = "pm2_5"
    FEATURE_COL = ["pm10", "no2", "so2", "co", "o3","nh3","no"]

    df = pd.read_csv(os.path.join(DATA_DIR, "delhi_aqi.csv"))

    label = np.array(df[LABEL_COL], dtype=np.float32)
    label = np.where(label > 200, 1, 0)
    feature = np.array(df[FEATURE_COL], dtype=np.float32)

    N = np.random.permutation(len(feature))

    train_idx = N[:int(0.7*len(feature))]
    val_idx = N[int(0.7*len(feature)):int(0.85*len(feature))]
    test_idx = N[int(0.85*len(feature)):]

    X_train, y_train = feature[train_idx], label[train_idx]
    X_val, y_val = feature[val_idx], label[val_idx]
    X_test, y_test = feature[test_idx], label[test_idx]

    # rf = RandomForest(n_trees= 100, max_depth=2)
    ab = adaboost(n_trees= 50, max_depth=2)

    weight_track, misclassification_track_idx = ab.fit(X_train, y_train)
    val_preds = ab.predict_adaboost(X_val)
    test_preds = ab.predict_adaboost(X_test)

    val_acc = np.mean(val_preds == y_val)
    test_acc = np.mean(test_preds == y_test)

    weight_track = np.array(weight_track)
    # misclassification_track_idx is a list of arrays with varying lengths — keep as list
    most_five_consistently_misclassified_idx = np.bincount(np.concatenate(misclassification_track_idx)).argsort()[-5:]
    weight_track_for_most_five_consistently_misclassified_idx = [weight_track[i][most_five_consistently_misclassified_idx] for i in range(len(weight_track))]

    # shape: (50 iterations, 5 samples) → transpose to (5 samples, 50 iterations) for plotting
    weight_track_for_most_five_consistently_misclassified_idx = np.array(weight_track_for_most_five_consistently_misclassified_idx)

    plt.figure(figsize=(12, 6))
    for i in range(5):
        plt.plot(weight_track_for_most_five_consistently_misclassified_idx[:, i], label=f'Sample Index {most_five_consistently_misclassified_idx[i]}')
    plt.xlabel('Iteration')
    plt.ylabel('Weight')
    plt.title('Weight Evolution of the 5 Most Consistently Misclassified Samples')
    plt.legend()
    plt.grid()
    plt.show()

    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    