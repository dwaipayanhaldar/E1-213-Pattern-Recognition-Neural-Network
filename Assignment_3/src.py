import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms, models
from torch import nn
import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
torch.manual_seed(25)
torch.mps.manual_seed(25)
from torchvision.io import read_image
device = "mps" if torch.mps.is_available() else "cpu"

#################################################
# Basic Utilities #
#################################################

def train_loop(train_loader, val_loader, model, loss_fn, optimizer, max_epoch):

    for iter in np.arange(max_epoch):
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


class TimeSeriesDataset(Dataset):
    """
    The Dataset definition for time series data specifically delhi_aqi.csv
    """
    
    def __init__(self, file_path, seq_len, hop, col_name='pm2_5', all_features=False, train_frac=0.7):
        df = pd.read_csv(file_path)

        if all_features:
            feature_cols = [c for c in df.columns if c != 'date']
            data = torch.tensor(df[feature_cols].values, dtype=torch.float32)  # (T, F)
            pm2_5_idx = feature_cols.index(col_name)
            self.input_dim = seq_len * len(feature_cols)
            self.num_features = len(feature_cols)
        else:
            data = torch.tensor(df[col_name].values, dtype=torch.float32)      # (T,)
            pm2_5_idx = None
            self.input_dim = seq_len
            self.num_features = 1

        # Normalise using training portion stats only — prevents leakage into val/test
        n_train = int(train_frac * len(data))
        self.X_mean = data[:n_train].mean(dim=0)
        self.X_std  = data[:n_train].std(dim=0).clamp(min=1e-8)
        data = (data - self.X_mean) / self.X_std

        # PM2.5 stats for denormalising predictions
        self.y_mean = self.X_mean[pm2_5_idx] if all_features else self.X_mean
        self.y_std  = self.X_std[pm2_5_idx]  if all_features else self.X_std

        self.X = []
        self.y = []

        max_start = len(data) - seq_len - hop + 1
        for i in range(max_start):
            x_seq = data[i:i+seq_len]
            target_idx = i + seq_len - 1 + hop
            y_target = data[target_idx, pm2_5_idx] if all_features else data[target_idx]

            # Each sample is a single feature vector formed by flattening the full window.
            self.X.append(x_seq.reshape(-1))
            self.y.append(y_target)

        self.X = torch.stack(self.X)
        self.y = torch.stack(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AQIDataset(Dataset):
    """Loads the DelhiAQI Dataset"""

    def __init__(self, features, labels=None, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        X = self.features[idx]
        if self.transform:
            X = self.transform(X)
        if self.labels is not None:
            y = self.labels[idx]
            return X, y
        else:
            return (X,)
        

class PlantVillageDataset(Dataset):
    """
    Loads PlantVillage images and gives class labels.
    """

    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.y = []

        class_names = sorted([d for d in os.listdir(root_dir)
                               if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        for class_name in class_names:
            class_dir = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(class_dir, fname))
                    self.y.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_tensor = read_image(self.image_paths[idx])[:3].float() / 255.0
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return self.transform(img_tensor) if self.transform else img_tensor, label


class UnlabeledDataset(Dataset):
    """Wraps any (X, y) dataset and drops the label — for unsupervised training."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        X = data[0] if isinstance(data, (tuple, list)) else data
        return (X,)

class AptosDatasetClass4(Dataset):
    """
    Loads Aptos images and gives class labels.
    """

    def __init__(self, csv_file, root_dir, transform=None):
        self.transform = transform
        self.image_paths = []

        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            img_path = os.path.join(root_dir, row['id_code'] + '.png')
            if os.path.exists(img_path):
                if row['diagnosis'] == 4:
                    self.image_paths.append(img_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_tensor = read_image(self.image_paths[idx])[:3].float() / 255.0
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)
        return self.transform(img_tensor) if self.transform else img_tensor


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

class GradientBoostedRegressor():

    def __init__(self, n_trees=50, max_depth=2, learning_rate=0.1):
        self.ntrees = n_trees
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.trees = []
        self.F0 = None

    def mse_split(self, y):
        """Variance of y — used as the regression split criterion."""
        if len(y) == 0:
            return 0
        return np.var(y) * len(y)

    def best_split_for_single_feature(self, X_col, y, number_of_thresholds=500):
        """Best split minimising weighted MSE (variance reduction) for regression."""
        best_score = float('inf')
        best_threshold = None

        thresholds = np.linspace(np.min(X_col), np.max(X_col), num=number_of_thresholds)
        for threshold in thresholds:
            left_idx  = np.where(X_col <  threshold)[0]
            right_idx = np.where(X_col >= threshold)[0]

            score = self.mse_split(y[left_idx]) + self.mse_split(y[right_idx])
            if score < best_score:
                best_score = score
                best_threshold = threshold

        parent_score = self.mse_split(y)
        return best_threshold, (parent_score - best_score)  # variance reduction (gain)

    def best_split(self, X, y, available_features, number_of_thresholds=500):
        """Find the best feature and threshold among available_features only."""

        best_gain = float('-inf')
        best_feature_idx = None
        best_threshold = None

        for feature_idx in available_features:
            threshold, gain = self.best_split_for_single_feature(X[:, feature_idx], y, number_of_thresholds)
            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold
                best_feature_idx = feature_idx

        return best_threshold, best_gain, best_feature_idx

    def build_tree(self, X, y, depth=0, available_features=None):
        """Regression tree: leaf value is the mean of residuals in that node."""

        if available_features is None:
            available_features = list(range(X.shape[1]))

        leaf_value = float(np.mean(y))  # regression leaf = mean of residuals

        if depth >= self.max_depth or len(y) == 0 or len(available_features) == 0:
            return {'leaf': True, 'value': leaf_value}

        threshold, gain, feature_idx = self.best_split(X, y, available_features)

        if gain <= 0 or feature_idx is None:
            return {'leaf': True, 'value': leaf_value}

        left_idx  = np.where(X[:, feature_idx] <  threshold)[0]
        right_idx = np.where(X[:, feature_idx] >= threshold)[0]

        if len(left_idx) == 0 or len(right_idx) == 0:
            return {'leaf': True, 'value': leaf_value}

        available_features.remove(feature_idx)

        return {
            'leaf'        : False,
            'feature_idx' : feature_idx,
            'threshold'   : threshold,
            'left'        : self.build_tree(X[left_idx],  y[left_idx],  depth + 1, available_features),
            'right'       : self.build_tree(X[right_idx], y[right_idx], depth + 1, available_features),
        }

    def predict_single(self, node, x):
        if node['leaf']:
            return node['value']
        if x[node['feature_idx']] < node['threshold']:
            return self.predict_single(node['left'], x)
        else:
            return self.predict_single(node['right'], x)

    def predict(self, tree, X):
        return np.array([self.predict_single(tree, x) for x in X])

    def predict_gradient_boosting(self, X):
        """Cumulative ensemble prediction: F_0 + lr * sum(h_m(X))"""
        pred = np.full(len(X), self.F0)
        for tree in self.trees:
            pred += self.learning_rate * self.predict(tree, X)
        return pred

    def fit(self, X, y_true):
        self.F0 = float(np.mean(y_true))  
        F = np.full(len(y_true), self.F0) 

        self.residual_variances = {}

        for m in range(self.ntrees):
            residual = y_true - F  

            if (m + 1) in [1, 5, 10, 50]:
                self.residual_variances[m + 1] = float(np.var(residual))

            tree = self.build_tree(X, residual)
            self.trees.append(tree)

            F += self.learning_rate * self.predict(tree, X) 

        return self.residual_variances


#################################################
# Phase 2 #
#################################################

def unsupervised_train_loop(train_loader, model, loss_fn, optimizer, max_epoch):

    model = model.to(device)
    for iter in np.arange(max_epoch):
        model.train()
        train_loss = 0
        for batch, (X,) in enumerate(train_loader):
            X = X.to(device)
            pred = model(X)
            loss = loss_fn(pred, X)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        if (iter+1) % 10 == 0:
            print(f"Epoch {iter+1}/{max_epoch}, Train Loss: {train_loss:.4f}")

class PCA():

    def __init__(self, explained_variance_threshold= None, n_components = None):
        self.explained_variance_threshold = explained_variance_threshold
        self.n_components = n_components
        self.mean = None
    
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        eigenvectors = Vt.T 
        eigenvalues  = S**2

        sorted_idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[sorted_idx]  
        self.eigenvectors = eigenvectors[:, sorted_idx]

        if self.explained_variance_threshold is not None:
            total_variance = np.sum(self.eigenvalues)
            variance_ratio = np.cumsum(self.eigenvalues) / total_variance
            self.n_components = np.searchsorted(variance_ratio, self.explained_variance_threshold) + 1

    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.eigenvectors[:, :self.n_components]
    
    def inverse_transform(self, X_reduced):
        return X_reduced @ self.eigenvectors[:, :self.n_components].T + self.mean
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class LinearAutoEncoder(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super(LinearAutoEncoder, self).__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

class gaussian_mixture():

    def __init__(self, data, K):
        self.X = data
        self.K = K
        self.alpha = None
        self.mu    = None
        self.sigma = None
        self.R     = None

    def fit(self, K, iter):
        """Updates the parameters. Runs the whole EM algorithm"""
        #initialization
        K = self.K
        N = self.X.shape[0]
        alpha = (1/K)*np.ones(K)
        idx = np.random.randint(0,N, size=(K))
        mu = self.X[idx]
        data_var = np.mean(np.var(self.X, axis=0))
        f = np.random.rand(K,)*data_var
        sigma = np.array([s*np.eye(self.X.shape[1]) for s in f])

        for i in range(iter):
            log_R = np.column_stack([
                np.log(alpha[k] + 1e-300) + sp.stats.multivariate_normal.logpdf(self.X, mean=mu[k], cov=sigma[k])
                for k in range(K)
            ])
            log_R -= log_R.max(axis=1, keepdims=True)  
            R = np.exp(log_R)
            R /= R.sum(axis=1, keepdims=True)

            # Update Alpha
            alpha_modified = R.sum(axis=0) / N  

            # Update mu
            mu_modified = np.zeros_like(mu)
            for k in range(K):
                mu_modified[k] = R[:, k] @ self.X / R[:, k].sum()

            #Update sigma
            sigma_modified = np.zeros_like(sigma)
            for k in range(K):
                diff = self.X - mu[k]
                sigma_modified[k] = (R[:, k].reshape(-1,1) * diff).T @ diff / R[:, k].sum()
                sigma_modified[k] += 1e-4 * np.eye(self.X.shape[1]) 

            alpha = alpha_modified
            mu = mu_modified
            sigma = sigma_modified

        self.alpha = alpha
        self.mu    = mu
        self.sigma = sigma

    def transform(self, X):
        log_R = np.column_stack([
            np.log(self.alpha[k] + 1e-300) + sp.stats.multivariate_normal.logpdf(X, mean=self.mu[k], cov=self.sigma[k])
            for k in range(self.K)
        ])
        log_R -= log_R.max(axis=1, keepdims=True)
        R = np.exp(log_R)
        R /= R.sum(axis=1, keepdims=True)
        return np.argmax(R, axis=1)

class ownCNN(nn.Module):

    def __init__(self, input_dim, output_classes):
        super().__init__()
        sp = input_dim // (2**3)  
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(3, 8, kernel_size=3, padding=1),  nn.BatchNorm2d(8) ,nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16) ,nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32) ,nn.ReLU(), nn.MaxPool2d(2),
        ])
        self.fc = nn.ModuleList([
            nn.Flatten(),
            nn.Linear(32 * sp * sp, 256),nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, output_classes),
        ])


    def forward(self, x, print_shapes=False):
        if print_shapes:
            print(f"Input:          {tuple(x.shape)}")
        for layer in self.conv_layers:
            x = layer(x)
            if print_shapes and not isinstance(layer, nn.ReLU):
                print(f"{layer.__class__.__name__:12s}: {tuple(x.shape)}")
        for layer in self.fc:
            x = layer(x)
            if print_shapes and not isinstance(layer, nn.ReLU):
                print(f"{layer.__class__.__name__:12s}: {tuple(x.shape)}")
        if print_shapes:
            print(f"Output:         {tuple(x.shape)}")
        return x.squeeze(1) 


#################################################
# Phase 3 #
#################################################

def VAE_training_loop(train_loader, model, optimizer, max_epoch, kl_warmup_epochs=5, free_bits=0.5):

    recon_loss_track, kl_loss_track = [], []
    model = model.to(device)
    for iter in np.arange(max_epoch):
        # KL annealing: linearly ramp beta from 0 → 1 over kl_warmup_epochs
        beta = min(1.0, (iter + 1) / kl_warmup_epochs)

        model.train()
        total_recon, total_kl, total_loss = 0, 0, 0
        for batch, (X,) in enumerate(train_loader):
            X = X.to(device)
            pred, mu, log_var = model(X)
            recon_loss = nn.functional.mse_loss(pred, X)

            # Free bits: KL per dimension, clamped to minimum free_bits nats
            # Prevents any latent dimension from fully collapsing
            kl_per_dim = -0.5 * (1 + log_var - mu**2 - torch.exp(log_var))  # (batch, latent_dim)
            kl_loss = torch.mean(torch.clamp(kl_per_dim, min=free_bits).sum(dim=1))

            loss = recon_loss + beta * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            total_loss += loss.item()

        total_loss /= len(train_loader) 
        total_recon /= len(train_loader)
        total_kl /= len(train_loader)

        recon_loss_track.append(total_recon)
        kl_loss_track.append(total_kl)

        print(f"Epoch {iter+1}/{max_epoch}, β={beta:.2f}, Loss: {total_loss:.4f}, Recon Loss: {total_recon:.4f}, KL Loss: {total_kl:.4f}")
    
    return np.array(recon_loss_track), np.array(kl_loss_track)
   

class VAE(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim          
        sp = input_dim // (2**3)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * sp * sp, 256), nn.ReLU(),  
            nn.Linear(256, latent_dim * 2)             
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 128 * sp * sp), nn.ReLU(),
            nn.Unflatten(1, (128, sp, sp)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = h[:, :self.latent_dim], h[:, self.latent_dim:]
        log_var = torch.clamp(log_var, -10, 10) 
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        x_recon = self.decoder(z)
        return x_recon, mu, log_var

    def sample(self, n_samples):
        z = torch.randn(n_samples, self.latent_dim).to(device)
        return self.decoder(z)


class GAN(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(GAN, self).__init__()
        self.latent_dim = latent_dim
        sp = input_dim // (2**3)

        # Generator: noise z → fake image
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 128 * sp * sp), nn.ReLU(),
            nn.Unflatten(1, (128, sp, sp)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3,  kernel_size=3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )
        
        self.discriminator = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.LeakyReLU(0.2),          
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * sp * sp, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1)   
        )

    def forward(self, z):
        return self.generator(z)

    def generate(self, n_samples):
        z = torch.randn(n_samples, self.latent_dim).to(device)
        with torch.no_grad():
            return self.generator(z)


def GAN_training_loop(train_loader, model, optimizer_G, optimizer_D, max_epoch):

    model = model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()  
    g_loss_track, d_loss_track = [], []

    for epoch in np.arange(max_epoch):
        model.train()
        total_g, total_d = 0, 0

        for batch, (X,) in enumerate(train_loader):
            X = X.to(device)
            B = X.shape[0]
            real_labels = torch.full((B, 1), 0.9).to(device)  # label smoothing: 0.9 not 1.0
            fake_labels = torch.zeros(B, 1).to(device)

            # Train Discriminator
            optimizer_D.zero_grad()
            d_real = model.discriminator(X)
            d_loss_real = loss_fn(d_real, real_labels)

            z = torch.randn(B, model.latent_dim).to(device)
            fake = model.generator(z).detach()
            d_fake = model.discriminator(fake)
            d_loss_fake = loss_fn(d_fake, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # Train Generator — use hard 1.0 labels, only smooth discriminator side
            optimizer_G.zero_grad()
            z = torch.randn(B, model.latent_dim).to(device)
            fake = model.generator(z)
            g_loss = loss_fn(model.discriminator(fake), torch.ones(B, 1).to(device))
            g_loss.backward()
            optimizer_G.step()

            total_d += d_loss.item()
            total_g += g_loss.item()

        total_d /= len(train_loader)
        total_g /= len(train_loader)
        g_loss_track.append(total_g)
        d_loss_track.append(total_d)

        print(f"Epoch {epoch+1}/{max_epoch}, G Loss: {total_g:.4f}, D Loss: {total_d:.4f}")

    return np.array(g_loss_track), np.array(d_loss_track)

def fid(feature1, feature2):
    mu1 = np.mean(feature1, axis=0)
    mu2 = np.mean(feature2, axis=0)
    sigma1 = np.cov(feature1, rowvar=False)
    sigma2 = np.cov(feature2, rowvar=False)
    diff = mu1 - mu2

    covmean = sp.linalg.sqrtm(sigma1 @ sigma2)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return diff.T @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


#################################################
# Phase 4 #
#################################################

class InfoNCEloss(nn.Module):

    def __init__(self, temperature=0.5):
        super(InfoNCEloss, self).__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        B = z1.shape[0]

        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        
        z = torch.cat([z1, z2], dim=0)
        sim = z @ z.T / self.temperature                      

        pos_sim = torch.cat([torch.diag(sim, B), torch.diag(sim, -B)]) 

        mask = torch.eye(2 * B, dtype=torch.bool, device=z1.device)
        sim.masked_fill_(mask, float('-inf'))

        loss = -pos_sim + torch.logsumexp(sim, dim=1)
        return loss.mean()


class SimCLR(nn.Module):

    def __init__(self, latent_dim, transform1, transform2):
        super(SimCLR, self).__init__()
        self.latent_dim = latent_dim
        self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet18.fc = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        self.transform1 = transform1
        self.transform2 = transform2

    def forward(self, x):
        x1 = self.transform1(x)
        x2 = self.transform2(x)
        z1 = self.resnet18(x1)
        z2 = self.resnet18(x2)
        return z1, z2


def SimCLRtraining_loop(train_loader, val_loader, model, optimizer, max_epoch):

    model = model.to(device)
    loss_fn = InfoNCEloss(temperature=0.5)  
    train_loss_track, val_loss_track = [], []

    for iter in np.arange(max_epoch):
        model.train()
        train_loss = 0
        for batch, (X,) in enumerate(train_loader):
            X = X.to(device)
            z1, z2 = model(X)
            loss = loss_fn(z1, z2)  

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_loss_track.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch, (X,) in enumerate(val_loader):
                X = X.to(device)
                z1, z2 = model(X)
                val_loss += loss_fn(z1, z2).item()

        val_loss /= len(val_loader)
        val_loss_track.append(val_loss)
        print(f"Epoch {iter+1}/{max_epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return np.array(train_loss_track), np.array(val_loss_track)


class LinearProbe(nn.Module):
    """Frozen SimCLR encoder + single trainable linear classification head."""

    def __init__(self, encoder, latent_dim, n_classes):
        super().__init__()
        self.encoder = encoder   
        self.head = nn.Linear(latent_dim, n_classes)

    def forward(self, x):
        with torch.no_grad():
            z = self.encoder(x)  
        return self.head(z)


#################################################
# Phase 5 #
#################################################

def get_reward(action, next_pm25):
    if action == 1:
        return -0.1  
    else:
        if next_pm25 > 150:
            return -10.0
        return 0.0
    
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.fc(x)
    
def collect_trajectory(policy_net, states, pm25_targets):
    log_probs = []
    rewards = []
    
    for t in range(len(states)):
        state = torch.as_tensor(states[t], dtype=torch.float32)
        prob_on = policy_net(state)
        
        action = 1 if np.random.rand() < prob_on.item() else 0

        if action == 1:
            log_prob = torch.log(prob_on)
        else:
            log_prob = torch.log(1 - prob_on)
            
        # Each state window is labeled with the PM2.5 value from its own final hour.
        reward = get_reward(action, pm25_targets[t])
        
        log_probs.append(log_prob)
        rewards.append(reward)
        
    return log_probs, rewards

def update_policy(optimizer, log_probs, rewards):

    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + 0.99 * G 
        returns.insert(0, G)
    
    returns = torch.FloatTensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    policy_loss = []
    for log_prob, Gt in zip(log_probs, returns):
        policy_loss.append(-log_prob * Gt)
    
    optimizer.zero_grad()
    sum(policy_loss).backward()
    optimizer.step()





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
    
    
