'''
Name : Azubuine Samuel Tochukwu
PSU ID: 960826967
EE552 Project 2
}
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import pearsonr
from scipy import linalg
from typing import Dict, List, Callable, Tuple, Optional, Union


def sequential_forward_selection(X: np.ndarray, y: np.ndarray,
                                 n_features: int = 10,
                                 cv_splits: int = 5,
                                 n_neighbors: int = 5) -> Tuple[np.ndarray, List[int]]:
    """
    Perform Sequential Forward Selection using KNN classifier

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target labels
        n_features: Number of features to select
        cv_splits: Number of cross-validation splits
        n_neighbors: Number of neighbors for KNN

    Returns:
        Selected features matrix and indices of selected features
    """
    n_total_features = X.shape[1]
    selected_features = []
    remaining_features = list(range(n_total_features))

    for _ in range(min(n_features, n_total_features)):
        best_score = -np.inf
        best_feature = None

        # Try each remaining feature
        for feature in remaining_features:
            current_features = selected_features + [feature]
            scores = []

            # Cross-validation
            for split in range(cv_splits):
                # Random split indices
                indices = np.random.permutation(len(y))
                split_point = len(y) // cv_splits
                test_idx = indices[split * split_point:(split + 1) * split_point]
                train_idx = np.setdiff1d(indices, test_idx)

                # Train classifier
                X_train = X[train_idx][:, current_features]
                X_test = X[test_idx][:, current_features]
                clf = KNeighborsClassifier(n_neighbors=n_neighbors)
                clf.fit(X_train, y[train_idx])
                score = clf.score(X_test, y[test_idx])
                scores.append(score)

            # Average score for this feature set
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_feature = feature

        if best_feature is not None:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)

    return X[:, selected_features], selected_features



def plot_feature_importance(feature_scores: np.ndarray, selected_indices: List[int],
                            dataset_type: int, subject_index: int):
    """
    Plot feature importance scores

    Args:
        feature_scores: Array of feature importance scores
        selected_indices: Indices of selected features
        dataset_type: Type of dataset
        subject_index: Current subject index
    """
    plt.figure(figsize=(10, 6))

    # Plot all feature scores
    plt.bar(range(len(feature_scores)), feature_scores, alpha=0.3, color='gray', label='All Features')

    # Highlight selected features
    plt.bar(selected_indices, feature_scores[selected_indices], color='blue', label='Selected Features')

    plt.title(f'Feature Importance Scores (Dataset {dataset_type}, Subject {subject_index})')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/feature_importance_d{dataset_type}_s{subject_index}.png')
    plt.close()


def plot_training_progress(accuracies: List[float], dataset_type: int):
    """
    Plot accuracy progression across subjects for a dataset, eg subject i=2 of dataset 100

    Args:
        accuracies: List of accuracy scores
        dataset_type: Type of dataset
    """
    plt.figure(figsize=(10, 6))

    # Plot accuracy progression
    plt.plot(range(len(accuracies)), accuracies, marker='o', linestyle='-', linewidth=2)

    # Adding a trend line
    z = np.polyfit(range(len(accuracies)), accuracies, 1)
    p = np.poly1d(z)
    plt.plot(range(len(accuracies)), p(range(len(accuracies))), "r--", alpha=0.8, label='Trend')

    plt.title(f'Training Progress (Dataset {dataset_type})')
    plt.xlabel('Subject Index')
    plt.ylabel('Test Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'results/training_progress_d{dataset_type}.png')
    plt.close()


def plot_dataset_comparison(results: Dict):
    """
    Plot accuracy comparison across different datasets

    Args:
        results: Dictionary containing results for each dataset
    """
    plt.figure(figsize=(12, 6))

    # Prepare data for box plot
    data = []
    labels = []

    for dataset_type in results:
        accuracies = [r['test_acc'] for r in results[dataset_type]['subject_results']]
        data.append(accuracies)
        labels.append(f'Dataset {dataset_type}')

    # Create box plot
    plt.boxplot(data, labels=labels)

    # Add individual points
    for i, d in enumerate(data, 1):
        plt.scatter([i] * len(d), d, alpha=0.5)

    plt.title('Accuracy Distribution Across Datasets')
    plt.ylabel('Test Accuracy')
    plt.grid(True, alpha=0.3)

    # Save plot
    plt.tight_layout()
    plt.savefig('results/dataset_comparison.png')
    plt.close()


def plot_feature_heatmap(feature_importance_matrix: np.ndarray, dataset_type: int):
    """
    Plot heatmap of feature importance across subjects

    Args:
        feature_importance_matrix: Matrix of feature importance scores (subjects × features)
        dataset_type: Type of dataset
    """
    plt.figure(figsize=(12, 8))

    # Create heatmap
    sns.heatmap(feature_importance_matrix, cmap='YlOrRd',
                xticklabels='auto', yticklabels='auto')

    plt.title(f'Feature Importance Across Subjects (Dataset {dataset_type})')
    plt.xlabel('Feature Index')
    plt.ylabel('Subject Index')

    # Save plot
    plt.tight_layout()
    plt.savefig(f'results/feature_importance_heatmap_d{dataset_type}.png')
    plt.close()




def calculate_feature_criterion(X: np.ndarray, y: np.ndarray, method: str = 'mrmr') -> np.ndarray:
    """
    Calculate feature selection criteria

    Args:
        X: Feature matrix (samples, features)
        y: Labels (samples,)
        method: Feature selection method ('vr', 'mrmr', 'avr')

    Returns:
        Calculated criterion scores (features,)
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # shape as (samples, features)

    # Feature selection methods
    def calculate_vr(_X: np.ndarray, _y: np.ndarray) -> np.ndarray:
        """Calculates Variance Ratio criterion"""
        classes = np.unique(_y)
        n_classes = len(classes)
        n_features = _X.shape[1]

        overall_mean = np.mean(_X, axis=0)
        between_var = np.zeros(n_features)
        within_var = np.zeros(n_features)

        for c in classes:
            class_mask = (_y == c)
            class_X = _X[class_mask]
            class_mean = np.mean(class_X, axis=0)

            between_var += np.sum(class_mask) * (class_mean - overall_mean) ** 2
            within_var += np.sum((class_X - class_mean) ** 2, axis=0)

        between_var /= (n_classes - 1)
        within_var /= (_X.shape[0] - n_classes)
        within_var = np.where(within_var == 0, 1e-10, within_var)

        return between_var / within_var

    def calculate_mrmr(_X: np.ndarray, _y: np.ndarray) -> np.ndarray:
        """Calculate mRMR criterion"""
        n_features = _X.shape[1]
        relevance = np.array([mutual_information(_X[:, i], _y) for i in range(n_features)])

        redundancy = np.zeros(n_features)
        for i in range(n_features):
            correlations = []
            for j in range(n_features):
                if i != j:
                    corr, _ = pearsonr(_X[:, i], _X[:, j])
                    correlations.append(abs(corr))
            redundancy[i] = np.mean(correlations) if correlations else 0

        return relevance - redundancy

    def calculate_avr(_X: np.ndarray, _y: np.ndarray) -> np.ndarray:
        """Calculates Augmented Variance Ratio criterion"""
        vr = calculate_vr(_X, _y)
        between_var = np.var(_X, axis=0)
        return vr * (1 + np.log(between_var + 1e-10))

    # Select method
    methods = {
        'vr': calculate_vr,
        'mrmr': calculate_mrmr,
        'avr': calculate_avr
    }

    if method not in methods:
        raise ValueError(f"Invalid method. Choose from {list(methods.keys())}")

    return methods[method](X_scaled, y)


def feature_selection(feats: np.ndarray, labels: np.ndarray = None,
                      method: str = 'mrmr', n_features: int = 30) -> np.ndarray:
    """
    Perform feature selection using specified method

    Args:
        feats: Feature matrix (n_features x n_samples)
        labels: Labels array
        method: Selection method
        n_features: Number of features to select

    Returns:
        Selected features matrix
    """
    if feats.shape[0] == 0:
        raise ValueError("Feature matrix is empty. Check data preprocessing.")

    # Limit number of features
    n_features = min(n_features, feats.shape[0])

    # Calculate criterion scores
    scores = calculate_feature_criterion(feats, labels, method)

    # Select top features
    selected_indices = np.argsort(scores)[-n_features:]

    return feats[selected_indices]


def mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """Calculates mutual information between feature and labels"""
    x_bins = np.histogram_bin_edges(x, bins='auto')
    x_discrete = np.digitize(x, x_bins)

    mutual_info = 0
    for x_val in np.unique(x_discrete):
        for y_val in np.unique(y):
            p_xy = np.mean((x_discrete == x_val) & (y == y_val))
            p_x = np.mean(x_discrete == x_val)
            p_y = np.mean(y == y_val)
            if p_xy > 0:
                mutual_info += p_xy * np.log2(p_xy / (p_x * p_y))

    return mutual_info


def calculate_vr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculates Variance Ratio criterion"""
    classes = np.unique(y)
    n_classes = len(classes)
    n_features = X.shape[1]

    overall_mean = np.mean(X, axis=0)
    between_var = np.zeros(n_features)
    within_var = np.zeros(n_features)

    for c in classes:
        class_mask = (y == c)
        class_X = X[class_mask]
        class_mean = np.mean(class_X, axis=0)

        between_var += np.sum(class_mask) * (class_mean - overall_mean) ** 2
        within_var += np.sum((class_X - class_mean) ** 2, axis=0)

    between_var /= (n_classes - 1)
    within_var /= (X.shape[0] - n_classes)
    within_var = np.where(within_var == 0, 1e-10, within_var)

    return between_var / within_var


def calculate_mrmr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculate mRMR criterion"""
    n_features = X.shape[1]
    relevance = np.array([mutual_information(X[:, i], y) for i in range(n_features)])

    redundancy = np.zeros(n_features)
    for i in range(n_features):
        correlations = []
        for j in range(n_features):
            if i != j:
                corr, _ = pearsonr(X[:, i], X[:, j])
                correlations.append(abs(corr))
        redundancy[i] = np.mean(correlations) if correlations else 0

    return relevance - redundancy


def calculate_avr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculates Augmented Variance Ratio criterion"""
    vr = calculate_vr(X, y)
    between_var = np.var(X, axis=0)
    return vr * (1 + np.log(between_var + 1e-10))