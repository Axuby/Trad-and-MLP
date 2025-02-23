'''
Name : Azubuine Samuel Tochukwu
PSU ID: 960826967
EE552 Project 2
}
'''

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from model_starter import TraditionalClassifier
from model_starter import  MLP
from torch.optim.lr_scheduler import StepLR
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from scipy import linalg
from typing import Callable, Dict, List, Tuple, Optional
from scipy import linalg
from typing import Callable, Dict, List, Tuple, Optional
from utils import plot_dataset_comparison, plot_feature_heatmap, plot_training_progress, plot_feature_importance, \
    calculate_feature_criterion, sequential_forward_selection, feature_selection




def fisher_projection(X: np.ndarray, y: np.ndarray, n_components: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implements Fisher Linear Discriminant Analysis for dimensionality reduction

    Args:
        X: Input features of shape (n_samples, n_features)
        y: Labels of shape (n_samples,)
        n_components: Number of components to keep (default: min(n_classes-1, n_features))

    Returns:
        projection_matrix: Matrix to project the data
        explained_variance_ratio: Ratio of variance explained by each component
    """
    classes = np.unique(y)
    n_classes = len(classes)
    n_features = X.shape[1]

    if n_components is None:
        n_components = min(n_classes - 1, n_features)

    # Calculate mean vectors
    overall_mean = np.mean(X, axis=0)
    class_means = {c: np.mean(X[y == c], axis=0) for c in classes}

    # Calculate scatter matrices
    Sw = np.zeros((n_features, n_features))  # Within-class scatter
    Sb = np.zeros((n_features, n_features))  # Between-class scatter

    for c in classes:
        # Get samples for this class
        X_c = X[y == c]
        n_c = len(X_c)

        # Calculate centered class samples
        centered_X = X_c - class_means[c]

        # Update within-class scatter
        Sw += centered_X.T @ centered_X

        # Update between-class scatter
        diff = (class_means[c] - overall_mean).reshape(-1, 1)
        Sb += n_c * (diff @ diff.T)

    # Add small regularization to avoid singularity
    Sw_reg = Sw + 1e-4 * np.eye(n_features)

    try:
        # Solve generalized eigenvalue problem
        eigenvalues, eigenvectors = linalg.eigh(Sb, Sw_reg)

        # Sort eigenvectors by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Calculate explained variance ratio
        explained_variance_ratio = eigenvalues / np.sum(eigenvalues)

        # Select top components
        projection_matrix = eigenvectors[:, :n_components]
        explained_variance_ratio = explained_variance_ratio[:n_components]

        return projection_matrix, explained_variance_ratio

    except np.linalg.LinAlgError as e:
        print(f"Error in Fisher LDA: {e}")
        return None, None


def plot_confusion_matrix(train_labels, pred_train_labels, test_labels, pred_test_labels, i, dataset):
    """
    Plot confusion matrices for training and testing data with enhanced error checking
    """
    # Input validation
    if len(test_labels) == 0 or len(pred_test_labels) == 0:
        print(f"Warning: Empty test labels for dataset {dataset}, subject {i}")
        return

    if len(train_labels) == 0 or len(pred_train_labels) == 0:
        print(f"Warning: Empty training labels for dataset {dataset}, subject {i}")
        return

    # Print debugging information
    print(f"\nDebugging information for dataset {dataset}, subject {i}:")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Pred train labels shape: {pred_train_labels.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    print(f"Pred test labels shape: {pred_test_labels.shape}")
    print(f"Unique train labels: {np.unique(train_labels)}")
    print(f"Unique test labels: {np.unique(test_labels)}")
    print(f"Unique pred test labels: {np.unique(pred_test_labels)}")

    try:
        plt.figure(figsize=(12, 5))

        # Training Confusion Matrix
        plt.subplot(1, 2, 1)
        train_cm = confusion_matrix(train_labels, pred_train_labels)
        train_display = ConfusionMatrixDisplay(
            confusion_matrix=train_cm,
            display_labels=np.unique(np.concatenate([train_labels, pred_train_labels]))
        )
        train_display.plot(cmap='Blues', values_format='d', ax=plt.gca())
        plt.title(f"Training Confusion Matrix of {dataset} of {i}")

        # Testing Confusion Matrix
        plt.subplot(1, 2, 2)
        test_cm = confusion_matrix(test_labels, pred_test_labels)
        test_display = ConfusionMatrixDisplay(
            confusion_matrix=test_cm,
            display_labels=np.unique(np.concatenate([test_labels, pred_test_labels]))
        )
        test_display.plot(cmap='Blues', values_format='d', ax=plt.gca())
        plt.title('Testing Confusion Matrix')

        # Save and close
        os.makedirs('results', exist_ok=True)
        plt.tight_layout()
        plt.savefig(f'results/confusion_matrices{dataset}_{i}.png')
        plt.close()

    except Exception as e:
        print(f"Error plotting confusion matrix: {str(e)}")
        print(f"Train CM shape: {train_cm.shape if 'train_cm' in locals() else 'Not created'}")
        print(f"Test CM shape: {test_cm.shape if 'test_cm' in locals() else 'Not created'}")
        plt.close()


def load_dataset(verbose: bool = True,
                 subject_index: int = 9,
                 dataset_type: int = 100,
                 use_feature_selection: bool = True,
                 selection_method: str = 'mrmr',
                 n_features: int = 10,
                 use_sfs: bool = False,
                 n_neighbors: int = 5) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Enhanced dataset loading with KNN-based feature selection"""
    try:
        filename = f"Taiji_dataset_{dataset_type}.csv"
        dataset = np.loadtxt(filename, delimiter=",", dtype=float, skiprows=1, usecols=range(0, 70))

        person_idxs = dataset[:, -1]
        labels = dataset[:, -2]
        feats = dataset[:, :-2].T

        if use_feature_selection:
            if use_sfs:
                feats, selected_indices = sequential_forward_selection(
                    feats.T, labels,
                    n_features=n_features,
                    n_neighbors=n_neighbors
                )
                feats = feats.T
            else:
                # Use criterion-based selection
                scores = calculate_feature_criterion(feats.T, labels, method=selection_method)
                selected_indices = np.argsort(scores)[-n_features:]
                feats = feats[selected_indices]
        else:
            # Basic variance filtering
            variance_threshold = 1e-6
            feature_mask = np.var(feats, axis=1) > variance_threshold
            feats = feats[feature_mask]

        # Train/test split
        train_mask = person_idxs != subject_index
        train_feats = feats[:, train_mask].T
        train_labels = labels[train_mask].astype(int)
        test_feats = feats[:, ~train_mask].T
        test_labels = labels[~train_mask].astype(int)

        if verbose:
            print(f'Dataset {dataset_type} Loaded')
            print(f'\t# of Classes: {len(np.unique(train_labels))}')
            print(f'\t# of Features: {train_feats.shape[1]}')
            print(f'\t# of Training Samples: {train_feats.shape[0]}')
            print(f'\t# of Testing Samples: {test_feats.shape[0]}')

        return train_feats, train_labels, test_feats, test_labels

    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None


def classification(dataset_types: List[int] = [100, 200, 300],
                   use_lda: bool = True,
                   n_features: int = 10,
                   n_neighbors: int = 5) -> Dict:
    """Enhanced classification with Fisher projection and visualization"""
    results = {}

    for dataset_type in dataset_types:
        print(f"\nProcessing Dataset {dataset_type}")
        subject_results = []
        feature_importance_scores = []
        accuracy_progression = []

        for subject_index in range(10):
            if subject_index == 0:
                continue
            print(f"\nProcessing subject {subject_index}")
            try:
                # Loading the dataset for specified dataset_type and subject_index
                data = load_dataset(
                    verbose=True,
                    subject_index=subject_index,
                    dataset_type=dataset_type,
                    use_feature_selection=False
                )

                if data is None:
                    continue

                train_feats, train_labels, test_feats, test_labels = data

                # For the Feature selection pipeline
                # 1. Calculating feature importance using mRMR
                scores = calculate_feature_criterion(train_feats, train_labels, method='mrmr')
                feature_importance_scores.append(scores)

                print("Hurray !!!!!!!!   got here ----------------------")


                print("Plotting feature importance scores...")
                plot_feature_importance(scores, range(n_features), dataset_type, subject_index)

                # 2. Select top K features
                top_k = min(20, train_feats.shape[1])
                top_indices = np.argsort(scores)[-top_k:]
                train_feats_filtered = train_feats[:, top_indices]
                test_feats_filtered = test_feats[:, top_indices]

                # 3. Apply Sequential Forward Selection
                train_feats_selected, selected_indices = sequential_forward_selection(
                    train_feats_filtered, train_labels,
                    n_features=n_features,
                    n_neighbors=n_neighbors
                )
                test_feats_selected = test_feats_filtered[:, selected_indices]
                print(f"Selected features: {selected_indices}")
                print(f"Subject {subject_index} - train_feats_selected shape: {train_feats_selected.shape}")
                print(f"Subject {subject_index} - test_feats_selected shape: {test_feats_selected.shape}")

                if use_lda:
                    print("Applying Fisher projection...")
                    projection_matrix, explained_variance_ratio = fisher_projection(train_feats_selected, train_labels)
                    # if projection_matrix:
                    train_feats_projected = train_feats_selected @ projection_matrix
                    test_feats_projected = test_feats_selected @ projection_matrix

                else:
                    train_feats_projected = train_feats_selected
                    test_feats_projected = test_feats_selected

                # Train and evaluate classifier
                print("Training classifier...")

                clf = TraditionalClassifier(n_neighbors=n_neighbors)
                clf.fit(train_feats_projected, train_labels)  # Pass projected features directly

                # Evaluate
                train_pred = clf.predict(train_feats_projected)
                test_pred = clf.predict(test_feats_projected)

                train_acc = accuracy_score(train_labels, train_pred)
                test_acc = accuracy_score(test_labels, test_pred)

                # Storing results
                subject_results.append({
                    'subject_index': subject_index,
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'train_report': classification_report(train_labels, train_pred),
                    'test_report': classification_report(test_labels, test_pred)
                })

                accuracy_progression.append(test_acc)

                # Plot confusion matrices
                print("Plotting confusion matrices...")
                plot_confusion_matrix(
                    train_labels, train_pred,
                    test_labels, test_pred,
                    subject_index, dataset_type
                )

                print(f"\nSubject {subject_index} Results:")
                print(f"Train Accuracy: {train_acc:.3f}")
                print(f"Test Accuracy: {test_acc:.3f}")

            except Exception as e:
                print(f"Error processing subject {subject_index}: {str(e)}")
                continue

        # Plot training progress
        print("Plotting training progress...")
        plot_training_progress(accuracy_progression, dataset_type)

        # Plot feature importance heatmap
        feature_importance_matrix = np.array(feature_importance_scores)
        print("Plotting feature importance heatmap...")
        plot_feature_heatmap(feature_importance_matrix, dataset_type)

        # Store dataset in results
        if subject_results:
            results[dataset_type] = {
                'subject_results': subject_results,
                'mean_test_acc': np.mean([r['test_acc'] for r in subject_results]),
                'std_test_acc': np.std([r['test_acc'] for r in subject_results]),
                'feature_importance_matrix': feature_importance_matrix
            }

            print(f"\nDataset {dataset_type} Summary:")
            print(f"Mean Test Accuracy: {results[dataset_type]['mean_test_acc']:.3f} "
                  f"± {results[dataset_type]['std_test_acc']:.3f}")

    # Plot dataset comparison
    print("Plotting dataset comparison...")
    plot_dataset_comparison(results)

    return results


def main():
    if not os.path.exists('results'):
        os.makedirs('results')
    # # Configuration for the varying N, for 100 -300, auto-changing the dataset on
    # completion of the previous
    dataset_types = [100, 200, 300]
    use_lda = True
    use_sfs = True
    n_features = 10
    n_neighbors = 5  # KNN parameter

    # Running the classification function
    results = classification(
        dataset_types=dataset_types,
        use_lda=use_lda,
        # use_sfs=use_sfs,
        n_features=n_features,
        n_neighbors=n_neighbors
    )

    # Results visualization
    plt.figure(figsize=(12, 6))
    for i, dataset_type in enumerate(dataset_types):
        if dataset_type in results:
            plt.subplot(1, len(dataset_types), i + 1)
            accuracies = [r['test_acc'] for r in results[dataset_type]['subject_results']]
            plt.bar(range(len(accuracies)), accuracies)
            plt.title(f'Dataset {dataset_type}\nKNN (k={n_neighbors})')
            plt.xlabel('Subject Index')
            plt.ylabel('Test Accuracy')
            plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig('results/accuracies_comparison.png')
    plt.close()


    # Zip results
    # os.system("zip -r results.zip results/")
    # from google.colab import files
    # files.download("results.zip")


main()






