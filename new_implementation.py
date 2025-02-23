import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, f1_score
from sklearn.model_selection import LeaveOneGroupOut
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from concurrent.futures import ProcessPoolExecutor

# Ensure results directory exists
os.makedirs('results', exist_ok=True)


def load_dataset(verbose=True, dataset_type=100):
    """
    Load dataset using LeaveOneGroupOut cross-validation
    """
    try:
        filename = f"Taiji_dataset_{dataset_type}.csv"
        dataset = np.loadtxt(filename, delimiter=",", dtype=float, skiprows=1, usecols=range(0, 70))

        # Extract features, labels, and subject IDs
        person_idxs = dataset[:, -1]
        labels = dataset[:, -2]
        features = dataset[:, :-2]

        if verbose:
            print(f'Dataset {dataset_type} Loaded')
            print(f'\t# of Classes: {len(np.unique(labels))}')
            print(f'\t# of Features: {features.shape[1]}')
            print(f'\t# of Samples: {features.shape[0]}')
            print(f'\t# of Subjects: {len(np.unique(person_idxs))}')

        return features, labels, person_idxs

    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None, None, None


class TraditionalClassifier(KNeighborsClassifier):
    """KNN-based classifier with integrated evaluation methods."""

    def __init__(self, n_neighbors=10):
        super().__init__(n_neighbors=n_neighbors)

    def evaluate(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Evaluate model performance on train, validation and test sets."""
        self.fit(X_train, y_train)

        train_pred = self.predict(X_train)
        val_pred = self.predict(X_val)
        test_pred = self.predict(X_test)

        train_accuracy = accuracy_score(y_train, train_pred)
        val_accuracy = accuracy_score(y_val, val_pred)
        test_accuracy = accuracy_score(y_test, test_pred)

        return {
            'train_acc': train_accuracy,
            'val_acc': val_accuracy,
            'test_acc': test_accuracy,
            'train_pred': train_pred,
            'val_pred': val_pred,
            'test_pred': test_pred
        }


class MLP(nn.Module):
    """Simple MLP classifier with two hidden layers."""

    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.layers(x)


def convert_features_to_loader(X, y, batch_size):
    """Convert numpy arrays to PyTorch DataLoader."""
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y.astype(int))
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_class_weights(y_train):
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    return torch.FloatTensor(class_weights)


def train_epoch(model, loader, criterion, optimizer):
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / len(loader), correct / total


def evaluate_model(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted')

    return total_loss / len(loader), accuracy, f1


def predict_model(model, loader):
    """Get predictions from model."""
    model.eval()
    all_preds = []
    all_targets = []
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return np.array(all_preds), correct / total


def fisher_projection(X: np.ndarray, y: np.ndarray, n_components: int = None) -> tuple:
    """
    Implements Fisher Linear Discriminant Analysis for dimensionality reduction.

    Args:
        X: Input features matrix
        y: Target labels
        n_components: Number of components to keep

    Returns:
        tuple: (projection_matrix, explained_variance_ratio)
    """
    classes = np.unique(y)
    n_classes = len(classes)
    n_features = X.shape[1]

    if n_components is None:
        n_components = min(n_classes - 1, n_features)

    # Calculate means
    overall_mean = np.mean(X, axis=0)
    class_means = {c: np.mean(X[y == c], axis=0) for c in classes}

    # Calculate scatter matrices
    Sw = np.zeros((n_features, n_features))  # Within-class scatter
    Sb = np.zeros((n_features, n_features))  # Between-class scatter

    for c in classes:
        X_c = X[y == c]
        n_c = len(X_c)
        centered_X = X_c - class_means[c]
        Sw += centered_X.T @ centered_X

        diff = (class_means[c] - overall_mean).reshape(-1, 1)
        Sb += n_c * (diff @ diff.T)

    # Add regularization
    Sw_reg = Sw + 1e-4 * np.eye(n_features)

    try:
        # Solve generalized eigenvalue problem
        eigenvalues, eigenvectors = np.linalg.eigh(np.linalg.inv(Sw_reg) @ Sb)

        # Sort eigenvalues and eigenvectors
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
        projection_matrix = eigenvectors[:, :n_components]
        explained_variance_ratio = explained_variance_ratio[:n_components]

        return projection_matrix, explained_variance_ratio

    except np.linalg.LinAlgError as e:
        print(f"Error in Fisher LDA: {e}")
        return None, None


def sequential_forward_selection(X, y, n_features=10, max_features_to_check=20):
    """
    Optimized sequential forward selection for feature selection.

    Args:
        X: Input features
        y: Target labels
        n_features: Number of features to select
        max_features_to_check: Maximum number of remaining features to evaluate at each step
    """
    start_time = time.time()
    n_total_features = X.shape[1]
    selected_features = []
    remaining_features = list(range(n_total_features))
    selected_scores = []

    # Use a simpler classifier for selection
    base_classifier = KNeighborsClassifier(n_neighbors=5)

    for i in range(n_features):
        best_score = 0
        best_feature = None

        # If there are too many remaining features, sample a subset
        features_to_check = remaining_features
        if len(remaining_features) > max_features_to_check:
            np.random.shuffle(features_to_check)
            features_to_check = features_to_check[:max_features_to_check]

        for feature in features_to_check:
            current_features = selected_features + [feature]
            X_subset = X[:, current_features]

            # Use a simple train/test split instead of cross-validation for speed
            n_samples = X.shape[0]
            indices = np.random.permutation(n_samples)
            train_idx, test_idx = indices[:int(0.7 * n_samples)], indices[int(0.7 * n_samples):]

            clf = base_classifier.fit(X_subset[train_idx], y[train_idx])
            score = accuracy_score(y[test_idx], clf.predict(X_subset[test_idx]))

            if score > best_score:
                best_score = score
                best_feature = feature

        if best_feature is not None:
            selected_features.append(best_feature)
            selected_scores.append(best_score)
            remaining_features.remove(best_feature)

        if (i + 1) % 5 == 0:
            elapsed = time.time() - start_time
            print(f"Selected {i + 1}/{n_features} features in {elapsed:.2f} seconds")

    return selected_features, selected_scores


def apply_feature_processing(X_train, X_val, X_test, y_train, use_lda=False):
    """Apply feature selection and optionally LDA projection."""
    print("Starting feature selection...")
    start_time = time.time()

    # Feature selection
    selected_features, feature_scores = sequential_forward_selection(X_train, y_train)

    print(f"Feature selection completed in {time.time() - start_time:.2f} seconds")

    # Apply selection
    X_train_selected = X_train[:, selected_features]
    X_val_selected = X_val[:, selected_features]
    X_test_selected = X_test[:, selected_features]

    if use_lda:
        print("Applying Fisher LDA projection...")
        projection_matrix, _ = fisher_projection(X_train_selected, y_train)
        if projection_matrix is not None:
            X_train_final = X_train_selected @ projection_matrix
            X_val_final = X_val_selected @ projection_matrix
            X_test_final = X_test_selected @ projection_matrix
        else:
            print("Fisher LDA failed, using selected features without projection")
            X_train_final = X_train_selected
            X_val_final = X_val_selected
            X_test_final = X_test_selected
    else:
        X_train_final = X_train_selected
        X_val_final = X_val_selected
        X_test_final = X_test_selected

    return X_train_final, X_val_final, X_test_final, selected_features, feature_scores


def plot_visualizations(y_val, y_test, val_pred, test_pred, train_metrics, val_metrics,
                        subject_idx, model_type, dataset_type,
                        selected_features, feature_scores):
    """Plot all visualizations for model analysis with enhanced aesthetics."""
    # Set a modern style with improved aesthetics
    plt.style.use('seaborn-v0_8-whitegrid')

    # Custom color palettes - Using blues and greens for better readability
    confusion_cmap = sns.color_palette("Blues", as_cmap=True)  # Blue scale for confusion matrix
    feature_cmap = sns.color_palette([
        (0.95, 0.95, 0.95),  # Very light gray
        (0.0, 0.5, 0.3),  # Forest green
    ], as_cmap=True)  # Green scale for feature importance

    line_colors = ['#2c5f2d', '#00a878']  # Dark green and sea green for line plots

    # Confusion Matrix - Validation
    plt.figure(figsize=(16, 7))
    plt.subplot(1, 2, 1)
    cm_val = confusion_matrix(y_val, val_pred)
    sns.heatmap(cm_val, annot=True, fmt='d', cmap=confusion_cmap,
                annot_kws={"size": 12, "weight": "bold"},
                linewidths=0.5, linecolor='white',
                cbar_kws={"shrink": 0.8})
    plt.title(f'{model_type} - Validation Confusion Matrix\n(Dataset {dataset_type}, Subject {subject_idx})',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicted', fontsize=12, labelpad=10)
    plt.ylabel('True', fontsize=12, labelpad=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Add a subtle background grid
    plt.grid(False)
    ax = plt.gca()
    ax.set_facecolor('#f8f9fa')

    # Confusion Matrix - Test
    plt.subplot(1, 2, 2)
    cm_test = confusion_matrix(y_test, test_pred)
    sns.heatmap(cm_test, annot=True, fmt='d', cmap=confusion_cmap,
                annot_kws={"size": 12, "weight": "bold"},
                linewidths=0.5, linecolor='white',
                cbar_kws={"shrink": 0.8})
    plt.title(f'{model_type} - Test Confusion Matrix\n(Dataset {dataset_type}, Subject {subject_idx})',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicted', fontsize=12, labelpad=10)
    plt.ylabel('True', fontsize=12, labelpad=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Add a subtle background grid
    plt.grid(False)
    ax = plt.gca()
    ax.set_facecolor('#f8f9fa')

    plt.tight_layout()
    plt.savefig(f'results/{model_type}_confusion_matrices_dataset_{dataset_type}_subject_{subject_idx}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Feature Importance with improved visualization
    plt.figure(figsize=(14, 6))
    feature_importance = np.zeros(70)
    feature_importance[selected_features] = feature_scores

    # Create a more visually informative heatmap
    ax = sns.heatmap(feature_importance.reshape(1, -1),
                     cmap=feature_cmap,
                     xticklabels=list(range(0, 70, 5)),
                     yticklabels=[f'Subject {subject_idx}'],
                     linewidths=0.1, linecolor='gray',
                     cbar_kws={"label": "Feature Importance", "shrink": 0.8})

    # Improve readability
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.set_ylabel("Feature Importance", fontsize=12, rotation=270, labelpad=20)

    plt.title(f'Feature Importance Analysis\nDataset {dataset_type}, Subject {subject_idx}',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Feature Index', fontsize=12, labelpad=10)

    # Add vertical gridlines for better feature tracking
    for i in range(5, 70, 5):
        plt.axvline(x=i, color='gray', linestyle='--', alpha=0.2)

    # Set background color
    ax.set_facecolor('#f8f9fa')

    plt.tight_layout()
    plt.savefig(f'results/feature_importance_dataset_{dataset_type}_subject_{subject_idx}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Training Metrics for MLP with enhanced visualization
    if train_metrics and val_metrics:
        plt.figure(figsize=(18, 7))

        # Loss subplot
        plt.subplot(1, 2, 1)
        plt.plot(train_metrics['loss'], color=line_colors[0], linewidth=2.5,
                 label='Training Loss', alpha=0.9)
        plt.plot(val_metrics['loss'], color=line_colors[1], linewidth=2.5,
                 label='Validation Loss', alpha=0.9)
        plt.title(f'{model_type} - Loss Progression\nDataset {dataset_type}, Subject {subject_idx}',
                  fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Epoch', fontsize=12, labelpad=10)
        plt.ylabel('Loss', fontsize=12, labelpad=10)
        plt.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='lightgray')
        plt.grid(True, linestyle='--', alpha=0.3)

        # Add shaded region between curves
        plt.fill_between(range(len(train_metrics['loss'])),
                         train_metrics['loss'], val_metrics['loss'],
                         color='#e6f3e6', alpha=0.3)

        # Accuracy subplot
        plt.subplot(1, 2, 2)
        plt.plot(train_metrics['acc'], color=line_colors[0], linewidth=2.5,
                 label='Training Accuracy', alpha=0.9)
        plt.plot(val_metrics['acc'], color=line_colors[1], linewidth=2.5,
                 label='Validation Accuracy', alpha=0.9)
        plt.title(f'{model_type} - Accuracy Progression\nDataset {dataset_type}, Subject {subject_idx}',
                  fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Epoch', fontsize=12, labelpad=10)
        plt.ylabel('Accuracy', fontsize=12, labelpad=10)
        plt.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='lightgray')
        plt.grid(True, linestyle='--', alpha=0.3)

        # Add shaded region between curves
        plt.fill_between(range(len(train_metrics['acc'])),
                         train_metrics['acc'], val_metrics['acc'],
                         color='#e6f3e6', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'results/{model_type}_training_metrics_dataset_{dataset_type}_subject_{subject_idx}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()


def plot_comparative_results(results, dataset_types):
    """Plot comparative results across different models and datasets with enhanced aesthetics."""
    plt.style.use('seaborn-v0_8-whitegrid')

    # Custom vibrant palette that's colorblind-friendly
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    hatches = ['', '///', '...', 'xxx']

    # Metrics to compare with better labels
    metrics = {
        'test_acc': 'Test Accuracy',
        'val_acc': 'Validation Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1 Score'
    }

    # Plot for each metric
    for metric_key, metric_label in metrics.items():
        plt.figure(figsize=(16, 9))

        # Set up positions for grouped bars
        model_configs = list(results[dataset_types[0]].keys())
        # Create more readable model config names
        model_config_labels = [config.replace('_', ' + ').title() for config in model_configs]

        n_configs = len(model_configs)
        n_datasets = len(dataset_types)
        bar_width = 0.8 / n_datasets
        dataset_positions = np.arange(n_configs)

        # Plot bars for each dataset with enhanced styling
        for i, dataset_type in enumerate(dataset_types):
            if dataset_type not in results:
                continue

            mean_values = [results[dataset_type][config][0][metric_key] for config in model_configs]
            std_values = [results[dataset_type][config][1][metric_key] for config in model_configs]
            positions = [p + (i - n_datasets / 2 + 0.5) * bar_width for p in dataset_positions]

            # Add bars with enhanced styling
            bars = plt.bar(positions, mean_values, width=bar_width,
                           yerr=std_values, capsize=5,
                           color=colors[i % len(colors)],
                           hatch=hatches[i % len(hatches)],
                           edgecolor='gray', linewidth=1,
                           alpha=0.85,
                           error_kw={'ecolor': 'black', 'elinewidth': 1.5, 'capthick': 1.5},
                           label=f'Dataset {dataset_type}')

            # Add value labels on top of bars
            for bar, value in zip(bars, mean_values):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{value:.3f}', ha='center', va='bottom',
                         fontsize=8, rotation=0, fontweight='bold')

        # Customize plot with enhanced styling
        plt.xticks(dataset_positions, model_config_labels, fontsize=11, rotation=10)
        plt.xlabel('Model Configuration', fontsize=14, labelpad=15)
        plt.ylabel(metric_label, fontsize=14, labelpad=15)
        plt.title(f'Comparison of {metric_label} Across Models and Datasets',
                  fontsize=16, fontweight='bold', pad=20)

        # Enhance legend
        legend = plt.legend(fontsize=12, frameon=True, facecolor='white',
                            edgecolor='lightgray', bbox_to_anchor=(1.02, 1), loc='upper left')
        frame = legend.get_frame()
        frame.set_boxstyle('round,pad=0.5')

        # Enhance grid and layout
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make room for legend

        # Add subtle background shading to enhance readability
        plt.gca().set_facecolor('#f8f9fa')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        plt.savefig(f'results/comparison_{metric_key}.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Feature importance variance across subjects with enhanced visualization
    plt.figure(figsize=(16, 9))

    # Custom diverging colormap for line visualization
    line_styles = ['-', '--', '-.', ':']
    line_widths = [2.5, 2.5, 2.5, 2.5]

    # Background enhancement
    ax = plt.gca()
    ax.set_facecolor('#f8f9fa')
    ax.grid(True, linestyle='--', alpha=0.5, color='gray')

    for i, dataset_type in enumerate(dataset_types):
        if dataset_type not in results:
            continue

        try:
            # Load feature importance data
            feature_std = np.load(f'results/feature_importance_variance_dataset_{dataset_type}.npy')

            # Plot with enhanced styling
            plt.plot(range(70), feature_std,
                     color=colors[i % len(colors)],
                     linestyle=line_styles[i % len(line_styles)],
                     linewidth=line_widths[i % len(line_widths)],
                     label=f'Dataset {dataset_type}',
                     alpha=0.85)

            # Add shaded area under the curve
            plt.fill_between(range(70), 0, feature_std,
                             color=colors[i % len(colors)], alpha=0.15)
        except:
            print(f"Could not load feature importance variance for dataset {dataset_type}")

    # Enhance feature markers
    for i in range(0, 70, 5):
        plt.axvline(x=i, color='darkgray', linestyle=':', alpha=0.4, linewidth=1)

    # Enhance axis labels and title
    plt.xlabel('Feature Index', fontsize=14, labelpad=15)
    plt.ylabel('Standard Deviation of Importance', fontsize=14, labelpad=15)
    plt.title('Feature Importance Variance Across Subjects',
              fontsize=16, fontweight='bold', pad=20)

    # Enhance ticks
    plt.xticks(range(0, 71, 5), fontsize=10)
    plt.yticks(fontsize=10)

    # Enhance legend
    legend = plt.legend(fontsize=12, frameon=True, facecolor='white',
                        edgecolor='lightgray', loc='upper right')
    frame = legend.get_frame()
    frame.set_boxstyle('round,pad=0.5')

    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('results/feature_importance_variance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_visualizations(y_val, y_test, val_pred, test_pred, train_metrics, val_metrics,
                        subject_idx, model_type, dataset_type,
                        selected_features, feature_scores):
    """Plot all visualizations for model analysis with enhanced aesthetics."""
    # Set a modern style with improved aesthetics
    plt.style.use('seaborn-v0_8-darkgrid')

    # Custom color palettes
    confusion_cmap = sns.color_palette("mako", as_cmap=True)  # More vibrant and modern
    feature_cmap = sns.color_palette("viridis", as_cmap=True)  # Better for feature importance
    line_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Expanded color palette

    # Confusion Matrix - Validation
    plt.figure(figsize=(16, 7))
    plt.subplot(1, 2, 1)
    cm_val = confusion_matrix(y_val, val_pred)
    sns.heatmap(cm_val, annot=True, fmt='d', cmap=confusion_cmap,
                annot_kws={"size": 12, "weight": "bold", "color": 'white'},  # White text for better contrast
                linewidths=0.5, linecolor='white',
                cbar_kws={"shrink": 0.8, "label": "Count"})
    plt.title(f'{model_type} - Validation Confusion Matrix\n(Dataset {dataset_type}, Subject {subject_idx})',
              fontsize=14, fontweight='bold', pad=20, color='#333333')
    plt.xlabel('Predicted', fontsize=12, labelpad=10, color='#333333')
    plt.ylabel('True', fontsize=12, labelpad=10, color='#333333')
    plt.xticks(fontsize=10, color='#333333')
    plt.yticks(fontsize=10, color='#333333')

    # Confusion Matrix - Test
    plt.subplot(1, 2, 2)
    cm_test = confusion_matrix(y_test, test_pred)
    sns.heatmap(cm_test, annot=True, fmt='d', cmap=confusion_cmap,
                annot_kws={"size": 12, "weight": "bold", "color": 'white'},
                linewidths=0.5, linecolor='white',
                cbar_kws={"shrink": 0.8, "label": "Count"})
    plt.title(f'{model_type} - Test Confusion Matrix\n(Dataset {dataset_type}, Subject {subject_idx})',
              fontsize=14, fontweight='bold', pad=20, color='#333333')
    plt.xlabel('Predicted', fontsize=12, labelpad=10, color='#333333')
    plt.ylabel('True', fontsize=12, labelpad=10, color='#333333')
    plt.xticks(fontsize=10, color='#333333')
    plt.yticks(fontsize=10, color='#333333')

    plt.tight_layout()
    plt.savefig(f'results/{model_type}_confusion_matrices_dataset_{dataset_type}_subject_{subject_idx}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Feature Importance with improved visualization
    plt.figure(figsize=(14, 8))
    feature_importance = np.zeros(70)  # Assuming 70 features
    feature_importance[selected_features] = feature_scores

    # Create a more visually informative heatmap
    ax = sns.heatmap(feature_importance.reshape(1, -1),
                     cmap=feature_cmap,
                     xticklabels=list(range(0, 70, 5)),
                     yticklabels=[f'Subject {subject_idx}'],
                     linewidths=0.1, linecolor='gray',
                     cbar_kws={"label": "Feature Importance Score", "shrink": 0.8})

    # Improve readability
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.set_ylabel("Feature Importance Score", fontsize=12, rotation=270, labelpad=20)

    plt.title(f'Feature Importance - Dataset {dataset_type}, Subject {subject_idx}',
              fontsize=16, fontweight='bold', pad=20, color='#333333')
    plt.xlabel('Feature Index', fontsize=14, labelpad=10, color='#333333')

    # Add a horizontal line grid to better track feature indices
    for i in range(5, 70, 5):
        plt.axvline(x=i, color='gray', linestyle='--', alpha=0.3)

    plt.savefig(f'results/feature_importance_dataset_{dataset_type}_subject_{subject_idx}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Training Metrics for MLP with enhanced visualization
    if train_metrics and val_metrics:
        plt.figure(figsize=(18, 7))

        # Loss subplot
        plt.subplot(1, 2, 1)
        plt.plot(train_metrics['loss'], color=line_colors[0], linewidth=2.5,
                 label='Training Loss', alpha=0.8)
        plt.plot(val_metrics['loss'], color=line_colors[1], linewidth=2.5,
                 label='Validation Loss', alpha=0.8)
        plt.title(f'{model_type} - Loss Progression\n(Dataset {dataset_type}, Subject {subject_idx})',
                  fontsize=14, fontweight='bold', pad=20, color='#333333')
        plt.xlabel('Epoch', fontsize=12, labelpad=10, color='#333333')
        plt.ylabel('Loss', fontsize=12, labelpad=10, color='#333333')
        plt.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='lightgray')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Add shaded region for visualization enhancement
        plt.fill_between(range(len(train_metrics['loss'])),
                         train_metrics['loss'], val_metrics['loss'],
                         color='lightgray', alpha=0.3)

        # Accuracy subplot
        plt.subplot(1, 2, 2)
        plt.plot(train_metrics['acc'], color=line_colors[0], linewidth=2.5,
                 label='Training Accuracy', alpha=0.8)
        plt.plot(val_metrics['acc'], color=line_colors[1], linewidth=2.5,
                 label='Validation Accuracy', alpha=0.8)
        plt.title(f'{model_type} - Accuracy Progression\n(Dataset {dataset_type}, Subject {subject_idx})',
                  fontsize=14, fontweight='bold', pad=20, color='#333333')
        plt.xlabel('Epoch', fontsize=12, labelpad=10, color='#333333')
        plt.ylabel('Accuracy', fontsize=12, labelpad=10, color='#333333')
        plt.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='lightgray')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Add shaded region for visualization enhancement
        plt.fill_between(range(len(train_metrics['acc'])),
                         train_metrics['acc'], val_metrics['acc'],
                         color='lightgray', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'results/{model_type}_training_metrics_dataset_{dataset_type}_subject_{subject_idx}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()


def plot_comparative_results(results, dataset_types):
    """Plot comparative results across different models and datasets with enhanced aesthetics."""
    plt.style.use('seaborn-v0_8-darkgrid')

    # Custom vibrant palette that's colorblind-friendly
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#34495e']
    hatches = ['', '///', '...', 'xxx', '+++', 'OOO']

    # Metrics to compare with better labels
    metrics = {
        'test_acc': 'Test Accuracy',
        'val_acc': 'Validation Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1 Score'
    }

    # Plot for each metric
    for metric_key, metric_label in metrics.items():
        plt.figure(figsize=(16, 9))

        # Set up positions for grouped bars
        model_configs = list(results[dataset_types[0]].keys())
        model_config_labels = [config.replace('_', ' + ').title() for config in model_configs]

        n_configs = len(model_configs)
        n_datasets = len(dataset_types)
        bar_width = 0.8 / n_datasets
        dataset_positions = np.arange(n_configs)

        # Plot bars for each dataset with enhanced styling
        for i, dataset_type in enumerate(dataset_types):
            if dataset_type not in results:
                continue

            mean_values = [results[dataset_type][config][0][metric_key] for config in model_configs]
            std_values = [results[dataset_type][config][1][metric_key] for config in model_configs]
            positions = [p + (i - n_datasets / 2 + 0.5) * bar_width for p in dataset_positions]

            # Add bars with enhanced styling
            bars = plt.bar(positions, mean_values, width=bar_width,
                           yerr=std_values, capsize=5,
                           color=colors[i % len(colors)],
                           hatch=hatches[i % len(hatches)],
                           edgecolor='gray', linewidth=1,
                           alpha=0.85,
                           error_kw={'ecolor': 'black', 'elinewidth': 1.5, 'capthick': 1.5},
                           label=f'Dataset {dataset_type}')

            # Add value labels on top of bars
            for bar, value in zip(bars, mean_values):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{value:.3f}', ha='center', va='bottom',
                         fontsize=8, rotation=0, fontweight='bold')

        # Customize plot with enhanced styling
        plt.xticks(dataset_positions, model_config_labels, fontsize=11, rotation=10, color='#333333')
        plt.xlabel('Model Configuration', fontsize=14, labelpad=15, color='#333333')
        plt.ylabel(metric_label, fontsize=14, labelpad=15, color='#333333')
        plt.title(f'Comparison of {metric_label} Across Models and Datasets',
                  fontsize=16, fontweight='bold', pad=20, color='#333333')

        # Enhance legend
        legend = plt.legend(fontsize=12, frameon=True, facecolor='white',
                            edgecolor='lightgray', bbox_to_anchor=(1.02, 1), loc='upper left')
        frame = legend.get_frame()
        frame.set_boxstyle('round,pad=0.5')

        # Enhance grid and layout
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make room for legend

        # Add subtle background shading to enhance readability
        plt.gca().set_facecolor('#f8f9fa')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        plt.savefig(f'results/comparison_{metric_key}.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Feature importance variance across subjects with enhanced visualization
    plt.figure(figsize=(16, 9))

    # Custom diverging colormap for line visualization
    line_styles = ['-', '--', '-.', ':']
    line_widths = [2.5, 2.5, 2.5, 2.5]

    # Background enhancement
    ax = plt.gca()
    ax.set_facecolor('#f8f9fa')
    ax.grid(True, linestyle='--', alpha=0.5, color='gray')

    for i, dataset_type in enumerate(dataset_types):
        if dataset_type not in results:
            continue

        try:
            # Load feature importance data
            feature_std = np.load(f'results/feature_importance_variance_dataset_{dataset_type}.npy')

            # Plot with enhanced styling
            plt.plot(range(70), feature_std,
                     color=colors[i % len(colors)],
                     linestyle=line_styles[i % len(line_styles)],
                     linewidth=line_widths[i % len(line_widths)],
                     label=f'Dataset {dataset_type}',
                     alpha=0.85)

            # Add shaded area under the curve
            plt.fill_between(range(70), 0, feature_std,
                             color=colors[i % len(colors)], alpha=0.15)
        except:
            print(f"Could not load feature importance variance for dataset {dataset_type}")

    # Enhance feature markers
    for i in range(0, 70, 5):
        plt.axvline(x=i, color='darkgray', linestyle=':', alpha=0.4, linewidth=1)

    # Enhance axis labels and title
    plt.xlabel('Feature Index', fontsize=14, labelpad=15, color='#333333')
    plt.ylabel('Standard Deviation of Importance', fontsize=14, labelpad=15, color='#333333')
    plt.title('Feature Importance Variance Across Subjects',
              fontsize=16, fontweight='bold', pad=20, color='#333333')

    # Enhance ticks
    plt.xticks(range(0, 71, 5), fontsize=10, color='#333333')
    plt.yticks(fontsize=10, color='#333333')

    # Enhance legend
    legend = plt.legend(fontsize=12, frameon=True, facecolor='white',
                        edgecolor='lightgray', loc='upper right')
    frame = legend.get_frame()
    frame.set_boxstyle('round,pad=0.5')

    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('results/feature_importance_variance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def process_subject(subject_idx, features, labels, subjects, dataset_type, model_type, use_lda):
    """Process a single subject for leave-one-subject-out validation."""
    print(f"Processing Subject {subject_idx} with {model_type} model" +
          f" and{'' if use_lda else ' no'} LDA (Dataset {dataset_type})")
    start_time = time.time()

    # Implement proper LOSO: test data is from subject_idx, rest is split into train/val
    test_mask = subjects == subject_idx
    train_val_mask = ~test_mask

    X_test = features[test_mask]
    y_test = labels[test_mask]

    # Get all non-test data
    X_train_val = features[train_val_mask]
    y_train_val = labels[train_val_mask]
    subjects_train_val = subjects[train_val_mask]

    # Split remaining data into train and validation, preserving subject grouping
    unique_train_subjects = np.unique(subjects_train_val)
    np.random.shuffle(unique_train_subjects)

    # 80% of remaining subjects for training, 10% for validation
    n_train_subjects = int(0.9 * len(unique_train_subjects))
    train_subjects = unique_train_subjects[:n_train_subjects]
    val_subjects = unique_train_subjects[n_train_subjects:]

    train_mask = np.isin(subjects_train_val, train_subjects)
    val_mask = np.isin(subjects_train_val, val_subjects)

    X_train = X_train_val[train_mask]
    y_train = y_train_val[train_mask]
    X_val = X_train_val[val_mask]
    y_val = y_train_val[val_mask]

    print(f"Data split - Train: {X_train.shape[0]} samples, " +
          f"Validation: {X_val.shape[0]} samples, Test: {X_test.shape[0]} samples")

    # Process features
    X_train_proc, X_val_proc, X_test_proc, selected_features, feature_scores = \
        apply_feature_processing(X_train, X_val, X_test, y_train, use_lda)

    train_metrics = val_metrics = None

    if model_type == 'traditional':
        clf = TraditionalClassifier()
        results = clf.evaluate(X_train_proc, y_train, X_val_proc, y_val, X_test_proc, y_test)

        train_acc = results['train_acc']
        val_acc = results['val_acc']
        test_acc = results['test_acc']
        val_pred = results['val_pred']
        test_pred = results['test_pred']

    else:  # MLP
        # Convert to PyTorch format
        train_loader = convert_features_to_loader(X_train_proc, y_train, 32)
        val_loader = convert_features_to_loader(X_val_proc, y_val, 32)
        test_loader = convert_features_to_loader(X_test_proc, y_test, 32)

        # Define model
        model = MLP(X_train_proc.shape[1], len(np.unique(labels)))

        class_weights = get_class_weights(y_train)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        # criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=5, verbose=True, min_lr=1e-6
        )

        # Training metrics tracking
        train_metrics = {'loss': [], 'acc': [], 'f1': []}
        val_metrics = {'loss': [], 'acc': [], 'f1': []}

        # Early stopping
        best_val_acc = 0
        best_val_f1 = 0
        patience = 10
        patience_counter = 0

        # Train with early stopping
        epochs = 100
        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, criterion)


            #"-------------------"




            train_metrics['loss'].append(train_loss)
            train_metrics['acc'].append(train_acc)
            val_metrics['loss'].append(val_loss)
            val_metrics['acc'].append(val_acc)

            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}/{epochs}")
                    break

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        # Load best model
        model.load_state_dict(best_model_state)

        # Evaluate on all sets
        _, train_acc = predict_model(model, train_loader)
        val_pred, val_acc = predict_model(model, val_loader)
        test_pred, test_acc = predict_model(model, test_loader)

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_pred, average='weighted')
    metrics = {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    # Plot visualizations
    plot_visualizations(y_val, y_test, val_pred, test_pred,
                        train_metrics, val_metrics,
                        subject_idx, f"{model_type}{'_lda' if use_lda else ''}",
                        dataset_type, selected_features, feature_scores)

    elapsed_time = time.time() - start_time
    print(f"Subject {subject_idx} processing completed in {elapsed_time:.2f} seconds")

    # Return results for this subject
    return {
        'subject_idx': subject_idx,
        'metrics': metrics,
        'feature_importance': feature_scores,
        'selected_features': selected_features
    }


def classification(features, labels, subjects, dataset_type, model_type='traditional', use_lda=False):
    """
    Perform classification with optional LDA projection and LOSO cross-validation.

    Args:
        features: Input features
        labels: Target labels
        subjects: Subject identifiers for LOSO
        dataset_type: Dataset type identifier (100, 200, or 300)
        model_type: 'traditional' or 'mlp'
        use_lda: Whether to use Fisher LDA projection
    """
    print(f"Starting classification for dataset {dataset_type} with {model_type}" +
          f" model and{'' if use_lda else ' no'} LDA")
    total_start_time = time.time()

    # Get unique subjects
    unique_subjects = np.unique(subjects)

    # List to hold results
    subject_results = []

    # Process each subject independently
    for subject_idx in unique_subjects:
        result = process_subject(
            subject_idx, features, labels, subjects,
            dataset_type, model_type, use_lda
        )
        subject_results.append(result)

    # Compile results
    metrics_list = [r['metrics'] for r in subject_results]

    # Calculate overall metrics
    mean_metrics = {k: np.mean([r[k] for r in metrics_list]) for k in metrics_list[0].keys()}
    std_metrics = {k: np.std([r[k] for r in metrics_list]) for k in metrics_list[0].keys()}

    # Compile feature importance statistics
    all_feature_importance = np.zeros((len(unique_subjects), 70))
    for i, result in enumerate(subject_results):
        selected_features = result['selected_features']
        feature_scores = result['feature_importance']
        all_feature_importance[i, selected_features] = feature_scores

    # Save feature importance variance
    feature_std = np.std(all_feature_importance, axis=0)
    np.save(f'results/feature_importance_variance_dataset_{dataset_type}.npy', feature_std)

    total_elapsed_time = time.time() - total_start_time
    print(f"Classification completed in {total_elapsed_time:.2f} seconds")

    return mean_metrics, std_metrics


def main():
    """Run the complete classification pipeline with all configurations."""
    dataset_types = [100, 200, 300]
    all_results = {}

    for dataset_type in dataset_types:
        print(f"\n===== Processing Dataset {dataset_type} =====")
        features, labels, subjects = load_dataset(verbose=True, dataset_type=dataset_type)

        if features is None:
            print(f"Skipping dataset {dataset_type} due to loading error")
            continue

        results = {}

        for model_type in ['mlp', 'traditional']:
            for use_lda in [True, False]:
                key = f"{model_type}{'_lda' if use_lda else ''}"
                print(f"\nRunning {key.upper()} on dataset {dataset_type}")

                results[key] = classification(
                    features, labels, subjects,
                    dataset_type=dataset_type,
                    model_type=model_type,
                    use_lda=use_lda
                )

        all_results[dataset_type] = results

        # Print results for this dataset
        print(f"\nResults for Dataset {dataset_type}:")
        for model_name, (metrics, std) in results.items():
            print(f"\n{model_name.upper()}:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f} ± {std[metric]:.4f}")

    # Plot comparative results across datasets
    plot_comparative_results(all_results, dataset_types)

    # os.system("zip -r results.zip results/")
    # from google.colab import files
    # files.download("results.zip")

    # Generate summary table
    print("\n===== SUMMARY ACROSS ALL DATASETS =====")
    for dataset_type in dataset_types:
        if dataset_type not in all_results:
            continue

        print(f"\nDataset {dataset_type}:")
        for model_name, (metrics, std) in all_results[dataset_type].items():
            print(f"{model_name}: Acc={metrics['test_acc']:.4f}±{std['test_acc']:.4f}, " +
                  f"Val Acc={metrics['val_acc']:.4f}±{std['val_acc']:.4f}, " +
                  f"F1={metrics['f1']:.4f}±{std['f1']:.4f}")


main()