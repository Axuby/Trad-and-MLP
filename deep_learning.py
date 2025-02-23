import os
import pickle
import time
import traceback
import zipfile

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import KNeighborsClassifier
from torch.optim.lr_scheduler import StepLR
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_recall_fscore_support, \
    accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from torch.utils.data import TensorDataset, DataLoader














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


import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-layer Perceptron for classification.

    Args:
        input_dim (int): Number of input features.
        output_dim (int): Number of classes.
        hidden_dim (int, optional): Size of hidden layers. Default is 128.
        activation (nn.Module, optional): Activation function class. Default is nn.ReLU.
        num_layers (int, optional): Number of hidden layers. Default is 2.
    """

    def __init__(self, input_dim, output_dim, hidden_dim=128, activation=nn.ReLU, num_layers=2):
        super(MLP, self).__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation())
        layers.append(nn.BatchNorm1d(hidden_dim))  # Normalize activations
        layers.append(nn.Dropout(0.3))  # Prevent overfitting

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.3))

        # Output layer (No activation function for logits)
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# def convert_features_to_loader(X, y, batch_size):
#     """Convert numpy arrays to PyTorch DataLoader."""
#     X_tensor = torch.FloatTensor(X)
#     y_tensor = torch.LongTensor(y.astype(int))
#     dataset = TensorDataset(X_tensor, y_tensor)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=True)


from torch.utils.data import DataLoader, TensorDataset
import torch


def convert_features_to_loader(train_feats, train_labels, test_feats, test_labels, batch_size):
    """
    Converts feature matrices and labels into PyTorch DataLoader objects.

    Args:
        train_feats: Training features (numpy array or tensor)
        train_labels: Training labels (numpy array or tensor)
        test_feats: Test features (numpy array or tensor)
        test_labels: Test labels (numpy array or tensor)
        batch_size: Mini-batch size for training

    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
    """
    # Convert NumPy arrays to PyTorch tensors (if needed)
    if isinstance(train_feats, np.ndarray):
        train_feats = torch.tensor(train_feats, dtype=torch.float32)
        train_labels = torch.tensor(train_labels, dtype=torch.long)

    if isinstance(test_feats, np.ndarray):
        test_feats = torch.tensor(test_feats, dtype=torch.float32)
        test_labels = torch.tensor(test_labels, dtype=torch.long)

    # Create TensorDatasets
    train_dataset = TensorDataset(train_feats, train_labels)
    test_dataset = TensorDataset(test_feats, test_labels)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def deep_learning(train_feats_proj, train_labels, test_feats_proj, test_labels,
                  input_dim, output_dim, hidden_dim=64, num_layers=3,
                  batch_size=32, learning_rate=0.001, epochs=100):
    """
    Enhanced deep learning implementation with comprehensive metrics tracking and visualization.

    Args:
        train_feats_proj: Training features
        train_labels: Training labels
        test_feats_proj: Test features
        test_labels: Test labels
        input_dim: Input dimension
        output_dim: Number of classes
        hidden_dim: Hidden layer dimension
        num_layers: Number of hidden layers
        batch_size: Batch size
        learning_rate: Learning rate
        epochs: Number of epochs
    """
    # Initialize model, criterion, optimizer
    model = MLP(input_dim, output_dim, hidden_dim, nn.ReLU, num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # Create data loaders
    train_loader, test_loader = convert_features_to_loader(
        train_feats_proj, train_labels, test_feats_proj, test_labels, batch_size
    )

    # Metrics tracking
    metrics = defaultdict(list)
    best_val_f1 = 0
    best_model_state = None

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_preds = []
        train_true = []

        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_true.extend(batch_labels.cpu().numpy())

        scheduler.step()

        # Calculate training metrics
        train_loss = train_loss / len(train_loader)
        train_f1 = f1_score(train_true, train_preds, average='weighted')
        train_acc = np.mean(np.array(train_preds) == np.array(train_true))

        # Evaluation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_true = []

        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_true.extend(batch_labels.cpu().numpy())

        # Calculate validation metrics
        val_loss = val_loss / len(test_loader)
        val_f1 = f1_score(val_true, val_preds, average='weighted')
        val_acc = np.mean(np.array(val_preds) == np.array(val_true))

        # Store metrics
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['train_f1'].append(train_f1)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        metrics['val_f1'].append(val_f1)

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

    # Load best model for final evaluation
    model.load_state_dict(best_model_state)

    # Final evaluation and metrics calculation
    model.eval()
    final_preds = []
    with torch.no_grad():
        for batch_data, _ in test_loader:
            outputs = model(batch_data)
            _, predicted = torch.max(outputs, 1)
            final_preds.extend(predicted.cpu().numpy())

    # Calculate final metrics
    final_metrics = {
        'accuracy': np.mean(np.array(final_preds) == test_labels),
        'f1_score': f1_score(test_labels, final_preds, average='weighted'),
        'std_acc': np.std(metrics['val_acc']),
        'std_f1': np.std(metrics['val_f1'])
    }

    # Plot training curves
    plot_training_curves_main(metrics)

    # Plot confusion matrix
    plot_confusion_matrix(test_labels, final_preds, output_dim)

    return model, final_metrics, metrics


def convert_features_to_loader(X, y, batch_size):
    """Convert numpy arrays to PyTorch DataLoader."""
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y.astype(int))
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

def plot_training_curves_main(metrics):
    """Plot training and validation metrics over time."""
    plt.figure(figsize=(15, 5))

    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['val_loss'], label='Val Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(metrics['train_acc'], label='Train Accuracy')
    plt.plot(metrics['val_acc'], label='Val Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # F1 Score plot
    plt.subplot(1, 3, 3)
    plt.plot(metrics['train_f1'], label='Train F1')
    plt.plot(metrics['val_f1'], label='Val F1')
    plt.title('F1 Score over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(true_labels, pred_labels, num_classes):
    """Plot confusion matrix with proper normalization."""
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(num_classes),
                yticklabels=range(num_classes))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()




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

    if model_type == 'traditional':
        clf = TraditionalClassifier()
        results = clf.evaluate(X_train_proc, y_train, X_val_proc, y_val, X_test_proc, y_test)

        train_acc = results['train_acc']
        val_acc = results['val_acc']
        test_acc = results['test_acc']
        val_pred = results['val_pred']
        test_pred = results['test_pred']
        metrics = results

    else:  # MLPmodel, final_metrics, metrics
        # Call the deep learning function with proper parameters
        model, final_metrics, training_metrics = deep_learning(
            train_feats_proj=X_train_proc,
            train_labels=y_train,
            test_feats_proj=X_test_proc,
            test_labels=y_test,
            input_dim=X_train_proc.shape[1],
            output_dim=len(np.unique(labels)),
            hidden_dim=64,
            num_layers=3,
            batch_size=32,
            learning_rate=0.001,
            epochs=100
        )

        # Extract metrics
        train_acc = final_metrics['train_accuracy']
        val_acc = final_metrics['val_accuracy']
        test_acc = final_metrics['accuracy']
        test_pred = final_metrics['predictions']
        val_pred = final_metrics['val_predictions']

        # Calculate additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_pred, average='weighted')

        metrics = {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'std_acc': final_metrics['std_acc'],
            'std_f1': final_metrics['std_f1']
        }

        # Store training metrics for visualization
        train_metrics = {
            'loss': training_metrics['train_loss'],
            'acc': training_metrics['train_acc'],
            'f1': training_metrics['train_f1']
        }

        val_metrics = {
            'loss': training_metrics['val_loss'],
            'acc': training_metrics['val_acc'],
            'f1': training_metrics['val_f1']
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
        'selected_features': selected_features,
        'training_history': training_metrics if model_type == 'mlp' else None
    }


def plot_visualizations(y_val, y_test, val_pred, test_pred, train_metrics, val_metrics,
                        subject_idx, model_name, dataset_type, selected_features, feature_scores):
    """Enhanced visualization function that plots all relevant metrics."""
    plt.figure(figsize=(20, 10))

    # Plot 1: Confusion Matrix
    plt.subplot(2, 3, 1)
    cm = confusion_matrix(y_test, test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (Subject {subject_idx})')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Plot 2: Training & Validation Accuracy
    plt.subplot(2, 3, 2)
    if train_metrics and 'acc' in train_metrics:
        plt.plot(train_metrics['acc'], label='Train')
        plt.plot(val_metrics['acc'], label='Validation')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

    # Plot 3: Training & Validation Loss
    plt.subplot(2, 3, 3)
    if train_metrics and 'loss' in train_metrics:
        plt.plot(train_metrics['loss'], label='Train')
        plt.plot(val_metrics['loss'], label='Validation')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

    # Plot 4: F1 Scores
    plt.subplot(2, 3, 4)
    if train_metrics and 'f1' in train_metrics:
        plt.plot(train_metrics['f1'], label='Train')
        plt.plot(val_metrics['f1'], label='Validation')
        plt.title('F1 Score over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()

    # Plot 5: Feature Importance
    plt.subplot(2, 3, 5)
    plt.bar(range(len(feature_scores)), feature_scores)
    plt.title('Feature Importance Scores')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance Score')

    plt.tight_layout()
    plt.savefig(f'results/metrics_subject_{subject_idx}_{model_name}_dataset_{dataset_type}.png')
    plt.close()


def classification(features, labels, subjects, dataset_type, model_type='traditional', use_lda=False):
    """
    Perform classification with optional LDA projection and LOSO cross-validation.
    Supports both traditional and deep learning approaches.

    Args:
        features: Input features array
        labels: Target labels array
        subjects: Subject identifiers for LOSO
        dataset_type: Dataset type identifier (100, 200, or 300)
        model_type: 'traditional' or 'mlp'
        use_lda: Whether to use Fisher LDA projection

    Returns:
        mean_metrics: Dictionary of mean performance metrics
        std_metrics: Dictionary of standard deviations for metrics
        aggregated_results: Dictionary containing additional analysis results
    """
    print(f"Starting classification for dataset {dataset_type} with {model_type}" +
          f" model and{'' if use_lda else ' no'} LDA")
    total_start_time = time.time()

    # Get unique subjects
    unique_subjects = np.unique(subjects)

    # Lists to hold results
    subject_results = []
    all_predictions = []
    all_true_labels = []
    training_histories = []

    # Process each subject independently
    for subject_idx in unique_subjects:
        result = process_subject(
            subject_idx, features, labels, subjects,
            dataset_type, model_type, use_lda
        )
        subject_results.append(result)

        # Store predictions and true labels for aggregate analysis
        metrics = result['metrics']
        all_predictions.extend(metrics.get('test_predictions', []))
        all_true_labels.extend(metrics.get('test_true_labels', []))

        # Store training history for MLP models
        if model_type == 'mlp' and result.get('training_history'):
            training_histories.append(result['training_history'])

    # Compile results
    metrics_list = [r['metrics'] for r in subject_results]

    # Calculate overall metrics
    mean_metrics = {k: np.mean([r[k] for r in metrics_list]) for k in metrics_list[0].keys()}
    std_metrics = {k: np.std([r[k] for r in metrics_list]) for k in metrics_list[0].keys()}

    # Compile feature importance statistics
    all_feature_importance = np.zeros((len(unique_subjects), features.shape[1]))
    for i, result in enumerate(subject_results):
        selected_features = result['selected_features']
        feature_scores = result['feature_importance']
        all_feature_importance[i, selected_features] = feature_scores

    # Calculate feature importance statistics
    feature_importance_mean = np.mean(all_feature_importance, axis=0)
    feature_importance_std = np.std(all_feature_importance, axis=0)

    # Save feature importance statistics
    np.save(f'results/feature_importance_mean_dataset_{dataset_type}.npy', feature_importance_mean)
    np.save(f'results/feature_importance_std_dataset_{dataset_type}.npy', feature_importance_std)

    # Calculate aggregate confusion matrix
    if all_predictions and all_true_labels:
        aggregate_cm = confusion_matrix(all_true_labels, all_predictions)
        plot_aggregate_confusion_matrix(
            aggregate_cm,
            f"{model_type}{'_lda' if use_lda else ''}",
            dataset_type
        )

    # For MLP models, plot aggregate training histories
    if model_type == 'mlp' and training_histories:
        plot_aggregate_training_curves(
            training_histories,
            f"{model_type}{'_lda' if use_lda else ''}",
            dataset_type
        )

    # Calculate additional aggregate metrics
    aggregated_results = {
        'feature_importance_mean': feature_importance_mean,
        'feature_importance_std': feature_importance_std,
        'processing_time': time.time() - total_start_time,
        'subject_count': len(unique_subjects),
        'model_type': model_type,
        'dataset_type': dataset_type,
        'use_lda': use_lda
    }

    # Print comprehensive results summary
    print_results_summary(mean_metrics, std_metrics, aggregated_results)

    return mean_metrics, std_metrics, aggregated_results


def plot_aggregate_confusion_matrix(cm, model_name, dataset_type):
    """Plot and save aggregate confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Aggregate Confusion Matrix\n{model_name} - Dataset {dataset_type}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'results/aggregate_cm_{model_name}_dataset_{dataset_type}.png')
    plt.close()


def plot_aggregate_training_curves(training_histories, model_name, dataset_type):
    """Plot and save aggregate training curves with confidence intervals."""
    metrics = ['loss', 'acc', 'f1']
    plt.figure(figsize=(15, 5))

    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)

        # Extract training and validation metrics
        train_metric = np.array([history[f'train_{metric}'] for history in training_histories])
        val_metric = np.array([history[f'val_{metric}'] for history in training_histories])

        # Calculate mean and std
        train_mean = np.mean(train_metric, axis=0)
        train_std = np.std(train_metric, axis=0)
        val_mean = np.mean(val_metric, axis=0)
        val_std = np.std(val_metric, axis=0)

        # Plot with confidence intervals
        epochs = range(1, len(train_mean) + 1)
        plt.plot(epochs, train_mean, label='Train')
        plt.fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.1)
        plt.plot(epochs, val_mean, label='Validation')
        plt.fill_between(epochs, val_mean - val_std, val_mean + val_std, alpha=0.1)

        plt.title(f'{metric.upper()} over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric.upper())
        plt.legend()

    plt.tight_layout()
    plt.savefig(f'results/aggregate_training_curves_{model_name}_dataset_{dataset_type}.png')
    plt.close()


def print_results_summary(mean_metrics, std_metrics, aggregated_results):
    """Print comprehensive results summary."""
    print("\n" + "=" * 50)
    print(f"Classification Results Summary")
    print("=" * 50)

    print("\nModel Configuration:")
    print(f"Model Type: {aggregated_results['model_type']}")
    print(f"Dataset Type: {aggregated_results['dataset_type']}")
    print(f"LDA Used: {aggregated_results['use_lda']}")
    print(f"Number of Subjects: {aggregated_results['subject_count']}")

    print("\nPerformance Metrics:")
    for metric in mean_metrics.keys():
        print(f"{metric}: {mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")

    print(f"\nTotal Processing Time: {aggregated_results['processing_time']:.2f} seconds")
    print("=" * 50)


def main():
    """
    Run the complete classification pipeline with all configurations.
    Includes comprehensive result tracking, visualization, and analysis.
    """
    # Configuration
    dataset_types = [100, 200, 300]
    all_results = {}
    experiment_timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Create results directory structure
    results_dir = create_results_directory(experiment_timestamp)

    # Initialize result tracking
    performance_summary = {
        'dataset': [],
        'model': [],
        'accuracy': [],
        'accuracy_std': [],
        'f1_score': [],
        'f1_std': [],
        'training_time': []
    }

    try:
        for dataset_type in dataset_types:
            print(f"\n{'=' * 20} Processing Dataset {dataset_type} {'=' * 20}")

            # Load and validate dataset
            features, labels, subjects = load_dataset(verbose=True, dataset_type=dataset_type)
            if features is None:
                print(f"Skipping dataset {dataset_type} due to loading error")
                continue

            # Print dataset statistics
            print_dataset_statistics(features, labels, subjects)

            results = {}

            for model_type in ['mlp']:
                for use_lda in [True, False]:
                    key = f"{model_type}{'_lda' if use_lda else ''}"
                    print(f"\n{'-' * 10} Running {key.upper()} on dataset {dataset_type} {'-' * 10}")

                    try:
                        # Run classification
                        start_time = time.time()
                        mean_metrics, std_metrics, aggregated_results = classification(
                            features, labels, subjects,
                            dataset_type=dataset_type,
                            model_type=model_type,
                            use_lda=use_lda
                        )
                        training_time = time.time() - start_time

                        # Store results
                        results[key] = (mean_metrics, std_metrics, aggregated_results)

                        # Update performance summary
                        update_performance_summary(
                            performance_summary,
                            dataset_type,
                            key,
                            mean_metrics,
                            std_metrics,
                            training_time
                        )

                        # Save detailed results
                        save_detailed_results(
                            results_dir,
                            dataset_type,
                            key,
                            mean_metrics,
                            std_metrics,
                            aggregated_results
                        )

                    except Exception as e:
                        print(f"Error processing {key} on dataset {dataset_type}: {str(e)}")
                        traceback.print_exc()
                        continue

            all_results[dataset_type] = results

            # Print current dataset results
            print_dataset_results(dataset_type, results)

        # Generate and save comparative visualizations
        generate_comparative_visualizations(
            all_results,
            dataset_types,
            results_dir
        )

        # Generate and save summary report
        generate_summary_report(
            all_results,
            dataset_types,
            performance_summary,
            results_dir
        )

        # Create and save results archive
        create_results_archive(results_dir)

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        traceback.print_exc()

    finally:
        # Ensure results are saved even if there's an error
        save_final_results(all_results, results_dir)


def create_results_directory(timestamp):
    """Create organized directory structure for results."""
    base_dir = f"results_{timestamp}"
    subdirs = ['metrics', 'plots', 'models', 'logs']

    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)

    return base_dir


def print_dataset_statistics(features, labels, subjects):
    """Print detailed dataset statistics."""
    print("\nDataset Statistics:")
    print(f"Number of samples: {len(features)}")
    print(f"Number of features: {features.shape[1]}")
    print(f"Number of classes: {len(np.unique(labels))}")
    print(f"Number of subjects: {len(np.unique(subjects))}")
    # print(f"Class distribution: {np.bincount(labels)}")


def update_performance_summary(summary, dataset_type, model_name, metrics, std_metrics, training_time):
    """Update the performance summary dictionary."""
    summary['dataset'].append(dataset_type)
    summary['model'].append(model_name)
    summary['accuracy'].append(metrics['test_acc'])
    summary['accuracy_std'].append(std_metrics['test_acc'])
    summary['f1_score'].append(metrics['f1'])
    summary['f1_std'].append(std_metrics['f1'])
    summary['training_time'].append(training_time)


def generate_comparative_visualizations(all_results, dataset_types, results_dir):
    """Generate comprehensive comparative visualizations."""
    plt.style.use('seaborn')

    # Accuracy comparison across datasets
    plt.figure(figsize=(12, 6))
    plot_metric_comparison(all_results, dataset_types, 'test_acc', 'Accuracy')
    plt.savefig(os.path.join(results_dir, 'plots', 'accuracy_comparison.png'))
    plt.close()

    # F1 score comparison
    plt.figure(figsize=(12, 6))
    plot_metric_comparison(all_results, dataset_types, 'f1', 'F1 Score')
    plt.savefig(os.path.join(results_dir, 'plots', 'f1_comparison.png'))
    plt.close()

    # Learning curves comparison
    plot_learning_curves_comparison(all_results, dataset_types, results_dir)


def generate_summary_report(all_results, dataset_types, performance_summary, results_dir):
    """Generate and save comprehensive summary report."""
    report = ["=" * 50 + "\nCLASSIFICATION SUMMARY REPORT\n" + "=" * 50 + "\n\n"]

    # Overall performance summary
    report.append("OVERALL PERFORMANCE SUMMARY:\n")
    for dataset_type in dataset_types:
        if dataset_type in all_results:
            report.append(f"\nDataset {dataset_type}:")
            for model_name, (metrics, std, _) in all_results[dataset_type].items():
                report.append(
                    f"{model_name}: "
                    f"Acc={metrics['test_acc']:.4f}±{std['test_acc']:.4f}, "
                    f"F1={metrics['f1']:.4f}±{std['f1']:.4f}"
                )

    # Save report
    with open(os.path.join(results_dir, 'summary_report.txt'), 'w') as f:
        f.write('\n'.join(report))


def create_results_archive(results_dir):
    """Create and save results archive."""
    archive_name = f"{results_dir}.zip"
    with zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(results_dir):
            for file in files:
                zipf.write(os.path.join(root, file))

    try:
        from google.colab import files
        files.download(archive_name)
    except ImportError:
        print(f"Results archived in {archive_name}")


def save_final_results(all_results, results_dir):
    """Save final results to disk."""
    results_file = os.path.join(results_dir, 'all_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(all_results, f)


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Tuple




def print_dataset_results(dataset_type: int, results: Dict):
    """
    Print detailed results for a specific dataset.

    Args:
        dataset_type: Dataset identifier (100, 200, or 300)
        results: Dictionary containing model results
    """
    print(f"\n{'=' * 20} Results for Dataset {dataset_type} {'=' * 20}")

    for model_name, (metrics, std_metrics, aggregated_results) in results.items():
        print(f"\n{model_name.upper()}:")
        print("-" * 40)

        # Print accuracy metrics
        print("Accuracy Metrics:")
        print(f"Test Accuracy:  {metrics['test_acc']:.4f} ± {std_metrics['test_acc']:.4f}")
        print(f"Val Accuracy:   {metrics['val_acc']:.4f} ± {std_metrics['val_acc']:.4f}")
        print(f"Train Accuracy: {metrics['train_acc']:.4f} ± {std_metrics['train_acc']:.4f}")

        # Print F1 scores
        print("\nF1 Scores:")
        print(f"F1 Score: {metrics['f1']:.4f} ± {std_metrics['f1']:.4f}")

        # Print additional metrics if available
        print("\nAdditional Metrics:")
        print(f"Precision: {metrics.get('precision', 0):.4f} ± {std_metrics.get('precision', 0):.4f}")
        print(f"Recall:    {metrics.get('recall', 0):.4f} ± {std_metrics.get('recall', 0):.4f}")

        # Print processing information
        print(f"\nProcessing Time: {aggregated_results['processing_time']:.2f} seconds")
        print(f"Number of Subjects: {aggregated_results['subject_count']}")


def plot_learning_curves_comparison(all_results: Dict, dataset_types: List[int], results_dir: str):
    """
    Plot learning curves comparison across different datasets.

    Args:
        all_results: Dictionary containing results for all datasets
        dataset_types: List of dataset identifiers
        results_dir: Directory to save the plots
    """
    plt.figure(figsize=(15, 10))

    # Create subplots for different metrics
    metrics = ['loss', 'acc', 'f1']

    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)

        for dataset_type in dataset_types:
            if dataset_type not in all_results:
                continue

            for model_name, (_, _, aggregated_results) in all_results[dataset_type].items():
                if 'training_history' not in aggregated_results:
                    continue

                history = aggregated_results['training_history']
                epochs = range(1, len(history[f'train_{metric}']) + 1)

                # Plot training curve
                plt.plot(epochs, history[f'train_{metric}'],
                         label=f'Dataset {dataset_type} - Train')
                # Plot validation curve
                plt.plot(epochs, history[f'val_{metric}'],
                         label=f'Dataset {dataset_type} - Val',
                         linestyle='--')

        plt.title(f'{metric.upper()} Learning Curves')
        plt.xlabel('Epoch')
        plt.ylabel(metric.title())
        plt.legend()
        plt.grid(True)

    # Add overall title
    plt.suptitle('Learning Curves Comparison Across Datasets', fontsize=16)
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'{results_dir}/plots/learning_curves_comparison.png')
    plt.close()


def plot_metric_comparison(all_results: Dict, dataset_types: List[int],
                           metric_name: str, metric_label: str):
    """
    Plot comparison of a specific metric across datasets and models.

    Args:
        all_results: Dictionary containing results for all datasets
        dataset_types: List of dataset identifiers
        metric_name: Name of the metric to plot
        metric_label: Label for the metric in the plot
    """
    plt.figure(figsize=(12, 6))

    # Prepare data for plotting
    datasets = []
    models = []
    values = []
    errors = []

    for dataset_type in dataset_types:
        if dataset_type not in all_results:
            continue

        for model_name, (metrics, std_metrics, _) in all_results[dataset_type].items():
            datasets.append(f'Dataset {dataset_type}')
            models.append(model_name)
            values.append(metrics[metric_name])
            errors.append(std_metrics[metric_name])

    # Create bar plot
    x = np.arange(len(datasets))
    width = 0.35

    plt.bar(x, values, width, yerr=errors, capsize=5,
            label=[m.upper() for m in models])

    plt.xlabel('Dataset')
    plt.ylabel(metric_label)
    plt.title(f'{metric_label} Comparison Across Datasets')
    plt.xticks(x, datasets)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add value labels on top of bars
    for i, v in enumerate(values):
        plt.text(i, v + errors[i], f'{v:.3f}',
                 ha='center', va='bottom')

    plt.tight_layout()


# Additional utility function for creating a custom color palette
def get_color_palette(n_colors: int) -> List[str]:
    """
    Create a custom color palette for consistent plotting.

    Args:
        n_colors: Number of colors needed

    Returns:
        List of color hex codes
    """
    return sns.color_palette("husl", n_colors=n_colors).as_hex()


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd


def save_detailed_results(results_dir, dataset_type, key, mean_metrics, std_metrics, aggregated_results):
    """
    Save detailed results including metrics, plots, and analysis.

    Args:
        results_dir: Base directory for saving results
        dataset_type: Type of dataset (100, 200, 300)
        key: Model configuration key
        mean_metrics: Dictionary of mean performance metrics
        std_metrics: Dictionary of standard deviation metrics
        aggregated_results: Dictionary of additional results and analysis
    """
    # Create subdirectory for this specific configuration
    config_dir = os.path.join(results_dir, f'dataset_{dataset_type}', key)
    os.makedirs(config_dir, exist_ok=True)

    # Save metrics as JSON
    metrics_data = {
        'mean_metrics': mean_metrics,
        'std_metrics': std_metrics,
        'aggregated_results': {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in aggregated_results.items()
        }
    }

    with open(os.path.join(config_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_data, f, indent=4)

    # Save feature importance plot
    # if 'feature_importance_mean' in aggregated_results:
    #     plot_feature_importance(
    #         aggregated_results['feature_importance_mean'],
    #         aggregated_results['feature_importance_std'],
    #         os.path.join(config_dir, 'feature_importance.png')
    #     )
    #
    # # Save learning curves if available
    # if 'training_history' in aggregated_results:
    #     plot_learning_curves(
    #         aggregated_results['training_history'],
    #         os.path.join(config_dir, 'learning_curves.png')
    #     )

    # Generate and save detailed report
    generate_detailed_report(
        dataset_type,
        key,
        mean_metrics,
        std_metrics,
        aggregated_results,
        os.path.join(config_dir, 'detailed_report.txt')
    )


def plot_visualizations(y_val, y_test, val_pred, test_pred, train_metrics, val_metrics,
                        subject_idx, model_name, dataset_type, selected_features, feature_scores):
    """
    Create comprehensive visualizations for model performance and analysis.

    Args:
        y_val: Validation true labels
        y_test: Test true labels
        val_pred: Validation predictions
        test_pred: Test predictions
        train_metrics: Dictionary of training metrics
        val_metrics: Dictionary of validation metrics
        subject_idx: Subject identifier
        model_name: Name of the model
        dataset_type: Type of dataset
        selected_features: Selected feature indices
        feature_scores: Feature importance scores
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))

    # 1. Confusion Matrix
    plt.subplot(3, 2, 1)
    plot_confusion_matrix_heat(y_test, test_pred, "Test Set Confusion Matrix")

    # 2. Training Curves
    plt.subplot(3, 2, 2)
    plot_training_curves(train_metrics, val_metrics)

    # 3. Feature Importance
    plt.subplot(3, 2, 3)
    plot_feature_importance_subset(feature_scores, selected_features)

    # 4. Class Distribution
    plt.subplot(3, 2, 4)
    plot_class_distribution(y_test, test_pred)

    # 5. Performance Metrics Over Time
    plt.subplot(3, 2, 5)
    plot_metrics_over_time(train_metrics, val_metrics)

    # 6. Error Analysis
    plt.subplot(3, 2, 6)
    plot_error_analysis(y_test, test_pred)

    # Add overall title and adjust layout
    plt.suptitle(f'Model Analysis - Subject {subject_idx} - {model_name} - Dataset {dataset_type}',
                 fontsize=16, y=1.02)
    plt.tight_layout()

    # Save the figure
    save_path = f'results/visualizations/subject_{subject_idx}_{model_name}_dataset_{dataset_type}.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_confusion_matrix_heat(y_true, y_pred, title):
    """Plot confusion matrix as heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')


def plot_training_curves(train_metrics, val_metrics):
    """Plot training and validation metrics over time."""
    epochs = range(1, len(train_metrics['loss']) + 1)

    plt.plot(epochs, train_metrics['loss'], 'b-', label='Training Loss')
    plt.plot(epochs, val_metrics['loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()


def plot_feature_importance_subset(feature_scores, selected_features):
    """Plot feature importance scores for selected features."""
    plt.bar(range(len(selected_features)), feature_scores)
    plt.title('Feature Importance Scores')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance Score')
    plt.xticks(range(len(selected_features)), selected_features, rotation=45)


def plot_class_distribution(y_true, y_pred):
    """Plot true vs predicted class distribution."""
    true_dist = pd.Series(y_true).value_counts()
    pred_dist = pd.Series(y_pred).value_counts()

    width = 0.35
    plt.bar(np.arange(len(true_dist)), true_dist, width, label='True')
    plt.bar(np.arange(len(pred_dist)) + width, pred_dist, width, label='Predicted')

    plt.title('Class Distribution Comparison')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.legend()


def plot_metrics_over_time(train_metrics, val_metrics):
    """Plot multiple metrics over training time."""
    epochs = range(1, len(train_metrics['acc']) + 1)

    plt.plot(epochs, train_metrics['acc'], 'b-', label='Train Acc')
    plt.plot(epochs, train_metrics['f1'], 'g-', label='Train F1')
    plt.plot(epochs, val_metrics['acc'], 'r-', label='Val Acc')
    plt.plot(epochs, val_metrics['f1'], 'm-', label='Val F1')

    plt.title('Performance Metrics Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()


def plot_error_analysis(y_true, y_pred):
    """Plot error analysis showing misclassification patterns."""
    error_mask = y_true != y_pred
    error_true = y_true[error_mask]
    error_pred = y_pred[error_mask]

    error_pairs = pd.DataFrame({'True': error_true, 'Predicted': error_pred})
    error_counts = error_pairs.groupby(['True', 'Predicted']).size().unstack(fill_value=0)

    sns.heatmap(error_counts, annot=True, fmt='d', cmap='Reds')
    plt.title('Misclassification Patterns')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')


def generate_detailed_report(dataset_type, model_name, mean_metrics, std_metrics,
                             aggregated_results, save_path):
    """Generate detailed performance report."""
    report = [
        f"Detailed Analysis Report",
        f"=====================",
        f"\nConfiguration:",
        f"- Dataset Type: {dataset_type}",
        f"- Model: {model_name}",
        f"\nPerformance Metrics:",
    ]

    # Add main metrics
    for metric, value in mean_metrics.items():
        report.append(f"- {metric}: {value:.4f} ± {std_metrics[metric]:.4f}")

    # Add processing information
    report.extend([
        f"\nProcessing Information:",
        f"- Total Processing Time: {aggregated_results.get('processing_time', 'N/A')} seconds",
        f"- Number of Features: {len(aggregated_results.get('feature_importance_mean', []))}",
    ])

    # Save report
    with open(save_path, 'w') as f:
        f.write('\n'.join(report))


main()