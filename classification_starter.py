'''
Name : Azubuine Samuel Tochukwu
PSU ID: 960826967
EE552 Project 2 Intermediate Submission
}

Process Flow and function calls:

The Main function ===>> Classification ===> Process_subjects( for each subject of LOSO dataset split) ===> Picks model_type function to run
 Then either the TraditionalClassifier or the deep_learning function executes

 Then Plot_Visualizations to plot identified metrics of performance and others including confusion matrix,
 class distribution, mis-classification rate, feature importance and  etc.
'''
import time
import traceback
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader
from model_starter import TraditionalClassifier, MLP
from utilities import  sequential_forward_selection
from helpers import create_results_archive, generate_summary_report, print_dataset_results, \
    generate_comparative_visualizations, save_detailed_results, update_performance_summary, print_dataset_statistics, \
    create_results_directory, plot_training_curves_main,  save_final_results, \
    plot_aggregate_confusion_matrix, plot_aggregate_training_curves, plot_confusion_matrix
from utils import plot_visualizations









def deep_learning(train_feats_proj, train_labels, test_feats_proj, test_labels,
                  input_dim, output_dim, hidden_dim=64, num_layers=3,
                  batch_size=32, learning_rate=0.001, epochs=100):
    """
    I updated this deep learning implementation with comprehensive metrics tracking and visualization.

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
    batch_size = max(2, min(batch_size, len(train_feats_proj)))
    print(f"Using batch size: {batch_size}")

    # Initialize model, criterion, optimizer
    model = MLP(input_dim, output_dim, hidden_dim, nn.ReLU, num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # Adjust BatchNorm behavior to handle small batch sizes
    for layer in model.modules():
        if isinstance(layer, torch.nn.BatchNorm1d):
            layer.track_running_stats = batch_size > 1


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

        # Calculates training metrics
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

        # calculates validation metrics
        val_loss = val_loss / len(test_loader)
        val_f1 = f1_score(val_true, val_preds, average='weighted')
        val_acc = np.mean(np.array(val_preds) == np.array(val_true))


        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['train_f1'].append(train_f1)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        metrics['val_f1'].append(val_f1)

        # Save best model state
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

    # load best model for last or final evaluation
    model.load_state_dict(best_model_state)

    # last evaluation and metrics calculation
    model.eval()
    final_preds = []
    with torch.no_grad():
        for batch_data, _ in test_loader:
            outputs = model(batch_data)
            _, predicted = torch.max(outputs, 1)
            final_preds.extend(predicted.cpu().numpy())


    final_metrics = {
        'train_accuracy': np.mean(metrics['train_acc']),
        'val_accuracy': np.mean(metrics['val_acc']),
        'accuracy': np.mean(np.array(final_preds) == test_labels),  # test accuracy
        'f1_score': f1_score(test_labels, final_preds, average='weighted'),
        'std_acc': np.std(metrics['val_acc']),
        'std_f1': np.std(metrics['val_f1']),
        'predictions': final_preds,  # test predictions
        'val_predictions': val_preds  #validation predictions
    }

    plot_training_curves_main(metrics)

    plot_confusion_matrix(test_labels, final_preds, output_dim)

    return model, final_metrics, metrics


def convert_features_to_loader(train_feats, train_labels, test_feats, test_labels, batch_size):
    """
    Converts feature matrices and labels into PyTorch DataLoader objects.

    params/Args:
        train_feats: Training features (numpy array or tensor)
        train_labels: Training labels (numpy array or tensor)
        test_feats: Test features (numpy array or tensor)
        test_labels: Test labels (numpy array or tensor)
        batch_size: Mini-batch size for training

    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
    """
    # First convert NumPy arrays to PyTorch tensors (if needed)
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







def load_dataset(verbose=True, dataset_type=100):
    """
    Load dataset using LOSO LeaveOneGroupOut cross-validation
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


def apply_feature_processing(X_train, X_val, X_test, y_train, use_lda=False, use_pca=False, pca_components=0.95):
    """
    Apply feature selection and optionally LDA or PCA projection.

    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        y_train: Training labels
        use_lda: Whether to use Fisher LDA for dimensionality reduction
        use_pca: Whether to use PCA for dimensionality reduction
        pca_components: Number of components or variance ratio to keep for PCA

    Returns:
        Processed feature matrices and feature selection information
    """
    print("Starting feature selection...")
    start_time = time.time()

    # Feature selection with SFS as recommended in class/assignment
    selected_features, feature_scores = sequential_forward_selection(X_train, y_train)

    print(f"Feature selection completed in {time.time() - start_time:.2f} seconds")

    # Apply selection
    X_train_selected = X_train[:, selected_features]
    X_val_selected = X_val[:, selected_features]
    X_test_selected = X_test[:, selected_features]

    # Apply dimensionality reduction if requested
    if use_lda:
        print("Applying Fisher LDA projection...")
        projection_matrix, explained_variance = fisher_projection(X_train_selected, y_train)
        if projection_matrix is not None:
            X_train_final = X_train_selected @ projection_matrix
            X_val_final = X_val_selected @ projection_matrix
            X_test_final = X_test_selected @ projection_matrix
            print(f"LDA reduced dimensions from {X_train_selected.shape[1]} to {X_train_final.shape[1]}")
            if explained_variance is not None:
                print(f"Explained variance ratio: {np.sum(explained_variance):.4f}")
        else:
            print("Fisher LDA failed, using selected features without projection")
            X_train_final = X_train_selected
            X_val_final = X_val_selected
            X_test_final = X_test_selected

    elif use_pca:
        print(f"Applying PCA with {pca_components} variance retention...")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_val_scaled = scaler.transform(X_val_selected)
        X_test_scaled = scaler.transform(X_test_selected)

        # Applying PCA
        pca = PCA(n_components=pca_components)
        pca.fit(X_train_scaled)

        X_train_final = pca.transform(X_train_scaled)
        X_val_final = pca.transform(X_val_scaled)
        X_test_final = pca.transform(X_test_scaled)

        print(f"PCA reduced dimensions from {X_train_selected.shape[1]} to {X_train_final.shape[1]}")
        print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")

    else:
        # No dimensionality reduction
        X_train_final = X_train_selected
        X_val_final = X_val_selected
        X_test_final = X_test_selected

    return X_train_final, X_val_final, X_test_final, selected_features, feature_scores






def fisher_projection(X: np.ndarray, y: np.ndarray, n_components: int = None) -> tuple:
    """
    My implementation of the Fisher Linear Discriminant Analysis for dimensionality reduction.

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

        # Sorting eigenvalues and eigenvectors
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


def process_subject(subject_idx, features, labels, subjects, dataset_type, model_type, use_lda, use_pca=False):
    """Process a single subject for leave-one-subject-out validation."""
    print(f"Processing Subject {subject_idx} with {model_type} model" +
          f" and{'' if use_lda else ' no'} LDA" +
          f"{' with PCA' if use_pca else ''} (Dataset {dataset_type})")
    start_time = time.time()

    #LOSO: test data is from subject_idx, rest is split into train/val
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
        apply_feature_processing(X_train, X_val, X_test, y_train, use_lda, use_pca)

    # Initialize empty metrics
    metrics = {}
    train_metrics, val_metrics = {'loss': [], 'acc': [], 'f1': []}, {'loss': [], 'acc': [], 'f1': []}

    if model_type == 'traditional':
        clf = TraditionalClassifier()
        results = clf.evaluate(X_train_proc, y_train, X_val_proc, y_val, X_test_proc, y_test)

        train_acc, val_acc, test_acc = results['train_acc'], results['val_acc'], results['test_acc']
        val_pred, test_pred = results['val_pred'], results['test_pred']

        # Populate metrics
        metrics.update(results)

    else:  # MLP
        # Calls the deep learning function with proper parameters
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

        train_acc, val_acc, test_acc = final_metrics['train_accuracy'], final_metrics['val_accuracy'], final_metrics['accuracy']
        test_pred, val_pred = final_metrics['predictions'], final_metrics['val_predictions']

        precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_pred, average='weighted')

        metrics.update({
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'std_acc': final_metrics.get('std_acc', np.nan),
            'std_f1': final_metrics.get('std_f1', np.nan)
        })

        train_metrics.update({
            'loss': training_metrics['train_loss'],
            'acc': training_metrics['train_acc'],
            'f1': training_metrics['train_f1']
        })

        val_metrics.update({
            'loss': training_metrics['val_loss'],
            'acc': training_metrics['val_acc'],
            'f1': training_metrics['val_f1']
        })

    # Ensuring f1 exists to avoid KeyError
    metrics.setdefault('f1', np.nan)

    # Plot visualizations
    plot_visualizations(y_val, y_test, val_pred, test_pred, train_metrics, val_metrics,
                        subject_idx, f"{model_type}{'_lda' if use_lda else ''}", dataset_type, selected_features, feature_scores)

    print(f"Subject {subject_idx} processing completed in {time.time() - start_time:.2f} seconds")

    return {
        'subject_idx': subject_idx,
        'metrics': metrics,
        'feature_importance': feature_scores,
        'selected_features': selected_features,
        'training_history': training_metrics if model_type == 'mlp' else None
    }


def classification(features, labels, subjects, dataset_type, model_type='traditional', use_lda=False, use_pca=True):
    """
    Perform classification with optional LDA/PCA projection and LOSO cross-validation.
    Supports both traditional and deep learning approaches.

    Args:
        features: Input features array
        labels: Target labels array
        subjects: Subject identifiers for LOSO
        dataset_type: Dataset type identifier (100, 200, or 300)
        model_type: 'traditional' or 'mlp'
        use_lda: Whether to use Fisher LDA projection
        use_pca: Whether to use PCA projection (ignored if use_lda is True)

    Returns:
        mean_metrics: Dictionary of mean performance metrics
        std_metrics: Dictionary of standard deviations for metrics
        aggregated_results: Dictionary containing additional analysis results
    """
    if use_lda and use_pca:
        print("Both LDA and PCA requested. Using LDA and ignoring PCA. "
              "Set LDA(use_lda) to False to use only PCA(use_lda")
        use_pca = False

    print(f"Starting classification for dataset {dataset_type} with {model_type}" +
          f" model and{'' if use_lda else ' no'} LDA" +
          f"{' with PCA' if use_pca else ''}")

    total_start_time = time.time()
    unique_subjects = np.unique(subjects)

    subject_results = []
    all_predictions = []
    all_true_labels = []
    training_histories = []

    for subject_idx in unique_subjects:
        result = process_subject(
            subject_idx, features, labels, subjects,
            dataset_type, model_type, use_lda, use_pca
        )
        subject_results.append(result)

        # Store predictions and true labels for aggregate analysis
        metrics = result['metrics']
        all_predictions.extend(metrics.get('test_predictions', []))
        all_true_labels.extend(metrics.get('test_true_labels', []))


        if model_type == 'mlp' and result.get('training_history'):
            training_histories.append(result['training_history'])

    # Collect all metrics for averaging
    metrics_list = [r['metrics'] for r in subject_results]

    # Ensure 'f1' is present in all dictionaries to avoid KeyError
    for metrics in metrics_list:
        metrics.setdefault('f1', np.nan)

    # Handle potential inconsistencies in metric shapes
    mean_metrics = {}
    std_metrics = {}

    for k in metrics_list[0].keys():
        values = [r[k] for r in metrics_list if k in r]  # Ensuring the key exists in all results

        # Check if values are lists/arrays (MLP case)
        if isinstance(values[0], (list, np.ndarray)):
            values = [np.mean(v) if isinstance(v, (list, np.ndarray)) else v for v in values]

        # Compute mean and std deviation
        mean_metrics[k] = np.mean(values)
        std_metrics[k] = np.std(values)

    # Ensure 'f1' is included in final output or set to nan
    mean_metrics.setdefault('f1', np.nan)
    std_metrics.setdefault('f1', np.nan)

    # Compute feature importance
    all_feature_importance = np.zeros((len(unique_subjects), features.shape[1]))
    for i, result in enumerate(subject_results):
        selected_features = result['selected_features']
        feature_scores = result['feature_importance']
        all_feature_importance[i, selected_features] = feature_scores

    feature_importance_mean = np.mean(all_feature_importance, axis=0)
    feature_importance_std = np.std(all_feature_importance, axis=0)

    # Save feature importance statistics
    np.save(f'results/feature_importance_mean_dataset_{dataset_type}.npy', feature_importance_mean)
    np.save(f'results/feature_importance_std_dataset_{dataset_type}.npy', feature_importance_std)

    # Aggregate confusion matrix
    if all_predictions and all_true_labels:
        aggregate_cm = confusion_matrix(all_true_labels, all_predictions)
        plot_aggregate_confusion_matrix(
            aggregate_cm,
            f"{model_type}{'_lda' if use_lda else ''}",
            dataset_type
        )

    # for the MLP model, plot aggregate training histories
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

    # Print results summary
    print_results_summary(mean_metrics, std_metrics, aggregated_results)

    return mean_metrics, std_metrics, aggregated_results





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
    # Configuration for iteration to loop through the taiji data while
    # varying the last digits of the csv filename
    dataset_types = [100, 200, 300]
    all_results = {}
    experiment_timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Creates results directory structure
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

            # loads and validate dataset
            features, labels, subjects = load_dataset(verbose=True, dataset_type=dataset_type)
            if features is None:
                print(f"Skipping dataset {dataset_type} due to loading error")
                continue

            # print some of the dataset info
            print_dataset_statistics(features, labels, subjects)

            results = {}

            for model_type in ['mlp', 'traditional']:
                for use_lda in [True, False]: # there's a new param in the classification func for use_pca, which is True by default
                    key = f"{model_type}{'_lda' if use_lda else ''}"
                    print(f"\n{'-' * 10} Running {key.upper()} on dataset {dataset_type} {'-' * 10}")

                    try:
                        # runs th classification func(monitor it )
                        start_time = time.time()
                        mean_metrics, std_metrics, aggregated_results = classification(
                            features, labels, subjects,
                            dataset_type=dataset_type,
                            model_type=model_type,
                            use_lda=use_lda,
                            use_pca=True #last requirements for more marks
                        )
                        training_time = time.time() - start_time

                        results[key] = (mean_metrics, std_metrics, aggregated_results)

                        # Updating performance summary
                        update_performance_summary(
                            performance_summary,
                            dataset_type,
                            key,
                            mean_metrics,
                            std_metrics,
                            training_time
                        )

                        # Saves detailed results
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
            print_dataset_results(dataset_type, results)


        generate_comparative_visualizations(
            all_results,
            dataset_types,
            results_dir
        )


        generate_summary_report(
            all_results,
            dataset_types,
            performance_summary,
            results_dir
        )


        create_results_archive(results_dir)

    except Exception as e:
        print(f" Encountered an Error in main execution: {str(e)}")
        traceback.print_exc()

    finally:
        save_final_results(all_results, results_dir)

    # Zip results for my tests on google colab
    # os.system("zip -r results.zip results/")
    # from google.colab import files
    # files.download("results.zip")


main()






