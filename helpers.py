import os
import pickle
import time
import traceback
import zipfile
import json
import os
import pandas as pd
from typing import Dict, List

import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



def plot_visualizations(y_val, y_test, val_pred, test_pred, train_metrics, val_metrics,
                        subject_idx, model_name, dataset_type, selected_features, feature_scores):
    """
    Creates comprehensive visualizations for model performance and analysis.
    where y_val: Validation true labels
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
    # Create these figure with subplots
    fig = plt.figure(figsize=(20, 15))

    # plots Confusion Matrix
    plt.subplot(3, 2, 1)
    plot_confusion_matrix_heat(y_test, test_pred, "Test Set Confusion Matrix")

    # Training Curves
    plt.subplot(3, 2, 2)
    plot_training_curves(train_metrics, val_metrics)

    # Feature Importance
    plt.subplot(3, 2, 3)
    plot_feature_importance_subset(feature_scores, selected_features)

    # Class Distribution
    plt.subplot(3, 2, 4)
    plot_class_distribution(y_test, test_pred)

    # Performance Metrics Over Time
    plt.subplot(3, 2, 5)
    plot_metrics_over_time(train_metrics, val_metrics)

    # Plot Error Analysis
    plt.subplot(3, 2, 6)
    plot_error_analysis(y_test, test_pred)

    plt.suptitle(f'Model Analysis - Subject {subject_idx} - {model_name} - Dataset {dataset_type}',
                 fontsize=16, y=1.02)
    plt.tight_layout()

    # Save the figure
    save_path = f'results/visualizations/subject_{subject_idx}_{model_name}_dataset_{dataset_type}.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()





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
    """Updates the performance summary dictionary."""
    summary['dataset'].append(dataset_type)
    summary['model'].append(model_name)
    summary['accuracy'].append(metrics['test_acc'])
    summary['accuracy_std'].append(std_metrics['test_acc'])
    summary['f1_score'].append(metrics['f1'])
    summary['f1_std'].append(std_metrics['f1'])
    summary['training_time'].append(training_time)


def generate_comparative_visualizations(all_results, dataset_types, results_dir):
    """Generates comprehensive comparative visualizations."""
    plt.style.use('seaborn')

    # Accuracy comparison across the datasets for [100, 200, 300]
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
    """Generates and save comprehensive summary report."""
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

    # Saves report
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
        print("Accuracy Metrics:Test Accuracy:  {metrics['test_acc']:.4f} ± {std_metrics['test_acc']:.4f}")
        print(f"Val Accuracy:   {metrics['val_acc']:.4f} ± {std_metrics['val_acc']:.4f}")
        print(f"Train Accuracy: {metrics['train_acc']:.4f} ± {std_metrics['train_acc']:.4f}")

        print("\nF1 Scores:")
        print(f"F1 Score: {metrics['f1']:.4f} ± {std_metrics['f1']:.4f}")

        print("\nAdditional Metrics:")
        print(f"Precision: {metrics.get('precision', 0):.4f} ± {std_metrics.get('precision', 0):.4f}")
        print(f"Recall:    {metrics.get('recall', 0):.4f} ± {std_metrics.get('recall', 0):.4f}")
        print(f"\nProcessing Time: {aggregated_results['processing_time']:.2f} seconds")
        print(f"Number of Subjects: {aggregated_results['subject_count']}")


def plot_learning_curves_comparison(all_results: Dict, dataset_types: List[int], results_dir: str):
    """
    Plot learning curves comparison across different datasets.
        all_results: Dictionary containing results for all datasets
        dataset_types: List of dataset identifiers
        results_dir: Directory to save the plots
    """
    plt.figure(figsize=(15, 10))

    # Creates subplots for different metrics
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

                # Plotting  training curve
                plt.plot(epochs, history[f'train_{metric}'],
                         label=f'Dataset {dataset_type} - Train')
                # Plotting validation curve
                plt.plot(epochs, history[f'val_{metric}'],
                         label=f'Dataset {dataset_type} - Val',
                         linestyle='--')

        plt.title(f'{metric.upper()} Learning Curves')
        plt.xlabel('Epoch')
        plt.ylabel(metric.title())
        plt.legend()
        plt.grid(True)

    plt.suptitle('Learning Curves Comparison Across Datasets', fontsize=16)
    plt.tight_layout()

    plt.savefig(f'{results_dir}/plots/learning_curves_comparison.png')
    plt.close()


def plot_metric_comparison(all_results: Dict, dataset_types: List[int],
                           metric_name: str, metric_label: str):
    """

    Args:
        all_results: Dictionary containing results for all datasets
        dataset_types: List of dataset identifiers
        metric_name: Name of the metric to plot
        metric_label: Label for the metric in the plot
    """
    plt.figure(figsize=(12, 6))

    # Prepares these data for plotting
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

    for i, v in enumerate(values):
        plt.text(i, v + errors[i], f'{v:.3f}',
                 ha='center', va='bottom')

    plt.tight_layout()



def get_color_palette(n_colors: int) -> List[str]:
    """Creates a custom color palette for consistent plotting.
        n_colors: Number of colors needed
        List of color hex codes
    """
    return sns.color_palette("husl", n_colors=n_colors).as_hex()




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
    # Creates subdirectory
    config_dir = os.path.join(results_dir, f'dataset_{dataset_type}', key)
    os.makedirs(config_dir, exist_ok=True)

    # Saving metrics as JSON
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
    """Plot metrics over time (epochs) for training and validation."""
    plt.figure(figsize=(15, 5))

    # Plot 1: Accuracy
    plt.subplot(1, 3, 1)
    if 'acc' in train_metrics and len(train_metrics['acc']) > 0:
        epochs = range(1, len(train_metrics['acc']) + 1)
        plt.plot(epochs, train_metrics['acc'], 'b-', label='Train Accuracy')
        if 'acc' in val_metrics and len(val_metrics['acc']) == len(epochs):
            plt.plot(epochs, val_metrics['acc'], 'r-', label='Validation Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
    else:
        plt.title('Accuracy (No Data Available)')
        plt.axis('off')

    # Plot 2: Loss
    plt.subplot(1, 3, 2)
    if 'loss' in train_metrics and len(train_metrics['loss']) > 0:
        epochs = range(1, len(train_metrics['loss']) + 1)
        plt.plot(epochs, train_metrics['loss'], 'b-', label='Train Loss')
        if 'loss' in val_metrics and len(val_metrics['loss']) == len(epochs):
            plt.plot(epochs, val_metrics['loss'], 'r-', label='Validation Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    else:
        plt.title('Loss (No Data Available)')
        plt.axis('off')

    # Plot 3: F1 Score
    plt.subplot(1, 3, 3)
    if 'f1' in train_metrics and len(train_metrics['f1']) > 0:
        epochs = range(1, len(train_metrics['f1']) + 1)
        plt.plot(epochs, train_metrics['f1'], 'g-', label='Train F1')
        if 'f1' in val_metrics and len(val_metrics['f1']) == len(epochs):
            plt.plot(epochs, val_metrics['f1'], 'r-', label='Validation F1')
        plt.title('F1 Score over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
    else:
        plt.title('F1 Score (No Data Available)')
        plt.axis('off')

    plt.tight_layout()

def plot_error_analysis(y_true, y_pred):
    """Plot error analysis showing misclassification patterns."""

    # Ensuring y_true and y_pred are NumPy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

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

    for metric, value in mean_metrics.items():
        report.append(f"- {metric}: {value:.4f} ± {std_metrics[metric]:.4f}")

    report.extend([
        f"\nProcessing Information:",
        f"- Total Processing Time: {aggregated_results.get('processing_time', 'N/A')} seconds",
        f"- Number of Features: {len(aggregated_results.get('feature_importance_mean', []))}",
    ])

    with open(save_path, 'w') as f:
        f.write('\n'.join(report))






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
