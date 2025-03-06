import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix_heat(y_true, y_pred, title, ax=None):
    """Plot confusion matrix as heatmap."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    return ax

def plot_training_curves(train_metrics, val_metrics, ax=None):
    """Plot training and validation metrics over time."""
    if train_metrics is None or val_metrics is None:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Training metrics not available",
                horizontalalignment='center', verticalalignment='center')
        ax.axis('off')
        return ax

    # checking first if loss exists and has values
    if 'loss' not in train_metrics or len(train_metrics['loss']) == 0:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Training loss data not available",
                horizontalalignment='center', verticalalignment='center')
        ax.axis('off')
        return ax

    epochs = range(1, len(train_metrics['loss']) + 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 5))

    # Since we have multiple plots, we need to create subplots here
    # For simplicity, let's just plot loss in the main subplot if using ax parameter
    ax.plot(epochs, train_metrics['loss'], 'b-', label='Training Loss')
    ax.plot(epochs, val_metrics['loss'], 'r-', label='Validation Loss')
    ax.set_title('Training and Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

    return ax

def plot_error_analysis(y_true, y_pred, ax=None):
    """Plot error analysis showing misclassification patterns."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Ensure y_true and y_pred are NumPy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    error_mask = y_true != y_pred  # Boolean mask

    #  case with no errors
    if not np.any(error_mask):
        ax.text(0.5, 0.5, "No misclassifications found",
                horizontalalignment='center', verticalalignment='center')
        ax.axis('off')
        return ax

    error_true = y_true[error_mask]
    error_pred = y_pred[error_mask]

    error_pairs = pd.DataFrame({'True': error_true, 'Predicted': error_pred})
    error_counts = error_pairs.groupby(['True', 'Predicted']).size().unstack(fill_value=0)

    sns.heatmap(error_counts, annot=True, fmt='d', cmap='Reds', ax=ax)
    ax.set_title('Misclassification Patterns')
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')

    return ax

def plot_feature_importance_subset(feature_scores, selected_features, top_n=20, ax=None):
    """Plot feature importance scores for selected features."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if feature_scores is None or len(feature_scores) == 0:
        ax.text(0.5, 0.5, "Feature importance not available",
                horizontalalignment='center', verticalalignment='center')
        ax.axis('off')
        return ax

    # If there are too many features, show only top N
    if len(selected_features) > top_n:
        # Sort features by importance
        sorted_indices = np.argsort(feature_scores)[::-1]
        top_indices = sorted_indices[:top_n]

        # Get top features and scores
        top_features = np.array(selected_features)[top_indices]
        top_scores = feature_scores[top_indices]

        ax.bar(range(len(top_features)), top_scores)
        ax.set_title(f'Top {top_n} Feature Importance Scores')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Importance Score')
        plt.sca(ax)  # Set current axis
        plt.xticks(range(len(top_features)), top_features, rotation=90)
    else:
        ax.bar(range(len(selected_features)), feature_scores)
        ax.set_title('Feature Importance Scores')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Importance Score')
        plt.sca(ax)  # Set current axis
        plt.xticks(range(len(selected_features)), selected_features, rotation=90)

    return ax

def plot_class_distribution(y_true, y_pred, ax=None):
    """Plot true vs predicted class distribution."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    true_dist = pd.Series(y_true).value_counts().sort_index()
    pred_dist = pd.Series(y_pred).value_counts().sort_index()

    # Ensure both distributions have the same classes
    all_classes = np.unique(np.concatenate([true_dist.index, pred_dist.index]))

    # Reindex with all classes, filling missing values with 0
    true_dist = true_dist.reindex(all_classes, fill_value=0)
    pred_dist = pred_dist.reindex(all_classes, fill_value=0)

    width = 0.35
    x = np.arange(len(all_classes))

    ax.bar(x - width/2, true_dist, width, label='True')
    ax.bar(x + width/2, pred_dist, width, label='Predicted')

    ax.set_title('Class Distribution Comparison')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    plt.sca(ax)  # Set current axis
    plt.xticks(x, all_classes)
    ax.legend()

    return ax



def plot_visualizations(y_val, y_test, val_pred, test_pred, train_metrics, val_metrics,
                        subject_idx, model_name, dataset_type, selected_features, feature_scores):
    """
    Creates comprehensive visualizations for model performance and analysis.
    """
    # Create figure with subplots
    fig, axs = plt.subplots(3, 2, figsize=(20, 15))

    # Confusion Matrix
    try:
        if y_test is not None and test_pred is not None and len(y_test) > 0 and len(test_pred) > 0:
            plot_confusion_matrix_heat(y_test, test_pred, "Test Set Confusion Matrix", ax=axs[0, 0])
        else:
            axs[0, 0].text(0.5, 0.5, "Test data not available for confusion matrix",
                     horizontalalignment='center', verticalalignment='center')
            axs[0, 0].axis('off')
    except Exception as e:
        axs[0, 0].text(0.5, 0.5, f"Cannot plot confusion matrix: {str(e)}",
                 horizontalalignment='center', verticalalignment='center')
        axs[0, 0].axis('off')

    # Training Curves
    try:
        if train_metrics is not None and val_metrics is not None:
            plot_training_curves(train_metrics, val_metrics, ax=axs[0, 1])
        else:
            axs[0, 1].text(0.5, 0.5, "Training metrics not available",
                     horizontalalignment='center', verticalalignment='center')
            axs[0, 1].axis('off')
    except Exception as e:
        axs[0, 1].text(0.5, 0.5, f"Cannot plot training curves: {str(e)}",
                 horizontalalignment='center', verticalalignment='center')
        axs[0, 1].axis('off')

    # Feature Importance
    try:
        if feature_scores is not None and selected_features is not None:
            plot_feature_importance_subset(feature_scores, selected_features, ax=axs[1, 0])
        else:
            axs[1, 0].text(0.5, 0.5, "Feature importance data not available",
                     horizontalalignment='center', verticalalignment='center')
            axs[1, 0].axis('off')
    except Exception as e:
        axs[1, 0].text(0.5, 0.5, f"Cannot plot feature importance: {str(e)}",
                 horizontalalignment='center', verticalalignment='center')
        axs[1, 0].axis('off')

    # Class Distribution
    try:
        if y_test is not None and test_pred is not None and len(y_test) > 0 and len(test_pred) > 0:
            plot_class_distribution(y_test, test_pred, ax=axs[1, 1])
        else:
            axs[1, 1].text(0.5, 0.5, "Test data not available for class distribution",
                     horizontalalignment='center', verticalalignment='center')
            axs[1, 1].axis('off')
    except Exception as e:
        axs[1, 1].text(0.5, 0.5, f"Cannot plot class distribution: {str(e)}",
                 horizontalalignment='center', verticalalignment='center')
        axs[1, 1].axis('off')

    # Todo: For metrics over time, implement a similar function with ax parameter
    # For now, let's just show a placeholder, I will come back later
    axs[2, 0].text(0.5, 0.5, "Metrics over time not implemented with ax parameter",
                 horizontalalignment='center', verticalalignment='center')
    axs[2, 0].axis('off')

    # Error Analysis
    try:
        if y_test is not None and test_pred is not None and len(y_test) > 0 and len(test_pred) > 0:
            plot_error_analysis(y_test, test_pred, ax=axs[2, 1])
        else:
            axs[2, 1].text(0.5, 0.5, "Test data not available for error analysis",
                     horizontalalignment='center', verticalalignment='center')
            axs[2, 1].axis('off')
    except Exception as e:
        axs[2, 1].text(0.5, 0.5, f"Cannot plot error analysis: {str(e)}",
                 horizontalalignment='center', verticalalignment='center')
        axs[2, 1].axis('off')

    plt.suptitle(f'Model Analysis - Subject {subject_idx} - {model_name} - Dataset {dataset_type}',
                 fontsize=16, y=1.02)
    plt.tight_layout()

    # Save the figure
    save_path = f'results/visualizations/subject_{subject_idx}_{model_name}_dataset_{dataset_type}.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()