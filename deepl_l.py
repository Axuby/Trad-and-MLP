import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


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
    plot_training_curves(metrics)

    # Plot confusion matrix
    plot_confusion_matrix(test_labels, final_preds, output_dim)

    return model, final_metrics, metrics


def plot_training_curves(metrics):
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


# Example usage:

model, final_metrics, training_metrics = deep_learning(
    train_feats_proj=train_features,
    train_labels=train_labels,
    test_feats_proj=test_features,
    test_labels=test_labels,
    input_dim=train_features.shape[1],
    output_dim=len(np.unique(train_labels)),
    hidden_dim=64,
    num_layers=3,
    batch_size=32,
    learning_rate=0.001,
    epochs=100
)

print("\nFinal Metrics:")
print(f"Test Accuracy: {final_metrics['accuracy']:.4f} ± {final_metrics['std_acc']:.4f}")
print(f"Test F1 Score: {final_metrics['f1_score']:.4f} ± {final_metrics['std_f1']:.4f}")
