# Taiji Movement Classification System Using an MLP and Traditional Classifier

## Overview
This project implements a machine learning system for classifying Taiji movements using sensor data. It supports both traditional classification methods and deep neural networks, with comprehensive feature processing, evaluation, and visualization capabilities.

## Features
- Processes three different Taiji movement datasets (100, 200, 300)
- Implements leave-one-subject-out (LOSO) cross-validation
- Supports feature selection via Sequential Forward Selection (SFS)
- Optional dimensionality reduction using Fisher Linear Discriminant Analysis (LDA)
- Includes both traditional and deep learning classification approaches
- Extensive performance evaluation and visualization tools
- Lastly, Optional dimensionality reduction using Principal Component Analysis (PCA)

## Requirements
- Python 3.6+
- PyTorch
- NumPy
- scikit-learn
- Matplotlib
- Seaborn

## Project Structure

### Core Components

- `fisher_projection`: Custom implementation of Fisher Linear Discriminant Analysis for dimensionality reduction
- `deep_learning`: Implementation of MLP training and evaluation with comprehensive metrics tracking
- `convert_features_to_loader`: Utility to create PyTorch DataLoaders from feature matrices
- `load_dataset`: Loads and preprocesses the Taiji datasets
- `apply_feature_processing`: Applies feature selection and optional LDA projection, (now includes PCA)
- `process_subject`: Processes a single subject for leave-one-subject-out validation
- `classification`: Main classification pipeline with support for multiple configurations
- `main`: Orchestrates the complete classification pipeline for all datasets and configurations [100, 200, 200]

### Helper Functions

- `plot_training_curves_main`: Generates training and validation curves
- `plot_confusion_matrix`: Creates confusion matrix visualizations
- `plot_visualizations`: Generates all visualizations for a subject
- `print_results_summary`: Prints comprehensive results summary
- `create_results_directory`: Sets up directory structure for results
- `update_performance_summary`: Updates the performance summary dictionary
- `save_detailed_results`: Saves detailed results to files
- `generate_comparative_visualizations`: Creates comparative visualizations across configurations
- `print_dataset_statistics`: Prints dataset characteristics
- `print_dataset_results`: Prints results for a specific dataset

## Usage

1. Ensure the dataset files (`Taiji_dataset_100.csv`, `Taiji_dataset_200.csv`, and `Taiji_dataset_300.csv`) are in the working directory
2. Run the main script:

```python
python classification_starter.py
```

3. Results will be stored in the `results/` directory with the following structure:
   - Detailed metrics for each configuration
   - Confusion matrices
   - Training and validation curves
   - Feature importance visualizations
   - Summary reports and comparative analyses

## Function Descriptions

### Main Processing Functions

#### `fisher_projection(X, y, n_components)`
Implements Fisher Linear Discriminant Analysis for dimensionality reduction.
- **Parameters**:
  - `X`: Input features matrix
  - `y`: Target labels
  - `n_components`: Number of components to keep
- **Returns**: Projection matrix and explained variance ratio

#### `deep_learning(train_feats_proj, train_labels, test_feats_proj, test_labels, input_dim, output_dim, hidden_dim, num_layers, batch_size, learning_rate, epochs)`
Trains and evaluates an MLP model with comprehensive metrics tracking.
- **Parameters**:
  - Training and testing data
  - Model configuration parameters
  - Training hyperparameters
- **Returns**: Trained model, final metrics, and training history

#### `load_dataset(verbose, dataset_type)`
Loads the specified Taiji dataset.
- **Parameters**:
  - `verbose`: Whether to print dataset information
  - `dataset_type`: Dataset identifier (100, 200, or 300)
- **Returns**: Features, labels, and subject identifiers

#### `apply_feature_processing(X_train, X_val, X_test, y_train, use_lda)`
Applies feature selection and optional LDA projection.
- **Parameters**:
  - Training, validation, and test feature matrices
  - Training labels
  - Whether to use LDA
- **Returns**: Processed feature matrices, selected features, and feature scores

#### `process_subject(subject_idx, features, labels, subjects, dataset_type, model_type, use_lda)`
Processes a single subject for leave-one-subject-out validation.
- **Parameters**:
  - Subject identifier and dataset
  - Configuration parameters
- **Returns**: Subject-specific results dictionary

#### `classification(features, labels, subjects, dataset_type, model_type, use_lda, use_pca)`
Performs classification with optional LDA projection and a PCA(use_pca param, but preference is
given to use_lda, LDA projection) and LOSO cross-validation.
- **Parameters**:
  - Dataset matrices
  - Configuration options
- **Returns**: Mean metrics, standard deviations, and aggregated results

### Utility Functions

#### `convert_features_to_loader(train_feats, train_labels, test_feats, test_labels, batch_size)`
Converts feature matrices to PyTorch DataLoaders.
- **Parameters**: Feature matrices, labels, and batch size
- **Returns**: Training and test DataLoaders

#### `print_results_summary(mean_metrics, std_metrics, aggregated_results)`
Prints a comprehensive summary of classification results.
- **Parameters**: Performance metrics and aggregated results

## Results
The system generates comprehensive results including:
- Performance metrics (accuracy, F1 score, precision, recall)
- Confusion matrices
- Training and validation curves
- Feature importance visualizations
- Comparative analyses across configurations

## Author
Azubuine Samuel Tochukwu