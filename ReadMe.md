from classification_starter_2 import process_subject

# Tai Chi Movement Classification using KNN and LOSO Cross-validation

## Project Overview
This project implements a machine learning system for classifying Tai Chi movements using K-Nearest Neighbors (KNN) classification with Leave-One-Subject-Out (LOSO) cross-validation. The system employs advanced feature selection techniques including mRMR (Minimum Redundancy Maximum Relevance) and Sequential Forward Selection (SFS) to optimize performance.

## Architecture
![System Architecture](path/to/image)

The system consists of four main components:

- **Data Loading & Preprocessing**
- **Feature Selection Pipeline**
- **KNN Classification**
- **Results Visualization**

## Key Features
- Leave-One-Subject-Out (LOSO) cross-validation for robust generalization testing
- Two-stage feature selection combining mRMR and Sequential Forward Selection
- Comprehensive visualization suite for performance analysis
- Support for multiple dataset types (100, 200, 300)

## Basic Requirements to Run
```
numpy
torch
sklearn
matplotlib
seaborn
scipy
```

## Core Functions
### Data Loading Load_dataset func
```python
load_dataset(verbose: bool = True,
            subject_index: int = 9,
            dataset_type: int = 100,
            use_feature_selection: bool = True,
            selection_method: str = 'mrmr',
            n_features: int = 10,
            use_sfs: bool = False,
            n_neighbors: int = 5)
```

### Feature Selection using SFS
```python
def sequential_forward_selection(X: np.ndarray, y: np.ndarray,
                               n_features: int = 10,
                               cv_splits: int = 5,
                               n_neighbors: int = 5)
```
Implements Sequential Forward Selection for optimal feature subset selection.

### Fisher Projection
```python
def fisher_projection(X: np.ndarray, y: np.ndarray, n_components: int = None) -> Tuple[np.ndarray, np.ndarray]:
```
Applies Fisher’s Linear Discriminant Analysis to reduce dimensionality while preserving class separability.


### Process Subjects; Runs the iterative process for the LOSO 
```python
process_subject(params)
```



### Classification
```python
def classification(dataset_types: List[int] = [200, 300],
                  use_lda: bool = True,
                  n_features: int = 10,
                  n_neighbors: int = 5)
```


### Main: starts the classifier processes for MLP or Traditional and oversees
### ensures the iterative run of the dataset  for the [100, 200, 300]
```python
def main()
```
## Results
The system generates various visualization outputs:

- Feature importance plots
- Confusion matrices
- Training progress graphs
- Dataset comparisons

Results are saved in the `results/` directory.

## Feature Selection Process
- Initial feature ranking using mRMR
- Selection of top K features
- Sequential Forward Selection for optimal subset
- Final feature set used for classification

## Cross-validation Strategy
- Leave-One-Subject-Out (LOSO) approach
- Training on 9 subjects, testing on 1
- Ensures robust generalization testing
- Repeats for all subjects

## Performance Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices

## Contributing
Feel free to submit issues and enhancement requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

