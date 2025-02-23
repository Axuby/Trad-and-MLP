'''
Name : Azubuine Samuel Tochukwu
PSU ID: 960826967
EE552 Project 2
}
'''
from networkx.classes import neighbors
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np


class TraditionalClassifier(KNeighborsClassifier):
    """
    Traditional KNN classifier with integrated scaling

    This implementation extends the standard KNeighborsClassifier with:
    - Integrated feature scaling
    - Enhanced prediction methods
    - Additional metrics tracking if you want
    """

    def __init__(self, n_neighbors: int = 5, **kwargs):
        """
        Initialize the classifier

        Args:
            n_neighbors: Number of neighbors for KNN
            **kwargs: Additional arguments for KNeighborsClassifier
        """
        super().__init__(n_neighbors=n_neighbors, **kwargs)
        self.scaler = StandardScaler()
        self.classes_ = None
        self.feature_weights_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TraditionalClassifier':
        """
        Fit the classifier

        Args:
            X: Training features
            y: Training labels

        Returns:
            self: The fitted classifier
        """
        # Store unique classes
        self.classes_ = np.unique(y)

        # Calculate feature weights based on variance
        self.feature_weights_ = np.var(X, axis=0)
        self.feature_weights_ = self.feature_weights_ / np.sum(self.feature_weights_)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit the base classifier
        return super().fit(X_scaled, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels

        Args:
            X: Test features

        Returns:
            Predicted class labels
        """
        if X.shape[0] == 0:
            return np.array([])

        # Scale features
        X_scaled = self.scaler.transform(X)

        return super().predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities

        Args:
            X: Test features

        Returns:
            Class probabilities
        """
        if X.shape[0] == 0:
            return np.array([])

        # Scale features
        X_scaled = self.scaler.transform(X)

        return super().predict_proba(X_scaled)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score

        Args:
            X: Test features
            y: True labels

        Returns:
            Accuracy score
        """
        return np.mean(self.predict(X) == y)

    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance scores based on variance

        Returns:
            Array of feature importance scores
        """
        return self.feature_weights_ if self.feature_weights_ is not None else None


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, activation=nn.ReLU, num_layers=2):
        super(MLP, self).__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
