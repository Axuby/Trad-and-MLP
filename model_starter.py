'''
Name : Azubuine Samuel Tochukwu
PSU ID: 960826967
EE552 Project 2

Process Flow and function calls:

The Main function ===>> Classification ===> Process_subjects( for each subject of LOSO dataset split) ===> Picks model_type function to run
 Then either the TraditionalClassifier or the deep_learning function executes

 Then Plot_Visualizations to plot identified metrics of performance and others including confusion matrix,
 class distribution, mis-classification rate, feature importance and  etc.

}
'''
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score



class TraditionalClassifier(KNeighborsClassifier):
    """KNN-based classifier with an evaluate method"""

    def __init__(self, n_neighbors=10):
        super().__init__(n_neighbors=n_neighbors)

    def evaluate(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Evaluates model performance on train, validation and test sets."""
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

