
import torch
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Define the CNN model with dropout and weight regularization
class RegressionCNN(nn.Module):
    def __init__(self, input_shape, num_outputs):
        super(RegressionCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.fc1 = nn.Linear(93440, 1024)
        # Calculate the number of input features for the fully connected layer
        num_features = self._calculate_num_features(input_shape)
        # Fully connected layer
        self.fc1 = nn.Linear(num_features, 1024)
        self.dropout2 = nn.Dropout(0.2)
        # Output layer with linear activation for regression
        self.fc3 = nn.Linear(1024, num_outputs)

    def _calculate_num_features(self, input_shape):
        """Calculates the number of input features for the fully connected layer."""

        # Track the output shape through the convolutional and pooling layers
        output_shape = input_shape
        for layer in [self.conv1, self.pool1, self.conv2, self.pool2, self.conv3, self.pool3]:
            output_shape = layer(torch.zeros(1, *output_shape)).shape[1:]  # Pass a dummy input

        # Flatten the output shape
        num_features = int(np.prod(output_shape))  # Calculate the product of dimensions

        return num_features

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))

        # x = x.view(-1, 128 * 4 * 74)  # Flatten the output before the dense layers
        x = torch.flatten(x, 1)

        x = torch.relu(self.fc1(x))
        # x = self.dropout1(x)
        # x = torch.relu(self.fcx1(x))
        # x = torch.relu(self.fcx2(x))
        # x = torch.relu(self.fcx3(x))
        # x = torch.relu(self.fcx4(x))
        # x = torch.relu(self.fcx5(x))
        # x = torch.tanh(self.fc2(x))
        # x = self.dropout2(x)
        x = self.fc3(x)

        return x


# Define a custom dataset for regression
class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)  # Convert X_train to PyTorch Tensor
        self.y = torch.FloatTensor(y)  # Convert y_train to PyTorch Tensor

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]