import torch
from torch import nn


# convolution-relu-
# convolution-relu-max pooling-dropout-dense-relu-dropout-
# dense-softmax, with 32 convolution kernels, 4x4 kernel size,
# 2x2 pooling, dense layer with 128 units, and dropout proba-
# bilities 0.25 and 0.5
class ReferenceCNN(nn.Module):

    def __init__(
        self,
        num_classes: int = 10,
        conv_channels: int = 32,
        kernel_size: int = 4,
        pool_size: int = 2,
        dense_units: int = 128,
        dropout1: float = 0.25,
        dropout2: float = 0.5,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, conv_channels, kernel_size=kernel_size, padding=0)
        self.conv2 = nn.Conv2d(
            conv_channels, conv_channels, kernel_size=kernel_size, padding=0
        )
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_size)
        self.drop1 = nn.Dropout(p=dropout1)

        flat_dim = conv_channels * 11 * 11
        self.fc1 = nn.Linear(flat_dim, dense_units)
        self.drop2 = nn.Dropout(p=dropout2)
        self.fc2 = nn.Linear(dense_units, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.drop2(x)
        return self.fc2(x)

    def forward_features(
        self, x: torch.Tensor, *, apply_dropout2: bool = True
    ) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        if apply_dropout2:
            x = self.drop2(x)
        return x
