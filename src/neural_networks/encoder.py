import torch
import torch.nn as nn
from src.utils.custom_typing import EncoderOutput


class BaseEncoderSH(nn.Module):
    def __init__(
            self,
            img_size1: int,
            img_size2: int,
            img_size3: int,
            in_channels: int,
            num_filters: int,
            kernel_size: int,
            repr_dim: int,

    ):
        """Encoder to extract the representations

        Args:
            img_size (int): [Image size (must be squared size)]
            in_channels (int): Number of input channels
            num_filters (int): Intermediate number of filters
            kernel_size (int): Convolution kernel size
            repr_dim (int): Dimension of the desired representation
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_filters * 2 ** 0,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=num_filters * 2 ** 0,
            out_channels=num_filters * 2 ** 1,
            kernel_size=kernel_size,
            stride=2,
        )
        self.bn2 = nn.BatchNorm2d(num_features=num_filters * 2 ** 1)
        self.conv3 = nn.Conv2d(
            in_channels=num_filters * 2 ** 1,
            out_channels=num_filters * 2 ** 2,
            kernel_size=kernel_size,
            stride=2,
        )
        self.bn3 = nn.BatchNorm2d(num_features=num_filters * 2 ** 2)
        self.conv4 = nn.Conv2d(
            in_channels=num_filters * 2 ** 2,
            out_channels=num_filters * 2 ** 2,
            kernel_size=kernel_size,
            stride=2,
        )
        self.bn4 = nn.BatchNorm2d(num_features=num_filters * 2 ** 2)

        self.leaky_relu = nn.LeakyReLU()
        self.flatten = nn.Flatten()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=1)

        img_size1 = (img_size1 - 2) // 1 + 1
        img_size2 = (img_size2 - 2) // 2 + 1
        img_size3 = (img_size3 - 2) // 2 + 1

        self.dense1 = nn.Linear(
            in_features=(img_size3 ** 2) * (num_filters * 2 ** 2) + (img_size2 ** 2) * (num_filters * 2 ** 2) + (
                    img_size1 ** 2) * (num_filters * 2 ** 1),
            out_features=repr_dim,
        )

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        """Forward encoder

        Args:
            x (torch.Tensor): Image from a given domain

        Returns:
            EncoderOutput: Representation and feature map
        """
        x = self.conv1(x)
        x = self.leaky_relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x1 = nn.AvgPool2d(kernel_size=2, stride=1)(x)

        y = self.conv3(x)
        y = self.bn3(y)
        y = self.leaky_relu(y)
        y1 = nn.AvgPool2d(kernel_size=2, stride=2)(y)

        z = self.conv4(y)
        z = self.bn4(z)
        feature = self.leaky_relu(z)
        z = self.leaky_relu(z)
        z1 = nn.AvgPool2d(kernel_size=2, stride=2)(z)

        flatten_x = self.flatten(x1)
        flatten_y = self.flatten(y1)
        flatten_z = self.flatten(z1)

        flatten_cat_xyz = torch.cat([flatten_x, flatten_y, flatten_z], dim=1)
        representation = self.dense1(flatten_cat_xyz)

        if self.training:  # Only add noise during training
            noise = torch.randn_like(representation) * 0.1
            representation = representation + noise

        return EncoderOutput(representation=representation, feature=feature)
        
        
class BaseEncoderEX(nn.Module):
    def __init__(
            self,
            img_size1: int,
            img_size2: int,
            img_size3: int,
            in_channels: int,
            num_filters: int,
            kernel_size: int,
            repr_dim: int,

    ):
        """Encoder to extract the representations

        Args:
            img_size (int): [Image size (must be squared size)]
            in_channels (int): Number of input channels
            num_filters (int): Intermediate number of filters
            kernel_size (int): Convolution kernel size
            repr_dim (int): Dimension of the desired representation
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_filters * 2 ** 0,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=num_filters * 2 ** 0,
            out_channels=num_filters * 2 ** 1,
            kernel_size=kernel_size,
            stride=2,
        )
        self.bn2 = nn.BatchNorm2d(num_features=num_filters * 2 ** 1)
        self.conv3 = nn.Conv2d(
            in_channels=num_filters * 2 ** 1,
            out_channels=num_filters * 2 ** 2,
            kernel_size=kernel_size,
            stride=2,
        )
        self.bn3 = nn.BatchNorm2d(num_features=num_filters * 2 ** 2)
        self.conv4 = nn.Conv2d(
            in_channels=num_filters * 2 ** 2,
            out_channels=num_filters * 2 ** 2,
            kernel_size=kernel_size,
            stride=2,
        )
        self.bn4 = nn.BatchNorm2d(num_features=num_filters * 2 ** 2)

        self.leaky_relu = nn.LeakyReLU()
        self.flatten = nn.Flatten()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=1)

        img_size1 = (img_size1 - 2) // 1 + 1
        img_size2 = (img_size2 - 2) // 2 + 1
        img_size3 = (img_size3 - 2) // 2 + 1

        self.dense1 = nn.Linear(
            in_features=(img_size3 ** 2) * (num_filters * 2 ** 2) + (img_size2 ** 2) * (num_filters * 2 ** 2) + (
                    img_size1 ** 2) * (num_filters * 2 ** 1),
            out_features=repr_dim,
        )

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        """Forward encoder

        Args:
            x (torch.Tensor): Image from a given domain

        Returns:
            EncoderOutput: Representation and feature map
        """
        x = self.conv1(x)
        x = self.leaky_relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x1 = nn.AvgPool2d(kernel_size=2, stride=1)(x)

        y = self.conv3(x)
        y = self.bn3(y)
        y = self.leaky_relu(y)
        y1 = nn.AvgPool2d(kernel_size=2, stride=2)(y)

        z = self.conv4(y)
        z = self.bn4(z)
        feature = self.leaky_relu(z)
        z = self.leaky_relu(z)
        z1 = nn.AvgPool2d(kernel_size=2, stride=2)(z)

        flatten_x = self.flatten(x1)
        flatten_y = self.flatten(y1)
        flatten_z = self.flatten(z1)

        flatten_cat_xyz = torch.cat([flatten_x, flatten_y, flatten_z], dim=1)
        representation = self.dense1(flatten_cat_xyz)

        if self.training:  # Only add noise during training
            noise = torch.randn_like(representation) * 0.1
            representation = representation + noise

        return EncoderOutput(representation=representation, feature=feature)
        