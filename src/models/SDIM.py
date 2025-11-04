import torch
import torch.nn as nn

from src.losses.Lipschitz import gradient_penalty
from src.neural_networks.encoder import BaseEncoderSH
from src.neural_networks.statistics_network import (
    WeightNet,
    global_and_concat,
    tile_and_concat,
    GlobalStatisticsNetworkSH,
    LocalStatisticsNetworkSH,
    weights_init,
)
from src.utils.custom_typing import SDIMOutputs, WeightNetOutputs
from src.neural_networks.classifier import Classifier


def normalize_feature_map(feature_map):
    mean = torch.mean(feature_map)
    std = torch.std(feature_map)
    normalized_feature_map = (feature_map - mean) / std
    return normalized_feature_map


class SDIM(nn.Module):
    def __init__(self, img_size: int, channels: int, shared_dim: int, switched: bool,
                 batch_size: int):
        """Shared Deep Info Max model. Extract the shared information from the images

        Args:
            img_size (int): Image size (must be squared size)
            channels (int): Number of inputs channels
            shared_dim (int): Dimension of the desired shared representation
            switched (bool): True to use cross mutual information, see paper
        """
        super().__init__()

        self.channels = channels
        self.shared_dim = shared_dim
        self.switched = switched
        self.batchSize = batch_size

        # Weights
        self.eps = 0.0000001
        self.netW_joint = WeightNet(in_channels=64 + self.shared_dim, out_dim=1)
        self.netW_margin = WeightNet(in_channels=64 + self.shared_dim, out_dim=1)

        self.kernel_size = 2
        self.img_size = (img_size - 3 + 2) // 1 + 1
        self.img_size1 = (self.img_size - self.kernel_size) // 2 + 1
        self.img_size2 = (self.img_size1 - self.kernel_size) // 2 + 1
        self.img_size3 = (self.img_size2 - self.kernel_size) // 2 + 1

        self.img_feature_size = [self.img_size3, self.img_size3]

        self.img_feature_channels = 64

        # Encoders
        self.sh_enc_x = BaseEncoderSH(
            img_size1=self.img_size1,
            img_size2=self.img_size2,
            img_size3=self.img_size3,
            in_channels=channels,
            num_filters=16,
            kernel_size=self.kernel_size,
            repr_dim=shared_dim,
        )

        self.sh_enc_y = BaseEncoderSH(
            img_size1=self.img_size1,
            img_size2=self.img_size2,
            img_size3=self.img_size3,
            in_channels=channels,
            num_filters=16,
            kernel_size=self.kernel_size,
            repr_dim=shared_dim,
        )

        self.local_stat_x = LocalStatisticsNetworkSH(
            img_feature_channels=self.img_feature_channels + self.shared_dim
        )

        self.local_stat_y = LocalStatisticsNetworkSH(
            img_feature_channels=self.img_feature_channels + self.shared_dim
        )

        # Global statistics network
        self.global_stat_x = GlobalStatisticsNetworkSH(
            feature_map_size=self.img_feature_size.copy(),
            feature_map_channels=self.img_feature_channels,
            latent_dim=self.shared_dim,
        )

        self.global_stat_y = GlobalStatisticsNetworkSH(
            feature_map_size=self.img_feature_size.copy(),
            feature_map_channels=self.img_feature_channels,
            latent_dim=self.shared_dim,
        )

        self.local_stat_x.apply(weights_init)
        self.local_stat_y.apply(weights_init)
        self.global_stat_x.apply(weights_init)
        self.global_stat_y.apply(weights_init)

        self.card_classifier = Classifier(feature_dim=shared_dim, output_dim=10, units=64)
        self.suit_classifier = Classifier(feature_dim=shared_dim, output_dim=4, units=64)

    def compute_weights(self, input_x, input_prime):
        weights_joint = self.netW_joint(input_x) + self.eps
        weights_joint = (weights_joint / weights_joint.sum()) * self.batchSize

        weights_margin = self.netW_margin(input_prime) + self.eps
        weights_margin = (weights_margin / weights_margin.sum()) * self.batchSize

        return weights_joint, weights_margin

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> SDIMOutputs:
        """Forward pass of the shared model

        Args:
            x (torch.Tensor): Image from domain X
            y (torch.Tensor): Image from domain Y

        Returns:
            SDIMOutputs: Outputs of the ROTDM model
        """

        # Get the shared and exclusive features from x and y
        shared_x, M_x = self.sh_enc_x(x)
        shared_y, M_y = self.sh_enc_y(y)

        M_x = normalize_feature_map(M_x)
        M_y = normalize_feature_map(M_y)

        # Shuffle M to create M'
        M_x_prime = torch.cat([M_x[1:], M_x[0].unsqueeze(0)], dim=0)
        M_y_prime = torch.cat([M_y[1:], M_y[0].unsqueeze(0)], dim=0)

        # Tile the exclusive representations (R) of each image and get the cross representations
        if self.switched:  # Shared representations are switched
            R_x_y = shared_x
            R_y_x = shared_y
        else:  # Shared representations are not switched
            R_x_y = shared_y
            R_y_x = shared_x

        global_input_x = global_and_concat(M_x, R_y_x)
        global_input_x_prime = global_and_concat(M_x_prime, R_y_x)
        global_input_y = global_and_concat(M_y, R_x_y)
        global_input_y_prime = global_and_concat(M_y_prime, R_x_y)

        local_input_x, concat_M_R_x = tile_and_concat(tensor=M_x, vector=R_y_x)
        local_input_x_prime, concat_M_R_x_prime = tile_and_concat(tensor=M_x_prime, vector=R_y_x)

        local_input_y, concat_M_R_y = tile_and_concat(tensor=M_y, vector=R_x_y)
        local_input_y_prime, concat_M_R_y_prime = tile_and_concat(tensor=M_y_prime, vector=R_x_y)

        # Global mutual information estimation
        global_mutual_M_R_x = self.global_stat_x(global_input_x)
        global_mutual_M_R_x_prime = self.global_stat_x(global_input_x_prime)
        global_gradient_penalty_x = gradient_penalty(self.global_stat_x, global_input_x, global_input_x_prime, 10)

        global_mutual_M_R_y = self.global_stat_y(global_input_y)
        global_mutual_M_R_y_prime = self.global_stat_y(global_input_y_prime)
        global_gradient_penalty_y = gradient_penalty(self.global_stat_y, global_input_y, global_input_y_prime, 10)

        # Local mutual information estimation
        local_mutual_M_R_x = self.local_stat_x(local_input_x)
        local_mutual_M_R_x_prime = self.local_stat_x(local_input_x_prime)
        local_gradient_penalty_x = gradient_penalty(self.local_stat_x, local_input_x, local_input_x_prime, 10)

        local_mutual_M_R_y = self.local_stat_y(local_input_y)
        local_mutual_M_R_y_prime = self.local_stat_y(local_input_y_prime)
        local_gradient_penalty_y = gradient_penalty(self.local_stat_y, local_input_y, local_input_y_prime, 10)

        input_x, _ = tile_and_concat(tensor=M_x, vector=shared_x)
        input_x_prime, _ = tile_and_concat(tensor=M_x_prime, vector=shared_x)

        input_y, _ = tile_and_concat(tensor=M_y, vector=shared_y)
        input_y_prime, _ = tile_and_concat(tensor=M_y_prime, vector=shared_y)

        weight_x_joint, weight_x_margin = self.compute_weights(input_x, input_x_prime)
        weight_y_joint, weight_y_margin = self.compute_weights(input_y, input_y_prime)

        card_logits = self.card_classifier(shared_y.detach())
        suit_logits = self.suit_classifier(shared_y.detach())

        return SDIMOutputs(
            global_mutual_M_R_x=global_mutual_M_R_x,
            global_mutual_M_R_x_prime=global_mutual_M_R_x_prime,
            global_mutual_M_R_y=global_mutual_M_R_y,
            global_mutual_M_R_y_prime=global_mutual_M_R_y_prime,
            local_mutual_M_R_x=local_mutual_M_R_x,
            local_mutual_M_R_x_prime=local_mutual_M_R_x_prime,
            local_mutual_M_R_y=local_mutual_M_R_y,
            local_mutual_M_R_y_prime=local_mutual_M_R_y_prime,
            global_M_R_x=global_input_x,
            global_M_R_x_prime=global_input_x_prime,
            global_M_R_y=global_input_y,
            global_M_R_y_prime=global_input_y_prime,
            shared_x=shared_x,
            shared_y=shared_y,
            global_gradient_penalty_x=global_gradient_penalty_x,
            global_gradient_penalty_y=global_gradient_penalty_y,
            local_gradient_penalty_x=local_gradient_penalty_x,
            local_gradient_penalty_y=local_gradient_penalty_y,
            weight_x_joint=weight_x_joint,
            weight_y_joint=weight_y_joint,
            weight_x_margin=weight_x_margin,
            weight_y_margin=weight_y_margin,
            card_logits=card_logits,
            suit_logits=suit_logits,
            input_x=input_x,
            input_x_prime=input_x_prime,
            input_y=input_y,
            input_y_prime=input_y_prime,
        )
