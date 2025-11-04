import torch
import torch.nn as nn
from src.losses.Lipschitz import gradient_penalty, gradient_penalty_representation
from src.neural_networks.encoder import BaseEncoderEX
from src.neural_networks.statistics_network import (
    LocalStatisticsNetworkEX,
    GlobalStatisticsNetworkEX,
    tile_and_concat,
    global_and_concat,
    weights_init,
    WeightNet,
    DisentangleWeightNet,
    Critic
)

from src.utils.custom_typing import (
    ClassifierOutputs,
    EDIMOutputs,
    DiscriminatorOutputs,
    DiscrLosses,
)
from src.neural_networks.classifier import Classifier


def normalize_feature_map(feature_map):
    mean = torch.mean(feature_map)
    std = torch.std(feature_map)
    normalized_feature_map = (feature_map - mean) / std
    return normalized_feature_map


class EDIM(nn.Module):
    """Exclusive Deep Info Max model. Extract the exclusive information from the images.

    Args:
        img_size (int): Image size (must be squared size)
        channels (int): Number of inputs channels
        shared_dim (int): Dimension of the pretrained shared representation
        exclusive_dim (int): Dimension of the desired shared representation
        trained_encoder_x (nn.Module): Trained encoder on domain X (pretrained with SDIM)
        trained_encoder_y (nn.Module): Trained encoder on domain Y (pretrained with SDIM)
    """

    def __init__(
            self,
            img_size: int,
            channels: int,
            shared_dim: int,
            exclusive_dim: int,
            trained_encoder_x: nn.Module,
            trained_encoder_y: nn.Module,
            batch_size: int
    ):

        super().__init__()

        self.img_size = img_size
        self.channels = channels
        self.shared_dim = shared_dim
        self.exclusive_dim = exclusive_dim
        self.batchSize = batch_size

        # Weights
        self.eps = 0.0000001
        self.netW_joint = WeightNet(in_channels=128 + self.shared_dim + self.exclusive_dim, out_dim=1)
        self.netW_margin = WeightNet(in_channels=128 + self.shared_dim + self.exclusive_dim, out_dim=1)

        self.disen_netW_joint = DisentangleWeightNet(nb_linear=3, input_dim=self.shared_dim + self.exclusive_dim)
        self.disen_netW_margin = DisentangleWeightNet(nb_linear=3, input_dim=self.shared_dim + self.exclusive_dim)

        # Encoders
        self.sh_enc_x = trained_encoder_x
        self.sh_enc_y = trained_encoder_y

        self.kernel_size = 2
        self.img_size = (img_size - 3 + 2) // 1 + 1
        self.img_size1 = (self.img_size - self.kernel_size) // 2 + 1
        self.img_size2 = (self.img_size1 - self.kernel_size) // 2 + 1
        self.img_size3 = (self.img_size2 - self.kernel_size) // 2 + 1

        self.img_feature_size = [self.img_size3, self.img_size3]

        self.img_feature_channels = 64

        self.ex_enc_x = BaseEncoderEX(
            img_size1=self.img_size1,
            img_size2=self.img_size2,
            img_size3=self.img_size3,
            in_channels=channels,
            num_filters=16,
            kernel_size=self.kernel_size,
            repr_dim=exclusive_dim,
        )

        self.ex_enc_y = BaseEncoderEX(
            img_size1=self.img_size1,
            img_size2=self.img_size2,
            img_size3=self.img_size3,
            in_channels=channels,
            num_filters=16,
            kernel_size=self.kernel_size,
            repr_dim=exclusive_dim,
        )

        # Local statistics network
        self.local_stat_x = LocalStatisticsNetworkEX(
            img_feature_channels=self.img_feature_channels * 2 + self.shared_dim + self.exclusive_dim
        )

        self.local_stat_y = LocalStatisticsNetworkEX(
            img_feature_channels=self.img_feature_channels * 2 + self.shared_dim + self.exclusive_dim
        )

        # Global statistics network
        self.global_stat_x = GlobalStatisticsNetworkEX(
            feature_map_size=self.img_feature_size.copy(),
            feature_map_channels=self.img_feature_channels * 2,
            latent_dim=self.shared_dim + self.exclusive_dim,
        )

        self.global_stat_y = GlobalStatisticsNetworkEX(
            feature_map_size=self.img_feature_size.copy(),
            feature_map_channels=self.img_feature_channels * 2,
            latent_dim=self.shared_dim + self.exclusive_dim,
        )

        self.local_stat_x.apply(weights_init)
        self.local_stat_y.apply(weights_init)
        self.global_stat_x.apply(weights_init)
        self.global_stat_y.apply(weights_init)

        # Metric nets
        self.card_classifier = Classifier(feature_dim=self.exclusive_dim, output_dim=10, units=64)
        self.suit_classifier = Classifier(feature_dim=self.exclusive_dim, output_dim=4, units=64)

        self.critic_x = Critic(nb_linear=4, input_dim=self.shared_dim + self.exclusive_dim)
        self.critic_y = Critic(nb_linear=4, input_dim=self.shared_dim + self.exclusive_dim)
        
        print("exclusive_dim", self.exclusive_dim)

    def compute_real_weights(self, input_x, input_prime):
        weights_joint = self.netW_joint(input_x) + self.eps
        weights_joint = (weights_joint / weights_joint.sum()) * self.batchSize

        weights_margin = self.netW_margin(input_prime) + self.eps
        weights_margin = (weights_margin / weights_margin.sum()) * self.batchSize

        return weights_joint, weights_margin

    def compute_real_disen_weights(self, rep_joint, rep_margin):
        disen_weights_joint = self.disen_netW_joint(rep_joint) + self.eps
        disen_weights_joint = (disen_weights_joint / disen_weights_joint.sum()) * self.batchSize

        disen_weights_margin = self.disen_netW_margin(rep_margin) + self.eps
        disen_weights_margin = (disen_weights_margin / disen_weights_margin.sum()) * self.batchSize

        return disen_weights_joint, disen_weights_margin

    def forward_generator(self, x: torch.Tensor, y: torch.Tensor) -> EDIMOutputs:
        """Forward pass of the generator

        Args:
            x (torch.Tensor): Image from domain X
            y (torch.Tensor): Image from domain Y

        Returns:
            EDIMOutputs:
        """
        # Get the shared and exclusive features from x and y
        shared_x, shared_M_x = self.sh_enc_x(x)
        shared_y, shared_M_y = self.sh_enc_y(y)

        exclusive_x, exclusive_M_x = self.ex_enc_x(x)
        exclusive_y, exclusive_M_y = self.ex_enc_y(y)

        shared_M_x = normalize_feature_map(shared_M_x)
        shared_M_y = normalize_feature_map(shared_M_y)
        exclusive_M_x = normalize_feature_map(exclusive_M_x)
        exclusive_M_y = normalize_feature_map(exclusive_M_y)

        # Concat exclusive and shared feature map
        M_x = torch.cat([shared_M_x, exclusive_M_x], dim=1)
        M_y = torch.cat([shared_M_y, exclusive_M_y], dim=1)

        # Shuffle M to create M'
        M_x_prime = torch.cat([M_x[1:], M_x[0].unsqueeze(0)], dim=0)
        M_y_prime = torch.cat([M_y[1:], M_y[0].unsqueeze(0)], dim=0)

        R_x_y = torch.cat([shared_x, exclusive_y], dim=1)
        R_y_x = torch.cat([shared_y, exclusive_x], dim=1)

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

        input_x, _ = tile_and_concat(tensor=M_x, vector=torch.cat([shared_x, exclusive_x], dim=1))
        input_x_prime, _ = tile_and_concat(tensor=M_x_prime, vector=torch.cat([shared_x, exclusive_x], dim=1))

        input_y, _ = tile_and_concat(tensor=M_y, vector=torch.cat([shared_y, exclusive_y], dim=1))
        input_y_prime, _ = tile_and_concat(tensor=M_y_prime, vector=torch.cat([shared_y, exclusive_y], dim=1))

        weight_x_joint, weight_x_margin = self.compute_real_weights(input_x, input_x_prime)
        weight_y_joint, weight_y_margin = self.compute_real_weights(input_y, input_y_prime)

        indices = torch.randperm(shared_x.shape[0])

        shared_x_prime = shared_x[indices]
        shared_y_prime = shared_y[indices]

        representation_x_joint = torch.cat([shared_y.detach(), exclusive_x], dim=1)
        representation_y_joint = torch.cat([shared_x.detach(), exclusive_y], dim=1)
        representation_x_margin = torch.cat([shared_y_prime.detach(), exclusive_x], dim=1)
        representation_y_margin = torch.cat([shared_x_prime.detach(), exclusive_y], dim=1)

        critic_x = self.critic_x(representation_x_joint)
        critic_x_prime = self.critic_x(representation_x_margin)

        critic_y = self.critic_y(representation_y_joint)
        critic_y_prime = self.critic_y(representation_y_margin)

        disen_weight_x_joint, disen_weight_x_margin = self.compute_real_disen_weights(representation_x_joint,
                                                                                      representation_x_margin)
        disen_weight_y_joint, disen_weight_y_margin = self.compute_real_disen_weights(representation_y_joint,
                                                                                      representation_y_margin)

        return EDIMOutputs(
            global_mutual_M_R_x=global_mutual_M_R_x,
            global_mutual_M_R_x_prime=global_mutual_M_R_x_prime,
            global_mutual_M_R_y=global_mutual_M_R_y,
            global_mutual_M_R_y_prime=global_mutual_M_R_y_prime,
            local_mutual_M_R_x=local_mutual_M_R_x,
            local_mutual_M_R_x_prime=local_mutual_M_R_x_prime,
            local_mutual_M_R_y=local_mutual_M_R_y,
            local_mutual_M_R_y_prime=local_mutual_M_R_y_prime,
            global_gradient_penalty_x=global_gradient_penalty_x,
            global_gradient_penalty_y=global_gradient_penalty_y,
            local_gradient_penalty_x=local_gradient_penalty_x,
            local_gradient_penalty_y=local_gradient_penalty_y,
            weight_x_joint=weight_x_joint,
            weight_y_joint=weight_y_joint,
            weight_x_margin=weight_x_margin,
            weight_y_margin=weight_y_margin,
            input_x=input_x,
            input_x_prime=input_x_prime,
            input_y=input_y,
            input_y_prime=input_y_prime,
            shared_x=shared_x,
            shared_y=shared_y,
            exclusive_x=exclusive_x,
            exclusive_y=exclusive_y,
            representation_x_joint=representation_x_joint,
            representation_y_joint=representation_y_joint,
            representation_x_margin=representation_x_margin,
            representation_y_margin=representation_y_margin,
            critic_x=critic_x,
            critic_y=critic_y,
            critic_x_prime=critic_x_prime,
            critic_y_prime=critic_y_prime,
            disen_weight_x_joint=disen_weight_x_joint,
            disen_weight_y_joint=disen_weight_y_joint,
            disen_weight_x_margin=disen_weight_x_margin,
            disen_weight_y_margin=disen_weight_y_margin,
        )

    def forward_dis_critic(self, edim_outputs: EDIMOutputs) -> DiscrLosses:
        out = edim_outputs
        representation_x_joint = out.representation_x_joint.detach().clone().requires_grad_(True)
        representation_y_joint = out.representation_y_joint.detach().clone().requires_grad_(True)
        representation_x_margin = out.representation_x_margin.detach().clone().requires_grad_(True)
        representation_y_margin = out.representation_y_margin.detach().clone().requires_grad_(True)

        critic_x = self.critic_x(representation_x_joint)
        critic_x_prime = self.critic_x(representation_x_margin)
        critic_gp_x = gradient_penalty_representation(self.critic_x, representation_x_joint, representation_x_margin,
                                                 10.0)

        critic_y = self.critic_y(representation_y_joint)
        critic_y_prime = self.critic_y(representation_y_margin)
        critic_gp_y = gradient_penalty_representation(self.critic_y, representation_y_joint, representation_y_margin,
                                                 10.0)

        loss_dis_x = - (torch.mean(critic_x * out.disen_weight_x_joint.detach()) - torch.mean(
            critic_x_prime * out.disen_weight_x_margin.detach()))
        loss_dis_y = - (torch.mean(critic_y * out.disen_weight_y_joint.detach()) - torch.mean(
            critic_y_prime * out.disen_weight_y_margin.detach()))

        loss_dis = loss_dis_x + loss_dis_y
        critic_gp = critic_gp_x + critic_gp_y

        loss_dis_d = (loss_dis + critic_gp) * 1.0

        return DiscrLosses(
            loss_dis_d=loss_dis_d,
            loss_dis=loss_dis,
            critic_gp=critic_gp
        )

    def forward_classifier(self, edim_outputs: EDIMOutputs) -> ClassifierOutputs:
        """Forward pass of the classifiers

        Args:
            edim_outputs (EDIMOutputs): Outputs from the generator

        Returns:
            ClassifierOutputs: Classifiers Outputs
        """
        out = edim_outputs
        card_logits = self.card_classifier(out.exclusive_y.detach())
        suit_logits = self.suit_classifier(out.exclusive_y.detach())

        return ClassifierOutputs(
            card_logits=card_logits,
            suit_logits=suit_logits,
        )