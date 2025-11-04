import torch.nn as nn
import torch
from src.losses.loss_functions import (
    DJSLoss,
    ClassifLoss,
    DiscriminatorLoss,
    GeneratorLoss,
)
from src.utils.custom_typing import (
    DiscriminatorOutputs,
    ClassifierOutputs,
    EDIMOutputs,
    GenLosses,
    DiscrLosses,
    ClassifLosses,
)
import torch.nn.functional as F


class EDIMLoss(nn.Module):
    """Loss function to extract exclusive information from the image, see paper equation (8)

    Args:
        local_mutual_loss_coeff (float): Coefficient of the local Jensen Shannon loss
        global_mutual_loss_coeff (float): Coefficient of the global Jensen Shannon loss
        disentangling_loss_coeff (float): Coefficient of the Gan loss
    """

    def __init__(
            self,
            local_mutual_loss_coeff: float,
            global_mutual_loss_coeff: float,
            disentangling_loss_coeff: float,
    ):

        super().__init__()
        self.local_mutual_loss_coeff = local_mutual_loss_coeff
        self.global_mutual_loss_coeff = global_mutual_loss_coeff
        self.disentangling_loss_coeff = disentangling_loss_coeff

        self.classif_loss = ClassifLoss()

        self.rho = 0.03
        self.disen_rho = 0.00001

    def ROTDM(self, T: torch.Tensor, T_prime: torch.Tensor, weight_joint, weight_margin):
        joint_expectation = torch.mean(T * weight_joint)
        marginal_expectation = torch.mean(T_prime * weight_margin)
        w_d = joint_expectation - marginal_expectation
        return -w_d


    def weight_loss(self, edim_outputs: EDIMOutputs):
        # Chi-squared
        soft_constraint_joint_x = 100 * F.relu(
            torch.mean(0.5 * ((edim_outputs.weight_x_joint - 1) ** 2)) - self.rho)
        soft_constraint_margin_x = 100 * F.relu(
            torch.mean(0.5 * ((edim_outputs.weight_x_margin - 1) ** 2)) - self.rho)
        soft_constraint_joint_y = 100 * F.relu(
            torch.mean(0.5 * ((edim_outputs.weight_y_joint - 1) ** 2)) - self.rho)
        soft_constraint_margin_y = 100 * F.relu(
            torch.mean(0.5 * ((edim_outputs.weight_y_margin - 1) ** 2)) - self.rho)

        global_mutual_loss_x = -1 * self.ROTDM(
            T=edim_outputs.global_mutual_M_R_x.detach(),
            T_prime=edim_outputs.global_mutual_M_R_x_prime.detach(),
            weight_joint=edim_outputs.weight_x_joint,
            weight_margin=edim_outputs.weight_x_margin,
        )
        global_mutual_loss_y = -1 * self.ROTDM(
            T=edim_outputs.global_mutual_M_R_y.detach(),
            T_prime=edim_outputs.global_mutual_M_R_y_prime.detach(),
            weight_joint=edim_outputs.weight_y_joint,
            weight_margin=edim_outputs.weight_y_margin,
        )
        local_mutual_loss_x = -1 * self.ROTDM(
            T=edim_outputs.local_mutual_M_R_x.detach(),
            T_prime=edim_outputs.local_mutual_M_R_x_prime.detach(),
            weight_joint=edim_outputs.weight_x_joint.unsqueeze(-1).unsqueeze(-1),
            weight_margin=edim_outputs.weight_x_margin.unsqueeze(-1).unsqueeze(-1),
        )
        local_mutual_loss_y = -1 * self.ROTDM(
            T=edim_outputs.local_mutual_M_R_y.detach(),
            T_prime=edim_outputs.local_mutual_M_R_y_prime.detach(),
            weight_joint=edim_outputs.weight_y_joint.unsqueeze(-1).unsqueeze(-1),
            weight_margin=edim_outputs.weight_y_margin.unsqueeze(-1).unsqueeze(-1),
        )

        loss_weight_x = global_mutual_loss_x + local_mutual_loss_x + soft_constraint_joint_x + soft_constraint_margin_x
        loss_weight_y = global_mutual_loss_y + local_mutual_loss_y + soft_constraint_joint_y + soft_constraint_margin_y
        loss_weights = loss_weight_x + loss_weight_y
        return loss_weights

    def disen_weight_loss(self, edim_outputs: EDIMOutputs):
        # Chi-squared
        soft_constraint_joint_x = 1000 * F.relu(
            torch.mean(0.5 * ((edim_outputs.disen_weight_x_joint - 1) ** 2)) - self.disen_rho)
        soft_constraint_margin_x = 1000 * F.relu(
            torch.mean(0.5 * ((edim_outputs.disen_weight_x_margin - 1) ** 2)) - self.disen_rho)
        soft_constraint_joint_y = 1000 * F.relu(
            torch.mean(0.5 * ((edim_outputs.disen_weight_y_joint - 1) ** 2)) - self.disen_rho)
        soft_constraint_margin_y = 1000 * F.relu(
            torch.mean(0.5 * ((edim_outputs.disen_weight_y_margin - 1) ** 2)) - self.disen_rho)

        loss_disen_x = -1 * self.ROTDM(
            T=edim_outputs.critic_x.detach(),
            T_prime=edim_outputs.critic_x_prime.detach(),
            weight_joint=edim_outputs.disen_weight_x_joint,
            weight_margin=edim_outputs.disen_weight_x_margin
        )

        loss_disen_y = -1 * self.ROTDM(
            T=edim_outputs.critic_y.detach(),
            T_prime=edim_outputs.critic_y_prime.detach(),
            weight_joint=edim_outputs.disen_weight_y_joint,
            weight_margin=edim_outputs.disen_weight_y_margin
        )

        loss_disen_weight_x = loss_disen_x + soft_constraint_joint_x + soft_constraint_margin_x
        loss_disen_weight_y = loss_disen_y + soft_constraint_joint_y + soft_constraint_margin_y
        loss_disen_weights = loss_disen_weight_x + loss_disen_weight_y
        return loss_disen_weights

    def compute_generator_loss(self, edim_outputs: EDIMOutputs, epoch) -> GenLosses:
        """Generator loss function

        Args:
            edim_outputs (EDIMOutputs): Output of the forward pass of the exclusive information model

        Returns:
            GenLosses: Generator losses
        """
        # Compute Global mutual loss
        global_mutual_loss_x = self.ROTDM(
            T=edim_outputs.global_mutual_M_R_x,
            T_prime=edim_outputs.global_mutual_M_R_x_prime,
            weight_joint=edim_outputs.weight_x_joint.detach(),
            weight_margin=edim_outputs.weight_x_margin.detach(),
        )
        global_mutual_loss_y = self.ROTDM(
            T=edim_outputs.global_mutual_M_R_y,
            T_prime=edim_outputs.global_mutual_M_R_y_prime,
            weight_joint=edim_outputs.weight_y_joint.detach(),
            weight_margin=edim_outputs.weight_y_margin.detach(),
        )

        global_mutual_loss_x = global_mutual_loss_x + edim_outputs.global_gradient_penalty_x
        global_mutual_loss_y = global_mutual_loss_y + edim_outputs.global_gradient_penalty_y

        global_mutual_loss = (global_mutual_loss_x + global_mutual_loss_y) * self.global_mutual_loss_coeff

        # Compute Local mutual loss
        local_mutual_loss_x = self.ROTDM(
            T=edim_outputs.local_mutual_M_R_x,
            T_prime=edim_outputs.local_mutual_M_R_x_prime,
            weight_joint=edim_outputs.weight_x_joint.unsqueeze(-1).unsqueeze(-1).detach(),
            weight_margin=edim_outputs.weight_x_margin.unsqueeze(-1).unsqueeze(-1).detach(),
        )
        local_mutual_loss_y = self.ROTDM(
            T=edim_outputs.local_mutual_M_R_y,
            T_prime=edim_outputs.local_mutual_M_R_y_prime,
            weight_joint=edim_outputs.weight_y_joint.unsqueeze(-1).unsqueeze(-1).detach(),
            weight_margin=edim_outputs.weight_y_margin.unsqueeze(-1).unsqueeze(-1).detach(),
        )

        local_mutual_loss_x = local_mutual_loss_x + edim_outputs.local_gradient_penalty_x
        local_mutual_loss_y = local_mutual_loss_y + edim_outputs.local_gradient_penalty_y

        local_mutual_loss = (local_mutual_loss_x + local_mutual_loss_y) * self.local_mutual_loss_coeff

        loss_disen_x = -1 * self.ROTDM(
            T=edim_outputs.critic_x,
            T_prime=edim_outputs.critic_x_prime,
            weight_joint=edim_outputs.disen_weight_x_joint.detach(),
            weight_margin=edim_outputs.disen_weight_x_margin.detach()
        )

        loss_disen_y = -1 * self.ROTDM(
            T=edim_outputs.critic_y,
            T_prime=edim_outputs.critic_y_prime,
            weight_joint=edim_outputs.disen_weight_y_joint.detach(),
            weight_margin=edim_outputs.disen_weight_y_margin.detach()
        )

        loss_disen_g = (loss_disen_x + loss_disen_y) * self.disentangling_loss_coeff

        # For each network, we assign a loss objective
        encoder_loss = global_mutual_loss + local_mutual_loss + loss_disen_g

        return GenLosses(
            encoder_loss=encoder_loss,
            local_mutual_loss=local_mutual_loss,
            global_mutual_loss=global_mutual_loss,
            loss_disen_g=loss_disen_g,
        )

    def compute_classif_loss(
            self,
            classif_outputs: ClassifierOutputs,
            card_labels: torch.Tensor,
            suit_labels: torch.Tensor,
    ) -> ClassifLosses:

        card_classif_loss, card_accuracy = self.classif_loss(
            y_pred=classif_outputs.card_logits, target=card_labels
        )

        suit_classif_loss, suit_accuracy = self.classif_loss(
            y_pred=classif_outputs.suit_logits, target=suit_labels
        )

        # Total classification loss
        classif_loss = card_classif_loss + suit_classif_loss

        return ClassifLosses(
            classif_loss=classif_loss,
            card_classif_loss=card_classif_loss,
            suit_classif_loss=suit_classif_loss,
            card_accuracy=card_accuracy,
            suit_accuracy=suit_accuracy,
        )