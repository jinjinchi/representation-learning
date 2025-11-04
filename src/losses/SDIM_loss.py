import torch
import torch.nn as nn
from src.losses.loss_functions import DJSLoss, ClassifLoss, WDLoss
from src.utils.custom_typing import SDIMLosses, SDIMOutputs, WeightNetOutputs
import torch.nn.functional as F


class SDIMLoss(nn.Module):
    """Loss function to extract shared information from the image, see paper

    Args:
        local_mutual_loss_coeff (float): Coefficient of the local ROTDM loss
        global_mutual_loss_coeff (float): Coefficient of the global ROTDM loss
        shared_loss_coeff (float): Coefficient of L1 loss, see paper
    """

    def __init__(
            self,
            local_mutual_loss_coeff: float,
            global_mutual_loss_coeff: float,
            shared_loss_coeff: float,
    ):

        super().__init__()
        self.local_mutual_loss_coeff = local_mutual_loss_coeff
        self.global_mutual_loss_coeff = global_mutual_loss_coeff
        self.shared_loss_coeff = shared_loss_coeff

        self.WPC_loss = WDLoss()
        self.classif_loss = ClassifLoss()
        self.l1_loss = nn.L1Loss()

        self.rho = 0.03

    def ROTDM(self, T: torch.Tensor, T_prime: torch.Tensor, weight_joint, weight_margin):
        joint_expectation = torch.mean(T * weight_joint)
        marginal_expectation = torch.mean(T_prime * weight_margin)
        w_d = joint_expectation - marginal_expectation
        return -w_d

    def weight_loss(self, sdim_outputs: SDIMOutputs):
        # Chi-squared
        soft_constraint_joint_x = 100 * F.relu(
            torch.mean(0.5 * ((sdim_outputs.weight_x_joint - 1) ** 2)) - self.rho)
        soft_constraint_margin_x = 100 * F.relu(
            torch.mean(0.5 * ((sdim_outputs.weight_x_margin - 1) ** 2)) - self.rho)
        soft_constraint_joint_y = 100 * F.relu(
            torch.mean(0.5 * ((sdim_outputs.weight_y_joint - 1) ** 2)) - self.rho)
        soft_constraint_margin_y = 100 * F.relu(
            torch.mean(0.5 * ((sdim_outputs.weight_y_margin - 1) ** 2)) - self.rho)
        # print(np.round(sdim_outputs.weight_x_joint.detach().cpu().numpy(), 2))
        # print(soft_constraint_joint_x+soft_constraint_margin_x+soft_constraint_joint_y+soft_constraint_margin_y)
        global_mutual_loss_x = -1 * self.ROTDM(
            T=sdim_outputs.global_mutual_M_R_x.detach(),
            T_prime=sdim_outputs.global_mutual_M_R_x_prime.detach(),
            weight_joint=sdim_outputs.weight_x_joint,
            weight_margin=sdim_outputs.weight_x_margin,
        )
        global_mutual_loss_y = -1 * self.ROTDM(
            T=sdim_outputs.global_mutual_M_R_y.detach(),
            T_prime=sdim_outputs.global_mutual_M_R_y_prime.detach(),
            weight_joint=sdim_outputs.weight_y_joint,
            weight_margin=sdim_outputs.weight_y_margin,
        )
        local_mutual_loss_x = -1 * self.ROTDM(
            T=sdim_outputs.local_mutual_M_R_x.detach(),
            T_prime=sdim_outputs.local_mutual_M_R_x_prime.detach(),
            weight_joint=sdim_outputs.weight_x_joint.unsqueeze(-1).unsqueeze(-1),
            weight_margin=sdim_outputs.weight_x_margin.unsqueeze(-1).unsqueeze(-1),
        )
        local_mutual_loss_y = -1 * self.ROTDM(
            T=sdim_outputs.local_mutual_M_R_y.detach(),
            T_prime=sdim_outputs.local_mutual_M_R_y_prime.detach(),
            weight_joint=sdim_outputs.weight_y_joint.unsqueeze(-1).unsqueeze(-1),
            weight_margin=sdim_outputs.weight_y_margin.unsqueeze(-1).unsqueeze(-1),
        )

        loss_weight_x = global_mutual_loss_x + local_mutual_loss_x + soft_constraint_joint_x + soft_constraint_margin_x
        loss_weight_y = global_mutual_loss_y + local_mutual_loss_y + soft_constraint_joint_y + soft_constraint_margin_y
        loss_weights = loss_weight_x + loss_weight_y
        return loss_weights

    def __call__(
            self,
            sdim_outputs: SDIMOutputs,
            card_labels: torch.Tensor,
            suit_labels: torch.Tensor,
    ) -> SDIMLosses:
        """Compute all the loss functions needed to extract the shared part

        Args:
            sdim_outputs (SDIMOutputs): Output of the forward pass of the shared information model

        Returns:
            SDIMLosses: Shared information losses
        """
        # Compute Global mutual loss
        global_mutual_loss_x = self.ROTDM(
            T=sdim_outputs.global_mutual_M_R_x,
            T_prime=sdim_outputs.global_mutual_M_R_x_prime,
            weight_joint=sdim_outputs.weight_x_joint.detach(),
            weight_margin=sdim_outputs.weight_x_margin.detach(),
        )
        global_mutual_loss_y = self.ROTDM(
            T=sdim_outputs.global_mutual_M_R_y,
            T_prime=sdim_outputs.global_mutual_M_R_y_prime,
            weight_joint=sdim_outputs.weight_y_joint.detach(),
            weight_margin=sdim_outputs.weight_y_margin.detach(),
        )

        global_mutual_loss_x = global_mutual_loss_x + sdim_outputs.global_gradient_penalty_x
        global_mutual_loss_y = global_mutual_loss_y + sdim_outputs.global_gradient_penalty_y

        global_mutual_loss = (global_mutual_loss_x + global_mutual_loss_y) * self.global_mutual_loss_coeff

        # Compute Local mutual loss
        local_mutual_loss_x = self.ROTDM(
            T=sdim_outputs.local_mutual_M_R_x,
            T_prime=sdim_outputs.local_mutual_M_R_x_prime,
            weight_joint=sdim_outputs.weight_x_joint.unsqueeze(-1).unsqueeze(-1).detach(),
            weight_margin=sdim_outputs.weight_x_margin.unsqueeze(-1).unsqueeze(-1).detach(),
        )
        local_mutual_loss_y = self.ROTDM(
            T=sdim_outputs.local_mutual_M_R_y,
            T_prime=sdim_outputs.local_mutual_M_R_y_prime,
            weight_joint=sdim_outputs.weight_y_joint.unsqueeze(-1).unsqueeze(-1).detach(),
            weight_margin=sdim_outputs.weight_y_margin.unsqueeze(-1).unsqueeze(-1).detach(),
        )

        local_mutual_loss_x = local_mutual_loss_x + sdim_outputs.local_gradient_penalty_x
        local_mutual_loss_y = local_mutual_loss_y + sdim_outputs.local_gradient_penalty_y

        local_mutual_loss = (local_mutual_loss_x + local_mutual_loss_y) * self.local_mutual_loss_coeff

        # Compute L1 on shared features
        shared_loss = self.l1_loss(sdim_outputs.shared_x, sdim_outputs.shared_y)
        shared_loss = shared_loss * self.shared_loss_coeff

        card_classif_loss, card_accuracy = self.classif_loss(
            y_pred=sdim_outputs.card_logits, target=card_labels
        )

        suit_classif_loss, suit_accuracy = self.classif_loss(
            y_pred=sdim_outputs.suit_logits, target=suit_labels
        )

        encoder_loss = global_mutual_loss + local_mutual_loss + shared_loss

        total_loss = (
                global_mutual_loss
                + local_mutual_loss
                + shared_loss
                + card_classif_loss
                + suit_classif_loss
        )

        return SDIMLosses(
            total_loss=total_loss,
            encoder_loss=encoder_loss,
            local_mutual_loss=local_mutual_loss,
            global_mutual_loss=global_mutual_loss,
            shared_loss=shared_loss,
            card_classif_loss=card_classif_loss,
            suit_classif_loss=suit_classif_loss,
            card_accuracy=card_accuracy,
            suit_accuracy=suit_accuracy,
        )