import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from src.models.EDIM import EDIM
from src.utils.custom_typing import (
    ClassifLosses,
    ClassifierOutputs,
    DiscrLosses,
    GenLosses,
    GeneratorOutputs,
    DiscriminatorOutputs,
    EDIMOutputs,
)
from src.losses.EDIM_loss import EDIMLoss
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torch.utils.data import Dataset
import mlflow
import mlflow.pytorch as mpy


def freeze_grad_and_eval(model: nn.Module):
    """Freeze a given network, disable batch norm and dropout layers

    Args:
        model (nn.Module): [Pretrained shared encoder]
    """
    for param in model.parameters():
        param.requires_grad = False
    model.eval()


class EDIMTrainer:
    """Exclusive Deep Info Max trainer

    Args:
        model (EDIM): Exclusive model backbone
        loss (EDIMLoss): Exclusive loss
        dataset_train (Dataset): Train dataset
        learning_rate (float): Learning rate
        batch_size (int): Batch size
        device (str): Device among cuda/cpu
    """

    def __init__(
            self,
            model: EDIM,
            loss: EDIMLoss,
            dataset_train: Dataset,
            learning_rate: float,
            batch_size: int,
            device: str,
    ):

        self.dataset_train = dataset_train

        self.model = model.to(device)
        self.loss = loss
        self.device = device

        self.learning_rate = learning_rate
        self.batch_size = batch_size

        betas = (0.9, 0.999)
        epsilon = 1e-8

        self.optimizerW_joint = optim.Adam(self.model.netW_joint.parameters(), lr=0.00001,
                                           betas=(0.5, 0.9))
        self.optimizerW_margin = optim.Adam(self.model.netW_margin.parameters(), lr=0.00001,
                                            betas=(0.5, 0.9))

        self.optimizer_disenW_joint = optim.Adam(self.model.disen_netW_joint.parameters(), lr=0.00001,
                                                 betas=(0.5, 0.9))
        self.optimizer_disenW_margin = optim.Adam(self.model.disen_netW_margin.parameters(), lr=0.00001,
                                                  betas=(0.5, 0.9))

        self.optimizer_encoder_x = optim.Adam(
            model.ex_enc_x.parameters(), lr=learning_rate, betas=betas, eps=epsilon
        )
        self.optimizer_encoder_y = optim.Adam(
            model.ex_enc_y.parameters(), lr=learning_rate, betas=betas, eps=epsilon
        )
        self.optimizer_local_stat_x = optim.Adam(
            model.local_stat_x.parameters(), lr=learning_rate, betas=betas, eps=epsilon
        )
        self.optimizer_local_stat_y = optim.Adam(
            model.local_stat_y.parameters(), lr=learning_rate, betas=betas, eps=epsilon
        )
        self.optimizer_global_stat_x = optim.Adam(
            model.global_stat_x.parameters(), lr=learning_rate, betas=betas, eps=epsilon
        )
        self.optimizer_global_stat_y = optim.Adam(
            model.global_stat_y.parameters(), lr=learning_rate, betas=betas, eps=epsilon
        )

        self.optimizer_card_classifier = optim.Adam(
            self.model.card_classifier.parameters(), lr=0.0001, betas=betas, eps=epsilon
        )
        self.optimizer_suit_classifier = optim.Adam(
            self.model.suit_classifier.parameters(), lr=0.0001, betas=betas, eps=epsilon
        )

        self.optimizer_critic_x = optim.Adam(self.model.critic_x.parameters(), lr=0.0008,
                                                    betas=(0.9, 0.999), weight_decay=0.0005)
        self.optimizer_critic_y = optim.Adam(self.model.critic_y.parameters(), lr=0.0008,
                                                    betas=(0.9, 0.999), weight_decay=0.0005)

    def update_generator(self, edim_outputs: EDIMOutputs, epoch) -> GenLosses:
        """Compute the generator gradient and make an optimisation step

        Args:
            edim_outputs (EDIMOutputs): Exclusive model outputs

        Returns:

            GenLosses: Generator losses
        """
        self.optimizer_encoder_x.zero_grad()
        self.optimizer_encoder_y.zero_grad()

        self.optimizer_local_stat_x.zero_grad()
        self.optimizer_local_stat_y.zero_grad()

        self.optimizer_global_stat_x.zero_grad()
        self.optimizer_global_stat_y.zero_grad()

        losses = self.loss.compute_generator_loss(
            edim_outputs=edim_outputs, epoch=epoch
        )
        losses.encoder_loss.backward()

        self.optimizer_encoder_x.step()
        self.optimizer_encoder_y.step()

        self.optimizer_local_stat_x.step()
        self.optimizer_local_stat_y.step()

        self.optimizer_global_stat_x.step()
        self.optimizer_global_stat_y.step()
        return losses

    def log_metrics_dict(self, prefix, metrics_dict, step):
        mlflow.log_metrics({f"{prefix}{k}": v.item() for k, v in metrics_dict.items()}, step=step)

    def clone_edim_outputs_for_weights(self, edim_outputs, weight_x_joint, weight_x_margin, weight_y_joint, weight_y_margin):
        """Clone edim_outputs; only weight_* participates in gradient computation."""
        return EDIMOutputs(
            global_mutual_M_R_x=edim_outputs.global_mutual_M_R_x.detach(),
            global_mutual_M_R_x_prime=edim_outputs.global_mutual_M_R_x_prime.detach(),
            global_mutual_M_R_y=edim_outputs.global_mutual_M_R_y.detach(),
            global_mutual_M_R_y_prime=edim_outputs.global_mutual_M_R_y_prime.detach(),
            local_mutual_M_R_x=edim_outputs.local_mutual_M_R_x.detach(),
            local_mutual_M_R_x_prime=edim_outputs.local_mutual_M_R_x_prime.detach(),
            local_mutual_M_R_y=edim_outputs.local_mutual_M_R_y.detach(),
            local_mutual_M_R_y_prime=edim_outputs.local_mutual_M_R_y_prime.detach(),
            global_gradient_penalty_x=edim_outputs.global_gradient_penalty_x.detach(),
            global_gradient_penalty_y=edim_outputs.global_gradient_penalty_y.detach(),
            local_gradient_penalty_x=edim_outputs.local_gradient_penalty_x.detach(),
            local_gradient_penalty_y=edim_outputs.local_gradient_penalty_y.detach(),
            weight_x_joint=weight_x_joint,
            weight_y_joint=weight_y_joint,
            weight_x_margin=weight_x_margin,
            weight_y_margin=weight_y_margin,
            input_x=edim_outputs.input_x.detach(),
            input_x_prime=edim_outputs.input_x_prime.detach(),
            input_y=edim_outputs.input_y.detach(),
            input_y_prime=edim_outputs.input_y_prime.detach(),
            shared_x=edim_outputs.shared_x.detach(),
            shared_y=edim_outputs.shared_y.detach(),
            exclusive_x=edim_outputs.exclusive_x.detach(),
            exclusive_y=edim_outputs.exclusive_y.detach(),
            representation_x_joint=edim_outputs.representation_x_joint.detach(),
            representation_y_joint=edim_outputs.representation_y_joint.detach(),
            representation_x_margin=edim_outputs.representation_x_margin.detach(),
            representation_y_margin=edim_outputs.representation_y_margin.detach(),
            critic_x=edim_outputs.critic_x.detach(),
            critic_y=edim_outputs.critic_y.detach(),
            critic_x_prime=edim_outputs.critic_x_prime.detach(),
            critic_y_prime=edim_outputs.critic_y_prime.detach(),
            disen_weight_x_joint=edim_outputs.disen_weight_x_joint.detach(),
            disen_weight_y_joint=edim_outputs.disen_weight_y_joint.detach(),
            disen_weight_x_margin=edim_outputs.disen_weight_x_margin.detach(),
            disen_weight_y_margin=edim_outputs.disen_weight_y_margin.detach(),
        )

    def clone_edim_outputs_for_disen(self, edim_outputs, disen_weight_x_joint, disen_weight_x_margin, disen_weight_y_joint,
                                     disen_weight_y_margin):
        """Clone edim_outputs; only disen_weight_* participates in gradient computation."""
        return EDIMOutputs(
            global_mutual_M_R_x=edim_outputs.global_mutual_M_R_x.detach(),
            global_mutual_M_R_x_prime=edim_outputs.global_mutual_M_R_x_prime.detach(),
            global_mutual_M_R_y=edim_outputs.global_mutual_M_R_y.detach(),
            global_mutual_M_R_y_prime=edim_outputs.global_mutual_M_R_y_prime.detach(),
            local_mutual_M_R_x=edim_outputs.local_mutual_M_R_x.detach(),
            local_mutual_M_R_x_prime=edim_outputs.local_mutual_M_R_x_prime.detach(),
            local_mutual_M_R_y=edim_outputs.local_mutual_M_R_y.detach(),
            local_mutual_M_R_y_prime=edim_outputs.local_mutual_M_R_y_prime.detach(),
            global_gradient_penalty_x=edim_outputs.global_gradient_penalty_x.detach(),
            global_gradient_penalty_y=edim_outputs.global_gradient_penalty_y.detach(),
            local_gradient_penalty_x=edim_outputs.local_gradient_penalty_x.detach(),
            local_gradient_penalty_y=edim_outputs.local_gradient_penalty_y.detach(),
            weight_x_joint=edim_outputs.weight_x_joint.detach(),
            weight_y_joint=edim_outputs.weight_y_joint.detach(),
            weight_x_margin=edim_outputs.weight_x_margin.detach(),
            weight_y_margin=edim_outputs.weight_y_margin.detach(),
            input_x=edim_outputs.input_x.detach(),
            input_x_prime=edim_outputs.input_x_prime.detach(),
            input_y=edim_outputs.input_y.detach(),
            input_y_prime=edim_outputs.input_y_prime.detach(),
            shared_x=edim_outputs.shared_x.detach(),
            shared_y=edim_outputs.shared_y.detach(),
            exclusive_x=edim_outputs.exclusive_x.detach(),
            exclusive_y=edim_outputs.exclusive_y.detach(),
            representation_x_joint=edim_outputs.representation_x_joint.detach(),
            representation_y_joint=edim_outputs.representation_y_joint.detach(),
            representation_x_margin=edim_outputs.representation_x_margin.detach(),
            representation_y_margin=edim_outputs.representation_y_margin.detach(),
            critic_x=edim_outputs.critic_x.detach(),
            critic_y=edim_outputs.critic_y.detach(),
            critic_x_prime=edim_outputs.critic_x_prime.detach(),
            critic_y_prime=edim_outputs.critic_y_prime.detach(),
            disen_weight_x_joint=disen_weight_x_joint,
            disen_weight_y_joint=disen_weight_y_joint,
            disen_weight_x_margin=disen_weight_x_margin,
            disen_weight_y_margin=disen_weight_y_margin,
        )

    def update_weights(self, edim_outputs, log_step):
        self.optimizerW_joint.zero_grad()
        self.optimizerW_margin.zero_grad()

        weight_x_joint, weight_x_margin = self.model.compute_real_weights(
            edim_outputs.input_x.detach(), edim_outputs.input_x_prime.detach()
        )
        weight_y_joint, weight_y_margin = self.model.compute_real_weights(
            edim_outputs.input_y.detach(), edim_outputs.input_y_prime.detach()
        )

        cloned_outputs = self.clone_edim_outputs_for_weights(
            edim_outputs, weight_x_joint, weight_x_margin, weight_y_joint, weight_y_margin
        )

        loss_weights = self.loss.weight_loss(cloned_outputs)
        loss_weights.backward()

        self.optimizerW_joint.step()
        self.optimizerW_margin.step()

        mlflow.log_metrics({"loss_weights": loss_weights.item()}, step=log_step)

    def update_disen_weights(self, edim_outputs, log_step):
        self.optimizer_disenW_joint.zero_grad()
        self.optimizer_disenW_margin.zero_grad()

        disen_weight_x_joint, disen_weight_x_margin = self.model.compute_real_disen_weights(
            edim_outputs.representation_x_joint.detach(), edim_outputs.representation_x_margin.detach()
        )
        disen_weight_y_joint, disen_weight_y_margin = self.model.compute_real_disen_weights(
            edim_outputs.representation_y_joint.detach(), edim_outputs.representation_y_margin.detach()
        )

        cloned_outputs = self.clone_edim_outputs_for_disen(
            edim_outputs, disen_weight_x_joint, disen_weight_x_margin, disen_weight_y_joint, disen_weight_y_margin
        )
        disen_loss_weights = self.loss.disen_weight_loss(cloned_outputs)
        disen_loss_weights.backward()

        self.optimizer_disenW_joint.step()
        self.optimizer_disenW_margin.step()

        mlflow.log_metrics({"disen_loss_weights": disen_loss_weights.item()}, step=log_step)


    def update_classifier(
            self,
            classif_outputs: ClassifierOutputs,
            card_labels: torch.Tensor,
            suit_labels: torch.Tensor,
    ) -> ClassifLosses:
        self.optimizer_card_classifier.zero_grad()
        self.optimizer_suit_classifier.zero_grad()

        losses = self.loss.compute_classif_loss(
            classif_outputs=classif_outputs,
            card_labels=card_labels,
            suit_labels=suit_labels,
        )

        losses.classif_loss.backward()

        self.optimizer_card_classifier.step()
        self.optimizer_suit_classifier.step()

        return losses
        
        
    def update_dis_critic(
            self,
            edim_outputs: EDIMOutputs,
    ) -> DiscrLosses:

        self.optimizer_critic_x.zero_grad()
        self.optimizer_critic_y.zero_grad()

        dis_loss = self.model.forward_dis_critic(edim_outputs=edim_outputs)
        dis_loss.loss_dis_d.backward()

        self.optimizer_critic_x.step()
        self.optimizer_critic_y.step()

        return dis_loss


    def train(self, epochs: int, xp_name: str = "test"):
        """Trained excluvise model and log losses and accuracy on Mlflow.

        Args:
            epochs (int): Number of epochs
            xp_name (str, optional): Name of the Mlfow experiment. Defaults to "test".
        """
        mlflow.set_experiment(experiment_name=xp_name)
        with mlflow.start_run() as run:
            mlflow.log_param("Batch size", self.batch_size)
            mlflow.log_param("Learning rate", self.learning_rate)
            mlflow.log_param("Local mutual weight", self.loss.local_mutual_loss_coeff)
            mlflow.log_param("Global mutual weight", self.loss.global_mutual_loss_coeff)
            mlflow.log_param("Exclusive weight", self.loss.disentangling_loss_coeff)
            log_step = 0

            train_dataloader = DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, drop_last=True)
            for epoch in tqdm(range(epochs)):
                for idx, train_batch in enumerate(train_dataloader):
                    sample = train_batch
                    edim_outputs = self.model.forward_generator(
                        x=sample.x.to(self.device), y=sample.y.to(self.device)
                    )

                    gen_losses = self.update_generator(edim_outputs=edim_outputs, epoch=epoch)

                    for _ in range(10):
                        dis_loss = self.update_dis_critic(
                            edim_outputs=edim_outputs
                        )

                    for _ in range(2):
                        self.update_weights(edim_outputs, log_step)

                    for _ in range(10):
                        self.update_disen_weights(edim_outputs, log_step)

                    classif_outputs = self.model.forward_classifier(
                        edim_outputs=edim_outputs
                    )
                    classif_losses = self.update_classifier(
                        classif_outputs=classif_outputs,
                        card_labels=sample.card_y.to(self.device),
                        suit_labels=sample.suit_y.to(self.device),
                    )

                    dict_gen_losses = gen_losses._asdict()
                    mlflow.log_metrics(
                        {k: v.item() for k, v in dict_gen_losses.items()}, step=log_step
                    )

                    dict_classif_losses = classif_losses._asdict()
                    mlflow.log_metrics(
                        {k: v.item() for k, v in dict_classif_losses.items()},
                        step=log_step,
                    )

                    dict_dis_loss = dis_loss._asdict()
                    mlflow.log_metrics(
                        {k: v.item() for k, v in dict_dis_loss.items()},
                        step=log_step,
                    )

                    log_step += 1

                print("card accuracy:", classif_losses.card_accuracy.item(), "suit accuracy:", classif_losses.suit_accuracy.item())
                print("total loss:", gen_losses.encoder_loss.item())

            encoder_x_path, encoder_y_path = "ex_encoder_x", "ex_encoder_y"
            mpy.log_state_dict(self.model.ex_enc_x.state_dict(), encoder_x_path)
            mpy.log_state_dict(self.model.ex_enc_y.state_dict(), encoder_y_path)
