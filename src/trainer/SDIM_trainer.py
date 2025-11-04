import torch.optim as optim
import torch
from torch import nn
from src.losses.loss_functions import ClassifLoss
from src.models.SDIM import SDIM
from src.losses.SDIM_loss import SDIMLoss
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import mlflow
import mlflow.pytorch as mpy
from utils.custom_typing import SDIMOutputs, SDIMLosses

def freeze_grad_and_eval(model: nn.Module):
    """Freeze a given network, disable batch norm and dropout layers

    Args:
        model (nn.Module): [Pretrained shared encoder]
    """
    for param in model.parameters():
        param.requires_grad = False
    model.eval()


def unfreeze_grad_and_train(model: nn.Module):
    """Unfreeze a given network, enable batch norm and dropout layers

    Args:
        model (nn.Module): [Pretrained shared encoder]
    """
    for param in model.parameters():
        param.requires_grad = True
    model.train()


class SDIMTrainer:
    def __init__(
            self,
            model: SDIM,
            loss: SDIMLoss,
            dataset_train: Dataset,
            dataset_test: Dataset,
            learning_rate: float,
            batch_size: int,
            device: str,
    ):
        """Shared Deep Info Max trainer

        Args:
            model (SDIM): Shared model backbone
            loss (SDIMLoss): Shared loss
            dataset_train (Dataset): Train dataset
            learning_rate (float): Learning rate
            batch_size (int): Batch size
            device (str): Device among cuda/cpu
        """
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.model = model.to(device)

        self.loss = loss
        self.classif_loss = ClassifLoss()
        self.device = device

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.eps = 0.0001

        betas = (0.9, 0.999)
        epsilon = 1e-8

        self.optimizerW_joint = optim.Adam(self.model.netW_joint.parameters(), lr=0.00001,
                                           betas=(0.5, 0.9))
        self.optimizerW_margin = optim.Adam(self.model.netW_margin.parameters(), lr=0.00001,
                                            betas=(0.5, 0.9))

        # Network optimizers
        self.optimizer_encoder_x = optim.Adam(
            model.sh_enc_x.parameters(), lr=self.learning_rate, betas=betas, eps=epsilon
        )
        self.optimizer_encoder_y = optim.Adam(
            model.sh_enc_y.parameters(), lr=self.learning_rate, betas=betas, eps=epsilon
        )
        self.optimizer_local_stat_x = optim.Adam(
            model.local_stat_x.parameters(), lr=self.learning_rate, betas=betas,
            eps=epsilon
        )
        self.optimizer_local_stat_y = optim.Adam(
            model.local_stat_y.parameters(), lr=self.learning_rate, betas=betas,
            eps=epsilon
        )
        self.optimizer_global_stat_x = optim.Adam(
            model.global_stat_x.parameters(), lr=self.learning_rate, betas=betas,
            eps=epsilon
        )
        self.optimizer_global_stat_y = optim.Adam(
            model.global_stat_y.parameters(), lr=self.learning_rate, betas=betas,
            eps=epsilon
        )

        self.optimizer_card_classifier = optim.Adam(
            self.model.card_classifier.parameters(), lr=0.0001, betas=betas, eps=epsilon
        )
        self.optimizer_suit_classifier = optim.Adam(
            self.model.suit_classifier.parameters(), lr=0.0001, betas=betas, eps=epsilon
        )

    def gradient_zero(self):
        """Set all the networks gradient to zero"""
        self.optimizer_encoder_x.zero_grad()
        self.optimizer_encoder_y.zero_grad()

        self.optimizer_local_stat_x.zero_grad()
        self.optimizer_local_stat_y.zero_grad()

        self.optimizer_global_stat_x.zero_grad()
        self.optimizer_global_stat_y.zero_grad()

        self.optimizer_card_classifier.zero_grad()
        self.optimizer_suit_classifier.zero_grad()

    def compute_gradient(
            self,
            sdim_output: SDIMOutputs,
            card_labels: torch.Tensor,
            suit_labels: torch.Tensor,
    ) -> SDIMLosses:
        """Compute the SDIM gradient

        Args:
            sdim_output (SDIMOutputs): Shared model outputs

        Returns:
            SDIMLosses: [Shared model losses value]
        """
        losses = self.loss(
            sdim_outputs=sdim_output,
            card_labels=card_labels,
            suit_labels=suit_labels,
        )
        losses.total_loss.backward()
        return losses

    def gradient_step(self):
        """Make an optimisation step for all the networks"""
        self.optimizer_encoder_x.step()
        self.optimizer_encoder_y.step()

        self.optimizer_local_stat_x.step()
        self.optimizer_local_stat_y.step()

        self.optimizer_global_stat_x.step()
        self.optimizer_global_stat_y.step()

        self.optimizer_card_classifier.step()
        self.optimizer_suit_classifier.step()

    def clone_sdim_outputs(self, sdim_outputs, weight_x_joint, weight_x_margin, weight_y_joint, weight_y_margin):
        """克隆 sdim_outputs，除权重外全部 detach()"""
        return SDIMOutputs(
            global_mutual_M_R_x=sdim_outputs.global_mutual_M_R_x.detach(),
            global_mutual_M_R_x_prime=sdim_outputs.global_mutual_M_R_x_prime.detach(),
            global_mutual_M_R_y=sdim_outputs.global_mutual_M_R_y.detach(),
            global_mutual_M_R_y_prime=sdim_outputs.global_mutual_M_R_y_prime.detach(),
            local_mutual_M_R_x=sdim_outputs.local_mutual_M_R_x.detach(),
            local_mutual_M_R_x_prime=sdim_outputs.local_mutual_M_R_x_prime.detach(),
            local_mutual_M_R_y=sdim_outputs.local_mutual_M_R_y.detach(),
            local_mutual_M_R_y_prime=sdim_outputs.local_mutual_M_R_y_prime.detach(),
            global_M_R_x=sdim_outputs.global_M_R_x.detach(),
            global_M_R_x_prime=sdim_outputs.global_M_R_x_prime.detach(),
            global_M_R_y=sdim_outputs.global_M_R_y.detach(),
            global_M_R_y_prime=sdim_outputs.global_M_R_y_prime.detach(),
            shared_x=sdim_outputs.shared_x.detach(),
            shared_y=sdim_outputs.shared_y.detach(),
            global_gradient_penalty_x=sdim_outputs.global_gradient_penalty_x.detach(),
            global_gradient_penalty_y=sdim_outputs.global_gradient_penalty_y.detach(),
            local_gradient_penalty_x=sdim_outputs.local_gradient_penalty_x.detach(),
            local_gradient_penalty_y=sdim_outputs.local_gradient_penalty_y.detach(),
            weight_x_joint=weight_x_joint,
            weight_y_joint=weight_y_joint,
            weight_x_margin=weight_x_margin,
            weight_y_margin=weight_y_margin,
            card_logits=sdim_outputs.card_logits.detach(),
            suit_logits=sdim_outputs.suit_logits.detach(),
            input_x=sdim_outputs.input_x.detach(),
            input_x_prime=sdim_outputs.input_x_prime.detach(),
            input_y=sdim_outputs.input_y.detach(),
            input_y_prime=sdim_outputs.input_y_prime.detach(),
        )

    def update_weights(self, sdim_outputs, log_step):
        self.optimizerW_joint.zero_grad()
        self.optimizerW_margin.zero_grad()

        weight_x_joint, weight_x_margin = self.model.compute_weights(
            sdim_outputs.input_x.detach(), sdim_outputs.input_x_prime.detach()
        )
        weight_y_joint, weight_y_margin = self.model.compute_weights(
            sdim_outputs.input_y.detach(), sdim_outputs.input_y_prime.detach()
        )

        cloned_outputs = self.clone_sdim_outputs(
            sdim_outputs, weight_x_joint, weight_x_margin, weight_y_joint, weight_y_margin
        )

        loss_weights = self.loss.weight_loss(cloned_outputs)
        loss_weights.backward()

        self.optimizerW_joint.step()
        self.optimizerW_margin.step()

        mlflow.log_metrics({"loss_weights": loss_weights.item()}, step=log_step)


    def train(self, epochs, xp_name="test", dataset_val=None):
        """Trained shared model and log losses and accuracy on Mlflow.

        Args:
            epochs (int): Number of epochs
            xp_name (str, optional): Name of the Mlfow experiment. Defaults to "test".
        """
        mlflow.set_experiment(experiment_name=xp_name)
        with mlflow.start_run() as run:
            params = {
                "Learning rate": self.learning_rate,
                "Local mutual weight": self.loss.local_mutual_loss_coeff,
                "Global mutual weight": self.loss.global_mutual_loss_coeff,
                "L1 weight": self.loss.shared_loss_coeff,
            }
            mlflow.log_dict(params, "parameters.json")

            train_dataloader = DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, drop_last=True)
            log_step = 0
            for epoch in tqdm(range(epochs)):
                cnt = None
                for idx, train_batch in enumerate(train_dataloader):
                    sample = train_batch
                    self.gradient_zero()
                    sdim_outputs = self.model(
                        x=sample.x.to(self.device), y=sample.y.to(self.device)
                    )

                    losses = self.compute_gradient(
                        sdim_output=sdim_outputs,
                        card_labels=sample.card_y.to(self.device),
                        suit_labels=sample.suit_y.to(self.device),
                    )
                    
                    dict_losses = losses._asdict()
                    mlflow.log_metrics(
                        {k: v.item() for k, v in dict_losses.items()}, step=log_step
                    )
                    
                    self.gradient_step()

                    for _ in range(2):
                        self.update_weights(sdim_outputs, log_step)

                    log_step += 1

                print(losses.card_accuracy.item(),
                      losses.suit_accuracy.item(),
                      losses.total_loss.item())

                print("share", sdim_outputs.shared_y[:])

            encoder_x_path, encoder_y_path = "sh_encoder_x", "sh_encoder_y"
            mpy.log_state_dict(self.model.sh_enc_x.state_dict(), encoder_x_path)
            mpy.log_state_dict(self.model.sh_enc_y.state_dict(), encoder_y_path)

            freeze_grad_and_eval(self.model)
