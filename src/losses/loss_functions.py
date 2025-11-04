import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.custom_typing import GanLossOutput, Tuple


class ClassifLoss(nn.Module):
    """Classifier loss"""

    @staticmethod
    def accuracy(y_pred, target):
        return torch.sum(y_pred == target).float().mean()

    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def __call__(
            self, y_pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[float, float]:
        """Compute cross entropy loss

        Args:
            y_pred (torch.Tensor): Classifier prediction
            target (torch.Tensor): Ground truth

        Returns:
            Tuple[float, float]: Error and accuracy over the current batch
        """
        batch_size = y_pred.size(0)

        classif_error = self.cross_entropy(
            F.softmax(y_pred, dim=1), target.long()
        ).mean()
        accuracy = self.accuracy(y_pred=torch.argmax(y_pred, dim=1), target=target)
        return classif_error, accuracy / batch_size


class DJSLoss(nn.Module):
    """Jensen Shannon Divergence loss"""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, T: torch.Tensor, T_prime: torch.Tensor) -> float:
        """Estimator of the Jensen Shannon Divergence see paper equation (2)

        Args:
            T (torch.Tensor): Statistique network estimation from the marginal distribution P(x)P(z)
            T_prime (torch.Tensor): Statistique network estimation from the joint distribution P(xz)

        Returns:
            float: DJS estimation value
        """
        joint_expectation = (-F.softplus(-T)).mean()
        marginal_expectation = F.softplus(T_prime).mean()
        mutual_info = joint_expectation - marginal_expectation

        return -mutual_info


class DiscriminatorLoss(nn.Module):
    """Basic discriminator GAN loss """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> float:
        """Discriminator loss gan

        Args:
            real_logits (torch.Tensor): Sample from the real distribution here from P(Sx)P(Ex)
            fake_logits (torch.Tensor): Sample from the fake (generated) distribution here from P(SxEx)

        Returns:
            float: Discriminator loss value
        """

        # Discriminator should predict real logits as logits from the real distribution
        discriminator_real = F.binary_cross_entropy_with_logits(
            input=real_logits, target=torch.ones_like(real_logits)
        )
        # Discriminator should predict fake logits as logits from the generated distribution
        discriminator_fake = F.binary_cross_entropy_with_logits(
            input=fake_logits, target=torch.zeros_like(fake_logits)
        )
        discriminator_loss = discriminator_real.mean() + discriminator_fake.mean()

        return discriminator_loss


class GeneratorLoss(nn.Module):
    """Basic generator GAN loss """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, fake_logits: torch.Tensor) -> float:
        """Generator loss

        Args:
            fake_logits (torch.Tensor): Sample from the fake (generated) distribution here from P(SxEx)

        Returns:
            float: Generator loss value
        """
        # Discriminator should generate fake logits that fool the discriminator
        generator_loss = F.binary_cross_entropy_with_logits(
            input=fake_logits, target=torch.ones_like(fake_logits)
        )
        return generator_loss


class WDLoss(nn.Module):
    """Wasserstein Dependency Measure loss"""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, T: torch.Tensor, T_prime: torch.Tensor, weight=1.0) -> float:
        joint_expectation = torch.mean(T * weight)
        marginal_expectation = torch.mean(T_prime)
        w_d = joint_expectation - marginal_expectation
        return -w_d


EPS = 1e-6
class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()

        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = grad_output * input.exp().detach() / \
            (running_mean + EPS) / input.shape[0]
        return grad, None


def ema(mu, alpha, past_ema):
    return alpha * mu + (1.0 - alpha) * past_ema


def ema_loss(x, running_mean, alpha):
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = ema(t_exp, alpha, running_mean.item())
    t_log = EMALoss.apply(x, running_mean)

    # Recalculate ema

    return t_log, running_mean
