import torch
from torch import autograd, nn
from torch.nn import Parameter
import torch.nn.functional as F


def gradient_penalty(netD, real_data, fake_data, lamb):
    tensor_shape = real_data.shape
    device = real_data.device
    assert real_data.size(0) == fake_data.size(0)

    alpha = torch.rand(tensor_shape, device=real_data.device)

    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lamb
    return gradient_penalty

def gradient_penalty_representation(critic, rep_1, rep_2, lamb):
    tensor_shape = rep_1.shape
    device = rep_2.device
    batch_size = rep_1.size(0)
    # Start by computing an interpolated distribution between our two distribution
    # We give random weight to the distribution - eps
    eps = torch.rand(tensor_shape).to(device)
    eps = eps.expand_as(rep_1)
    interpolation = eps * rep_1 + (1 - eps) * rep_2

    # Pass the interpolation into the critic from which we want to retrieve and norm the gradients
    interp_logits = critic(interpolation)
    grad_outputs = torch.ones_like(interp_logits)

    # Computes and returns the sum of gradients of outputs with respect to the inputs.
    gradients = autograd.grad(
        outputs=interp_logits,
        inputs=interpolation,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, 1)
    return torch.mean((grad_norm - 1) ** 2) * lamb