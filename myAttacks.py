from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np


def fgsm_attack(model, data, target, epsilon):
    # Set requires_grad attribute of tensor. Important for Attack
    data.requires_grad = True

    # Forward pass the data through the model
    output = model(data)

    # init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
    # # If the initial prediction is wrong, dont bother attacking, just move on
    # if init_pred.item() != target.item():
    #     continue

    # Calculate the loss
    loss = F.nll_loss(output, target)

    # Zero all existing gradients
    model.zero_grad()

    # Calculate gradients of model in backward pass
    loss.backward()

    # Collect datagrad
    data_grad = data.grad.data


    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = data + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1).detach()
    # Return the perturbed image
    return perturbed_image

def pgd_attack(model, data, target, epsilon, alpha=2/255, iters=4):
    ori_data=data.clone()
    for i in range(iters):
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()

        perturbed_image=data+alpha*sign_data_grad
        eta=torch.clamp(perturbed_image-ori_data,min=-epsilon,max=epsilon)
        data=torch.clamp(ori_data+eta,0,1).detach_()

    # Return the perturbed image
    return data


def cw_attack():
    pass

def bim_attack():
    pass

def jsma_attack():
    pass