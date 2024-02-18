import os

import numpy as np
import math

import torch
import torch.optim as optim


def save_model(model, optim, iter, name):
    torch.save({
        'iter': iter,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
    }, os.path.join(name, "iter_{}.pth.tar".format(iter)))


def load_model(model, optim, path):
    if torch.cuda.is_available():
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iter']
    return iteration


def load_model_inference(model, optim, path):
    if torch.cuda.is_available():
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=torch.device('cpu'))

    # Check if any key in the state_dict contains the prefix "module."
    contains_module_prefix = any(key.startswith('module.') for key in checkpoint['model_state_dict'].keys())
    if contains_module_prefix:
        pretrained_dict = {key.replace("module.", ""): value for key, value in checkpoint['model_state_dict'].items()}
    else:
        pretrained_dict = checkpoint['model_state_dict']

    model.load_state_dict(pretrained_dict)
    if optim is not None:
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iter']
    return iteration

def optimizer_choice(method, y_cur, z_cur, lr=1e-4, momentum=0):
    if method == "adam":
        optimizer = optim.Adam([y_cur, z_cur], lr=lr)
    elif method == "sgd":
        if momentum == 0:
            optimizer = optim.SGD([y_cur, z_cur], lr=lr)
        else:
            optimizer = optim.SGD([y_cur, z_cur], lr=lr, momentum=momentum)
    elif method == 'LBFGS':
        optimizer = optim.LBFGS([y_cur, z_cur], lr=lr)
    else:
        raise ValueError(f"Method: {method} is not implemented!!!")
    return optimizer


def custom_learning_rate(optimizer, iteration, custom_lr):
    if iteration < custom_lr['start_iters']:
        lr = custom_lr['start_lr']
    elif iteration < custom_lr['start_iters'] + custom_lr['warmup_steps']:
        progress = (iteration - custom_lr['start_iters']) / custom_lr['warmup_steps']
        lr = custom_lr['start_lr'] - progress * (custom_lr['start_lr'] - custom_lr['end_lr'])
    else:
        lr = custom_lr['end_lr']

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def lambda_adjust_channels(train_lambda):
    if train_lambda == 0.001:
        out_channel_N, out_channel_M = 128, 128
    elif train_lambda < 0.04:
        out_channel_N, out_channel_M = 192, 192
    elif train_lambda < 0.1:
        out_channel_N, out_channel_M = 256, 256
    else:
        raise ValueError(f"Please select something appropriate: lambda adjust channels!")
    return out_channel_N, out_channel_M


def compute_likelihood_z(net, z):
    # z from B x C x ... to C x B x ...
    perm = np.arange(len(z.shape))
    perm[0], perm[1] = perm[1], perm[0]

    # Compute inverse permutation
    inv_perm = np.arange(len(z.shape))[np.argsort(perm)]
    z = z.permute(*perm).contiguous()
    shape = z.size()
    z_values = z.reshape(z.size(0), 1, -1)
    likelihood_z = net.entropy_bottleneck._likelihood(z_values)
    if net.entropy_bottleneck.use_likelihood_bound:
        likelihood_z = net.entropy_bottleneck.likelihood_lower_bound(likelihood_z)
    likelihood_z = likelihood_z.reshape(shape)
    likelihood_z = likelihood_z.permute(*inv_perm).contiguous()
    return likelihood_z


def compute_likelihood_y(net, y, scales, means):
    likelihood_y = net.gaussian_conditional._likelihood(y, scales=scales, means=means)
    if net.gaussian_conditional.use_likelihood_bound:
        likelihood_y = net.gaussian_conditional.likelihood_lower_bound(likelihood_y)
    return likelihood_y

def compute_distortion(x_hat, x):
    distortion = torch.mean((x_hat - x).pow(2))
    return distortion


def compute_bpp(likelihoods, x):
    num_pixels = x.size()[0] * x.size()[2] * x.size()[3]
    bpp = torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
    return bpp

