import torch
import numpy as np


class StraightThroughEstimator(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input


ste_round = StraightThroughEstimator.apply


def map_quantize_z(z, mode):

    if mode == "noise":
        # Uniform noise U(-0.5, 0.5)
        noise = torch.rand_like(z) - 0.5
        z_hat = z + noise
        return z_hat

    elif mode == "round":
        z_hat = ste_round(z)
        return z_hat

    elif mode == "symbols":
        z_hat = torch.round(z).int()
        return z_hat

    else:
        raise ValueError(f"Unknown mode: {mode}")


def map_quantize_y(y_cur, mu, mode):
    if mode == "noise":
        # Uniform noise U(-0.5, 0.5)
        noise = torch.rand_like(y_cur) - 0.5
        y_hat = y_cur + noise
        return y_hat

    y = y_cur.clone()

    if mu is not None:
        y -= mu

    y = ste_round(y)

    if mode == "round":
        if mu is not None:
            y += mu
        y_hat = y
        return y_hat

    if mode == "symbols":
        y_hat = torch.round(y).int()
        return y_hat

    else:
        raise ValueError(f"Unknown mode: {mode}")


def sga(v, T, eps=1e-5, straight_through=False):
    """
    Re-implementation SGA from: https://arxiv.org/abs/2006.04240
    :param v: input
    :param T: temperature tau
    :param eps: epsilon
    :param straight_through: True for sampling hard with Straight-through trick
    if False: "reparametrization" trick (as in TensorFLow)
    :return: v_hat
    """
    v_floor = torch.floor(v)
    v_ceil = torch.ceil(v)
    v_bds = torch.stack([v_floor, v_ceil], dim=-1)

    left = (v - v_floor).clamp(-1 + eps, 1 - eps)
    right = (v_ceil - v).clamp(-1 + eps, 1 - eps)
    rv_logits = torch.stack([- torch.atanh(left) / T, - torch.atanh(right) / T], dim=-1)

    # Sample from Gumbel-Softmax distribution
    rv_sample = torch.nn.functional.gumbel_softmax(rv_logits, tau=T, hard=straight_through)
    v_hat = torch.sum(v_bds * rv_sample, dim=-1)
    return v_hat


def annealed_temperature(train_step, rate=4e-3, upper_bound=0.2, lower_bound=1e-8, backend=np):
    """
    Re-implementation annealed temperature from: https://arxiv.org/abs/2006.04240
    :param train_step: Iteration step
    :param rate: Rate constant
    :param upper_bound: Upper bound
    :param lower_bound: Lower bound
    :param backend: Using numpy or without (same computation)
    :return: tau the temperature parameter
    """
    if backend is None:
        tau = min(max(np.exp(-rate * train_step), lower_bound), upper_bound)
        return tau
    else:
        tau = backend.minimum(backend.maximum(backend.exp(-rate * train_step), lower_bound), upper_bound)
        return tau


def deterministic_annealing(v, T, eps=1e-5):
    """
    Re-implementation deterministic annealing from: https://arxiv.org/abs/2006.04240
    :param v: Input variable v
    :param T: Temperature parameter tau
    :param eps: Constant
    :return: Variable v^
    """
    v_floor = torch.floor(v)
    v_ceil = torch.ceil(v)
    v_bds = torch.stack([v_floor, v_ceil], dim=-1)

    left = (v - v_floor).clamp(-1 + eps, 1 - eps)
    right = (v_ceil - v).clamp(-1 + eps, 1 - eps)
    rv_logits = torch.stack([- torch.atanh(left) / T, - torch.atanh(right) / T], dim=-1)

    rv = torch.nn.functional.softmax(rv_logits, dim=-1)
    v_hat = torch.sum(v_bds * rv, dim=-1)
    return v_hat


class Bottleneck(torch.nn.Module):
    def __init__(self, method):
        super(Bottleneck, self).__init__()
        for m, b in method.items():
            if b:
                self.method = m

    def computing_method(self, z_cur, y_cur, train_step, compute_zhat=True, mu=None):
        if self.method == "ste":
            if compute_zhat:
                z_hat = ste_round(z_cur)
                return z_hat
            else:
                y_cur = y_cur - mu
                y_hat = ste_round(y_cur)
                y_hat = y_hat + mu
                return y_hat

        elif self.method == "unoise":
            if compute_zhat:
                z_hat = map_quantize_z(z_cur, mode="noise")
                return z_hat
            else:
                y_cur = y_cur - mu
                y_hat = map_quantize_y(y_cur, mu, mode="noise")
                y_hat = y_hat + mu
                return y_hat

        elif self.method == "danneal":
            tau = annealed_temperature(train_step)
            if compute_zhat:
                z_hat = deterministic_annealing(v=z_cur, T=tau)
                return z_hat
            else:
                y_cur = y_cur - mu
                y_hat = deterministic_annealing(v=y_cur, T=tau)
                y_hat = y_hat + mu
                return y_hat

        elif self.method == "sga":
            tau = annealed_temperature(train_step)
            if compute_zhat:
                z_hat = sga(v=z_cur, T=tau)
                return z_hat
            else:
                y_cur = y_cur - mu
                y_hat = sga(v=y_cur, T=tau)
                y_hat = y_hat + mu
                return y_hat
