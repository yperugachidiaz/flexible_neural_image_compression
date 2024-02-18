import numpy as np
import math
import torch


class Temperature:
    def __init__(self, method=None, rate=4e-3, ub=1., lb=0., temp_schedule='exp'):
        self.method = method
        self.rate = rate
        self.ub = ub
        self.lb = lb
        self.temp_schedule = temp_schedule

    def annealed_temperature(self, train_step, eps=1e-4):
        """
        Re-implementation annealed temperature from: https://arxiv.org/abs/2006.04240
        :param train_step: Iteration step
        :param eps: Stability factor
        :return: tau: annealed temperature parameter
        """
        if self.temp_schedule == 'exp':
            temp = np.exp(-self.rate * train_step)
        elif self.temp_schedule == 'exp0':
            t0 = 700
            temp = self.ub * np.exp(-self.rate * (train_step - t0))
        else:
            raise ValueError(f"This temperature schedule {self.temp_schedule} is not implemented yet!!!")
        tau = 1 - np.clip(temp, a_min=self.lb, a_max=self.ub)

        if self.method == 'temp_ste':
            tau = np.clip(tau, eps, a_max=None)
        return tau


class StraightThroughEstimator(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input


ste_round = StraightThroughEstimator.apply


# straight-through estimator with annealed temperature
def temperature_round(x, temp):
    return ste_round(x / temp) * temp


def temperature_unoise_round(x, temp, eps=1e-8, base_noise=.5):
    # shift T from T ∈ (0, 1) to T ∈ (0, .5)
    tau = (1. - temp) * base_noise
    rho = temp
    if rho == 0:
        rho = np.clip(rho, eps, a_max=None)

    # Shift unoise to U(-T, T)
    unoise = torch.rand_like(x)
    custom_unoise = tau * (2 * unoise - 1.)
    return ste_round((x + custom_unoise) / rho) * rho


def annealed_unoise_z(z_cur, tau, base_noise=0.5):
    tau = tau * base_noise  # T ∈ (0, 0.5)
    # Uniform noise: u ~ U(-T, T)
    unoise = torch.rand_like(z_cur)
    custom_unoise = tau * (2 * unoise - 1.)
    # z^ = z + U(-T, T)
    z_hat = z_cur + custom_unoise
    return z_hat


def annealed_unoise_y(y_cur, tau, base_noise=0.5):
    tau = tau * base_noise  # T ∈ (0, 0.5)
    # Uniform noise: u ~ U(-T, T)
    unoise = torch.rand_like(y_cur)
    custom_unoise = tau * (2 * unoise - 1.)
    # y^ = y + u
    y_hat = y_cur + custom_unoise
    return y_hat


def annealed_gnoise_z(z_cur, tau, base_noise=0.25):
    tau = tau * base_noise  # T ∈ (0, 0.25)
    # Gaussian noise: g ~ N(0, 1)
    gnoise = torch.randn_like(z_cur)
    # Shift noise: g ~ N(0, T)
    custom_gnoise = gnoise * tau
    # z^ = z + N(0, T)
    z_hat = z_cur + custom_gnoise
    return z_hat


def annealed_gnoise_y(y_cur, tau, base_noise=0.25):
    tau = tau * base_noise  # T ∈ (0, 0.25)
    # Gaussian noise: g ~ N(0, 1)
    gnoise = torch.randn_like(y_cur)
    # Shift noise: g ~ N(0, T)
    custom_gnoise = gnoise * tau
    # y^ = y + N(0, T)
    y_hat = y_cur + custom_gnoise
    return y_hat


def sga_logits(x, T, logits="ssl", a=1., eps=1e-5, straight_through=False):
    """
    Implementation different methods for SGA+ two-class rounding
    :param v: input
    :param T: temperature tau
    :param eps: epsilon
    :param a: scale factor for SSL
    :param straight_through: True for sampling hard with Straight-through trick
    if False: "reparametrization" trick (as in TensorFLow)
    :return: v_hat
    """
    x_floor = torch.floor(x)
    x_ceil = torch.ceil(x)
    x_bds = torch.stack([x_floor, x_ceil], dim=-1)

    left = (x - x_floor).clamp(-1 + eps, 1 - eps)
    right = (x_ceil - x).clamp(-1 + eps, 1 - eps)
    if logits == "atanh":
        rx_logits = torch.stack([- torch.atanh(left) / T, - torch.atanh(right) / T], dim=-1)

    elif logits == "logx":
        rx_logits = torch.stack([torch.log(1 - left) / T, torch.log(1 - right) / T], dim=-1)

    elif logits == "cosx":
        pi = torch.Tensor([math.pi]).to(x.device)

        def clamped_squared_cos(x, eps=1e-7):
            y = torch.pow(torch.cos(x * pi * 0.5), 2)
            y_scaled = (1 - eps) * y + eps  # Scale from (0-1) to (eps-1)
            return y_scaled

        inner_left = clamped_squared_cos(left)
        inner_right = clamped_squared_cos(right)
        rx_logits = torch.stack([torch.log(inner_left) / T, torch.log(inner_right) / T], dim=-1)

    elif logits == 'expxlogx':
        rx_logits = torch.stack([torch.exp(a * left) * torch.log(1-left**2) / T, torch.exp(a * right) * torch.log(1 - right**2) / T], dim=-1)

    elif logits == 'ssl':

        def clamped_ssl(x, a, eps=1e-5):
            x = x.clamp(0. + eps, 1. - eps)
            x_scale = (1 - x) / x
            temperature_scaled = (x_scale ** -(a))
            return -torch.log1p(temperature_scaled)

        rx_logits = torch.stack([clamped_ssl(left, a) / T, clamped_ssl(right, a) / T], dim=-1)
    else:
        raise ValueError(f"Method: {logits} is not implemented!!!")

    # Sample from Gumbel-Softmax distribution (from logits and tempr)
    rx_sample = torch.nn.functional.gumbel_softmax(rx_logits, tau=T, hard=straight_through)
    x_hat = torch.sum(x_bds * rx_sample, dim=-1)
    return x_hat


def sga_logits_3c(x, T, logits="log_absx_factor_pow", fac=0.98, pow=1.5, a=1.4, straight_through=False):
    """
    Implementation different methods for SGA+ three-class rounding
    :param x: input
    :param T: temperature tau
    :param fac: factor
    :param pow: power
    :param a: scale factor for SSL
    :param straight_through: True for sampling hard with Straight-through trick
    if False: "reparametrization" trick (as in TensorFLow)
    :return: v_hat
    """
    x_round = torch.round(x)
    x_left_round = x_round - 1
    x_right_round = x_round + 1

    x_bds = torch.stack([x_left_round, x_round, x_right_round], dim=-1)

    x_center = x - x_round
    x_cleft = x_center + 1
    x_cright = x_center - 1

    if logits == "absx":

        def absx(x):
            y = 1 - torch.abs(x) * (2/3)
            return y

        rx_logits = torch.stack([absx(x_cleft) / T, absx(x_center) / T, absx(x_cright) / T], dim=-1)

    elif logits == "log_absx_factor_pow":

        def log_absx_factor_pow(x, factor=0.98, power=1.5, eps=1e-6):
            y = 1 - torch.abs(x) * factor
            return torch.log(torch.pow(torch.clamp(y, min=eps), power))

        rx_logits = torch.stack([log_absx_factor_pow(x_cleft, factor=fac, power=pow) / T,
                                 log_absx_factor_pow(x_center, factor=fac, power=pow) / T,
                                 log_absx_factor_pow(x_cright, factor=fac, power=pow) / T], dim=-1)

    elif logits == "log_cosx_n":
        pi = torch.Tensor([math.pi]).to(x.device)

        def clamped_squared_cosx_n(x, power=3, eps=1e-7):
            y = torch.pow(torch.cos(x * pi * (1/3)), power)
            y_scaled = (1 - eps) * y + eps  # Scale from (0-1) to (eps-1)
            return torch.log(y_scaled)

        rx_logits = torch.stack([clamped_squared_cosx_n(x_cleft, power=pow) / T, clamped_squared_cosx_n(x_center, power=pow) / T, clamped_squared_cosx_n(x_cright, power=pow) / T], dim=-1)

    elif logits == "log_cosx_factor_pow":
        pi = torch.Tensor([math.pi]).to(x.device)

        def clamped_squared_cosx_factor_pow(x, factor=0.98, power=2., eps=1e-7):
            x_factor = x * factor
            x_clamp = torch.clamp(x_factor, min=-1, max=1)
            y = torch.pow(torch.cos(x_clamp * pi * (1/2)), power)
            return torch.log(y + eps)

        rx_logits = torch.stack([clamped_squared_cosx_factor_pow(x_cleft, factor=fac, power=pow) / T,
                                 clamped_squared_cosx_factor_pow(x_center, factor=fac, power=pow) / T,
                                 clamped_squared_cosx_factor_pow(x_cright, factor=fac, power=pow) / T], dim=-1)

    elif logits == "log_ssl_a_factor_pow":
        def sigmoid(x):
            return 1 / (1 + torch.exp(-x))

        def inv_sigmoid(x):
            return torch.log(x / (1 - x))

        def sigmoid_a_logit_pow(x, factor=1.4, power=1.5, a=1.4, eps=1e-6):
            x_factor = torch.abs(((factor / 1.5) * x))
            x_clamp = torch.clamp(x_factor, min=eps, max=1 - eps)
            y = sigmoid(-a * inv_sigmoid(x_clamp))
            return torch.log(torch.pow(torch.clamp(y, min=eps), power))

        rx_logits = torch.stack([sigmoid_a_logit_pow(x_cleft, factor=fac, power=pow, a=a) / T,
                                 sigmoid_a_logit_pow(x_center, factor=fac, power=pow, a=a) / T,
                                 sigmoid_a_logit_pow(x_cright, factor=fac, power=pow, a=a) / T], dim=-1)

    else:
        raise ValueError(f"Method: {logits} is not implemented!!!")

    # Sample from Gumbel-Softmax distribution
    rx_sample = torch.nn.functional.gumbel_softmax(rx_logits, tau=T, hard=straight_through)
    x_hat = torch.sum(x_bds * rx_sample, dim=-1)
    return x_hat


class ExperimentalBottleneck(torch.nn.Module):

    def __init__(self, method, **kwargs):
        super(ExperimentalBottleneck, self).__init__()
        if 'temp_method_class' in kwargs:
            self.temp_method = kwargs.get('temp_method_class')
        if 'gnoise_param' in kwargs:
             self.g_sigma = kwargs.get('gnoise_param')
        if 'logits' in kwargs:
             self.logits = kwargs.get('logits')
        if 'scale_factor' in kwargs:
            self.scale_factor = kwargs.get('scale_factor')
        if '3c_power' in kwargs:
            self.pow = kwargs.get('3c_power')
        if '3c_factor' in kwargs:
            self.fac = kwargs.get('3c_factor')

        for m, b in method.items():
            if b:
                self.method = m

    def computing_method(self, z_cur, y_cur, train_step, compute_zhat=True, mu=None):
        if self.method == "temp_ste":
            tau = self.temp_method.annealed_temperature(train_step)
            tau = torch.tensor(tau)
            if compute_zhat:
                z_hat = temperature_round(z_cur, tau)
                return z_hat
            else:
                y_cur = y_cur - mu
                y_hat = temperature_round(y_cur, tau)
                y_hat = y_hat + mu
                return y_hat

        elif self.method == "temp_unoise":
            tau = self.temp_method.annealed_temperature(train_step)
            if compute_zhat:
                z_hat = annealed_unoise_z(z_cur, tau)
                return z_hat
            else:
                y_cur = y_cur - mu
                y_hat = annealed_unoise_y(y_cur, tau)
                y_hat = y_hat + mu
                return y_hat

        elif self.method == "temp_uste":
            tau = self.temp_method.annealed_temperature(train_step)
            tau = torch.tensor(tau)
            if compute_zhat:
                z_hat = temperature_unoise_round(z_cur, tau)
                return z_hat
            else:
                y_cur = y_cur - mu
                y_hat = temperature_unoise_round(y_cur, tau)
                y_hat = y_hat + mu
                return y_hat

        elif self.method == "gnoise":
            if compute_zhat:
                noise = torch.randn_like(z_cur) * self.g_sigma
                z_hat = z_cur + noise
                return z_hat
            else:
                y_cur = y_cur - mu
                noise = torch.randn_like(y_cur) * self.g_sigma
                y_hat = y_cur + noise
                y_hat = y_hat + mu
                return y_hat

        elif self.method == "temp_gnoise":
            tau = self.temp_method.annealed_temperature(train_step)
            if compute_zhat:
                z_hat = annealed_gnoise_z(z_cur, tau)
                return z_hat
            else:
                y_cur = y_cur - mu
                y_hat = annealed_gnoise_y(y_cur, tau)
                y_hat = y_hat + mu
                return y_hat

        elif self.method == "sga_logits":
            tau = self.temp_method.annealed_temperature(train_step)
            tau = 1. - tau
            if compute_zhat:
                z_hat = sga_logits(z_cur, T=tau, logits=self.logits, a=self.scale_factor)
                return z_hat
            else:
                y_cur = y_cur - mu
                y_hat = sga_logits(y_cur, T=tau, logits=self.logits, a=self.scale_factor)
                y_hat = y_hat + mu
                return y_hat

        elif self.method == "sga_logits_3c":
            tau = self.temp_method.annealed_temperature(train_step)
            tau = 1. - tau
            if compute_zhat:
                z_hat = sga_logits_3c(z_cur, T=tau, logits=self.logits, fac=self.fac, pow=self.pow, a=self.scale_factor)
                return z_hat
            else:
                y_cur = y_cur - mu
                y_hat = sga_logits_3c(y_cur, T=tau, logits=self.logits, fac=self.fac, pow=self.pow, a=self.scale_factor)
                y_hat = y_hat + mu
                return y_hat

