"""
reference:
https://github.com/yuyang-long/SSA
"""

from torch.autograd import Variable as V
from abc import abstractmethod
from torch import Tensor
from math import ceil
import numpy as np
import torch
from torch import nn
from typing import Callable, List

def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.fft.fft(v)

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    # V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
    V = Vc.real * W_r - Vc.imag * W_i
    if norm == "ortho":
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == "ortho":
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    tmp = torch.complex(real=V[:, :, 0], imag=V[:, :, 1])
    v = torch.fft.ifft(tmp)

    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, : N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, : N // 2]

    return x.view(*x_shape).real


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)

def clamp(x: torch.tensor, min_value=0, max_value=1):
    return torch.clamp(x, min=min_value, max=max_value)

class AdversarialInputAttacker():
    def __init__(self, model: List[torch.nn.Module],
                 epsilon=16 / 255,
                 norm='Linf'):
        assert norm in ['Linf', 'L2']
        self.norm = norm
        self.epsilon = epsilon
        self.models = model
        self.init()
        self.model_distribute()
        self.device = torch.device('cuda')
        self.n = len(self.models)

    @abstractmethod
    def attack(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.attack(*args, **kwargs)

    def model_distribute(self):
        '''
        make each model on one gpu
        :return:
        '''
        num_gpus = torch.cuda.device_count()
        models_each_gpu = ceil(len(self.models) / num_gpus)
        for i, model in enumerate(self.models):
            model.to(torch.device(f'cuda:{num_gpus - 1 - i // models_each_gpu}'))
            model.device = torch.device(f'cuda:{num_gpus - 1 - i // models_each_gpu}')

    def init(self):
        # set the model parameters requires_grad is False
        for model in self.models:
            model.requires_grad_(False)
            model.eval()

    def to(self, device: torch.device):
        for model in self.models:
            model.to(device)
            model.device = device
        self.device = device

    def clamp(self, x: Tensor, ori_x: Tensor) -> Tensor:
        B = x.shape[0]
        if self.norm == 'Linf':
            x = torch.clamp(x, min=ori_x - self.epsilon, max=ori_x + self.epsilon)
        elif self.norm == 'L2':
            difference = x - ori_x
            distance = torch.norm(difference.view(B, -1), p=2, dim=1)
            mask = distance > self.epsilon
            if torch.sum(mask) > 0:
                difference[mask] = difference[mask] / distance[mask].view(torch.sum(mask), 1, 1, 1) * self.epsilon
                x = ori_x + difference
        x = torch.clamp(x, min=0, max=1)
        return x


class SSA_CommonWeakness(AdversarialInputAttacker):
    def __init__(
        self,
        model: List[nn.Module],
        total_step: int = 10,
        random_start: bool = False,
        step_size: float = 16 / 255 / 5,
        criterion: Callable = nn.CrossEntropyLoss(),
        targeted_attack=False,
        mu=1,
        outer_optimizer=None,
        reverse_step_size=16 / 255 / 15,
        inner_step_size: float = 250,
        *args,
        **kwargs
    ):
        self.random_start = random_start
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        self.outer_optimizer = outer_optimizer
        self.reverse_step_size = reverse_step_size
        super(SSA_CommonWeakness, self).__init__(model, *args, **kwargs)
        self.inner_step_size = inner_step_size

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def attack(self, x, y):
        N = x.shape[0]
        original_x = x.clone()
        inner_momentum = torch.zeros_like(x)
        self.outer_momentum = torch.zeros_like(x)
        if self.random_start:
            x = self.perturb(x)

        for _ in range(self.total_step):
            # # --------------------------------------------------------------------------------#
            # # first step
            # self.begin_attack(x.clone().detach())
            # x.requires_grad = True
            # logit = 0
            # for model in self.models:
            #     logit += model(x.to(model.device)).to(x.device)
            # loss = self.criterion(logit, y)
            # loss.backward()
            # grad = x.grad
            # x.requires_grad = False
            # if self.targerted_attack:
            #     x += self.reverse_step_size * grad.sign()
            # else:
            #     x -= self.reverse_step_size * grad.sign()
            # x = self.clamp(x, original_x)
            # # --------------------------------------------------------------------------------#
            # # second step
            x.grad = None
            self.begin_attack(x.clone().detach())
            for model in self.models:
                x.requires_grad = True
                grad = self.get_grad(x, y, model)
                self.grad_record.append(grad)
                x.requires_grad = False
                # update
                if self.targerted_attack:
                    inner_momentum = self.mu * inner_momentum - grad / (
                        torch.norm(grad.reshape(N, -1), p=2, dim=1).view(N, 1, 1, 1) + 1e-5
                    )
                    x += self.inner_step_size * inner_momentum
                else:
                    inner_momentum = self.mu * inner_momentum + grad / (
                        torch.norm(grad.reshape(N, -1), p=2, dim=1).view(N, 1, 1, 1) + 1e-5
                    )
                    x += self.inner_step_size * inner_momentum
                x = clamp(x)
                x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)
            x = self.end_attack(x)
            x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)
        return x

    @torch.no_grad()
    def begin_attack(self, origin: torch.tensor):
        self.original = origin
        self.grad_record = []

    @torch.no_grad()
    def end_attack(self, now: torch.tensor, ksi=16 / 255 / 5):
        """
        theta: original_patch
        theta_hat: now patch in optimizer
        theta = theta + ksi*(theta_hat - theta), so:
        theta =(1-ksi )theta + ksi* theta_hat
        """
        patch = now
        if self.outer_optimizer is None:
            fake_grad = patch - self.original
            self.outer_momentum = self.mu * self.outer_momentum + fake_grad / torch.norm(fake_grad, p=1)
            patch.mul_(0)
            patch.add_(self.original)
            patch.add_(ksi * self.outer_momentum.sign())
            # patch.add_(ksi * fake_grad)
        else:
            fake_grad = -ksi * (patch - self.original)
            self.outer_optimizer.zero_grad()
            patch.mul_(0)
            patch.add_(self.original)
            patch.grad = fake_grad
            self.outer_optimizer.step()
        patch = clamp(patch)
        del self.grad_record
        del self.original
        return patch

    def get_grad(self, x, y, model):
        rho = 0.5
        N = 20
        sigma = 16
        noise = 0
        for n in range(N):
            x.requires_grad = True
            gauss = torch.randn(*x.shape) * (sigma / 255)
            gauss = gauss.cuda()
            x_dct = dct_2d(x + gauss).cuda()
            mask = (torch.rand_like(x) * 2 * rho + 1 - rho).cuda()
            x_idct = idct_2d(x_dct * mask)
            x_idct = V(x_idct, requires_grad=True)
            logit = model(x_idct.to(model.device)).to(x_idct.device)
            loss = self.criterion(logit, y)
            loss.backward()
            x.requires_grad = False
            noise += x_idct.grad.data
            x.grad = None
        noise = noise / N
        return noise
