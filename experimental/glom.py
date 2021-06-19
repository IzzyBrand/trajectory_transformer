import numpy as np
from matplotlib import pyplot as plt
import sys
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from models.generic import FeedForward
from train import get_datasets


def normalize(x, axis=None):
    return x / torch.norm(x, dim=axis, keepdim=True)

def pca(X, d=2):
    m = X.mean(axis=0)
    s = X.std(axis=0)
    X = (X - m)/s
    cov = np.cov(X.T)
    vals, vecs = np.linalg.eig(cov)
    return vecs[:, np.argsort(vals)[-d:]]


class TrajectoryGLOM(nn.Module):

    def __init__(self, s_in, a_in, n_levels=5):
        super(TrajectoryGLOM, self).__init__()
        self.s_in = s_in
        self.a_in = a_in
        self.h_dim = 32
        self.n_levels = 5

        self._up_net = FeedForward(self.h_dim, self.h_dim, final_batchnorm=True)
        self._down_net = FeedForward(self.h_dim, self.h_dim, final_batchnorm=True)
        self.in_net = FeedForward(self.s_in + self.a_in, self.h_dim, final_batchnorm=True)

    def up_net(self, x):
        return self._up_net(x.view(-1, x.shape[-1])).view(*x.shape)

    def down_net(self, x):
        return self._down_net(x.view(-1, x.shape[-1])).view(*x.shape)

    def step(self, x, hidden):
        T, _ = x.shape

        # compute vertical messages
        h_up = torch.cat([self.in_net(x)[None, ...], self.up_net(hidden[:-1])], axis=0)
        h_down = torch.cat([self.down_net(hidden[1:]), torch.zeros(1, T, self.h_dim)], axis=0)

        # heirarchical attention (cosine distance)
        # v = normalize(hidden, axis=-1)
        # attn = torch.einsum('ltd, ltd -> lt', v[:,1:], v[:,:-1])[..., None] / 2 + 0.5
        # hrcl_attn = torch.cumprod((attn**10).flip(0), dim=0).flip(0)

        # heirarchical attention (L2 distance)
        attn = torch.norm(hidden[:,1:] - hidden[:,:-1], dim=2, keepdim=True)**4
        hrcl_attn = 1. - torch.exp(-torch.cumsum(attn.flip(0), dim=0).flip(0))

        # for i, a in enumerate(hrcl_attn):
        #     plt.hist(a.detach().numpy(), alpha=0.5, bins=50, label=f'Level {i}')
        # plt.legend()
        # plt.pause(0.01)
        # plt.clf()

        # compute horizontal messages (only adjacent)
        pad = torch.zeros(self.n_levels, 1, self.h_dim)
        h_left = torch.cat([hrcl_attn * hidden[:,1:], pad], axis=1)
        h_right = torch.cat([pad, hrcl_attn * hidden[:,:-1]], axis=1)

        return hidden + h_up + h_down + h_left + h_right

    def forward(self, x, iters=10):
        T, _ = x.shape

        # initialize the hidden state
        hidden = torch.randn(self.n_levels, T, self.h_dim) * 1e-2

        # iterate till convergence
        for i in range(iters):
            hidden = self.step(x, hidden)

        # return the hidden state
        return hidden


def plot_all_layers_together(hidden):
    hidden = hidden.detach().numpy()
    hidden_flat = hidden.reshape(-1, hidden.shape[-1])
    hidden_flat_pca = pca(hidden_flat.T)
    hidden_pca = hidden_flat_pca.reshape(*hidden.shape[:-1], 2)
    for i, level in enumerate(hidden_pca):
        plt.scatter(*level.T, label=f'Level {i}', s=10, alpha=0.25)
    plt.legend()
    plt.show()

def plot_all_layers_separately(hidden):
    hidden = hidden.detach().numpy()
    fig, axs = plt.subplots(1, hidden.shape[0])
    for i, level in enumerate(hidden):
        axs[i].scatter(*pca(level.T).T, s=10, alpha=0.25)
        axs[i].set_title(f'Level {i}')
    plt.show()


if __name__ == '__main__':
    tg = TrajectoryGLOM(24, 4)

    for i, dataset in enumerate(get_datasets()):
        for s, a in DataLoader(dataset, batch_size=512, shuffle=False):
            traj = torch.cat([s, a], axis=1)
            hidden = tg(traj, iters=30)
            break
        break

    plot_all_layers_together(hidden)
    # plot_all_layers_separately(hidden)
