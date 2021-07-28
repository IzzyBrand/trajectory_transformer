""" greater connectivity than GLOM. Learned message functions
"""
from matplotlib import pyplot as plt
import numpy as np
import sys
import torch
from torch import nn
from torch.distributions import Bernoulli
from torch.nn import functional as F
from torch.utils.data import DataLoader

from models.generic import GoalConditionedPolicyNet, DynamicsNet, FeedForward, GraphNetLayer
from util import get_datasets


class FCSegmenter(nn.Module):
    def __init__(self, s_in, a_in, z_out, g_out, n_levels=5):
        super(FCSegmenter, self).__init__()
        self.s_in = s_in
        self.a_in = a_in
        self.z_out = z_out
        self.g_out = g_out
        self.n_levels = n_levels

        self.h_dim = 32

        # # map a low-level state and a high-level action goal to low-level actions
        self.pi = [FeedForward(self.z_out + self.g_out, self.g_out) for _ in range(self.n_levels)]
        self.f = [FeedForward(self.z_out + self.g_out, self.z_out) for _ in range(self.n_levels)]
        self.c = [FeedForward(self.z_out, self.z_out) for _ in range(self.n_levels - 1)]

        self.in_net = FeedForward(self.s_in + self.a_in, self.h_dim)
        # NOTE(izzy): in the future each level's hidden dim and output dim might differ
        self.out_net = FeedForward(self.h_dim, self.z_out + self.g_out + 1)

        self.u_net = FeedForward(self.h_dim * 2, self.h_dim)
        self.d_net = FeedForward(self.h_dim * 2, self.h_dim)
        self.l_net = FeedForward(self.h_dim * 2, self.h_dim)
        self.r_net = FeedForward(self.h_dim * 2, self.h_dim)


    def encode(self, s, a, iters=10):
        traj = torch.cat([s, a], axis=1)
        T = traj.shape[0]
        z = torch.zeros(self.n_levels, T, self.h_dim)

        for i in range(iters):
            # pass input to the lowest level
            z[0] += self.in_net(traj)

            # pair off adjacent nodes to produce the input to the message nets
            z_ud = torch.cat([z[:-1], z[1:]], axis=2).reshape(-1, self.h_dim*2)
            z_lr = torch.cat([z[:, :-1], z[:, 1:]], axis=2).reshape(-1, self.h_dim*2)

            # compute and pass the messages
            z[1:] += self.u_net(z_ud).reshape(self.n_levels-1, T, self.h_dim)
            z[:-1] += self.d_net(z_ud).reshape(self.n_levels-1, T, self.h_dim)
            z[:,:-1] += self.l_net(z_lr).reshape(self.n_levels, T-1, self.h_dim)
            z[:,1:] += self.r_net(z_lr).reshape(self.n_levels, T-1, self.h_dim)

        return z


    def likelihood(self, s, a, b):
        loss = 0

        for l in range(self.n_levels):

            # compute the state-hold loss. This says that if b_t = 0, then
            # s_t should be equal to s_(t-1)
            loss += (torch.pow(s[l,:-1] - s[l,1:], 2) * (1 - b[l, 1:])).sum()

            # compute the action-hold loss. This says that if b_t = 0, then
            # a_t should be equal to a_(t-1)
            loss += (torch.pow(a[l,:-1] - a[l,1:], 2) * (1 - b[l, 1:])).sum()

            # compute the dynamics loss. this says that if b_t = 1, then
            # s_t should be equal to f(s_(t-1), a_(t-1))
            s_target = s[l,1:]
            s_pred = self.f[i](s[l, :-1], a[l, :-1])
            loss += (torch.pow(s_target - s_pred, 2) * b[l, 1:]).sum()

            # the following losses are hierarchical, so they can't be computed
            # at the top level
            if l < self.n_levels - 1:
                # compute the policy loss. this says that if b_t = 1, then
                # a_t shuold be equal to pi(s_t, g_t)
                a_target = a[l]
                a_pred = self.pi[i](s[l], a[l+1])
                loss += (torch.pow(a_target - a_pred, 2) * b[l]).sum()

                # compute the correspondence loss. this says that if b_t = 1,
                # then  z_t should be equal to c(s_t)
                s_target = s[l+1]
                s_pred = self.c[l](s[l])
                loss += (torch.pow(s_target - s_pred, 2) * b[l]).sum()

                # enforce hierarchy. this says that if a transition occurs at
                # the level above, then one should occur at this level too
                loss += (b[l+1] * (1. - b[l])).sum()

        return loss


    def decode(self, z):
        T = z.shape[1]
        out = self.out_net(z.view(-1, self.h_dim)).view(self.n_levels, T, -1)
        return torch.split(out, [self.z_out, self.g_out, 1], dim=2)


if __name__ == '__main__':
    fc = FCSegmenter(24, 4, 10, 6)

    for i, dataset in enumerate(get_datasets()):
        for j, (s, a) in enumerate(DataLoader(dataset, batch_size=50, shuffle=False)):
            with torch.no_grad():
                z = fc.encode(s, a)
                s, a, b = fc.decode(z)
                fc.likelihood(s, a, b)

            if j > 2:
                sys.exit(0)
