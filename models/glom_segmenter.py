""" inspired by GLOM

iterative message passing process to infer the latent variables

"""
from matplotlib import pyplot as plt
import numpy as np
import sys
import torch
from torch import nn
from torch.distributions import Bernoulli
from torch.nn import functional as F
from torch.utils.data import DataLoader

from experimental.glom import TrajectoryGLOM
from models.generic import GoalConditionedPolicyNet, DynamicsNet, FeedForward, GraphNetLayer
from util import get_datasets


class GLOMSegmenter(nn.Module):
    def __init__(self, s_in, a_in, z_out, g_out):
        super(GLOMSegmenter, self).__init__()
        self.s_in = s_in
        self.a_in = a_in
        self.z_out = z_out
        self.g_out = g_out

        # map a low-level state and a high-level action goal to low-level actions
        self.pi = GoalConditionedPolicyNet(s_in, g_out, a_in)
        # predict next high level state
        self.f = DynamicsNet(z_out, g_out, 2*z_out)
        # map from low to high level state
        self.c = FeedForward(s_in, 2*z_out)

        self.glom = TrajectoryGLOM(s_in + a_in, z_out + g_out, n_levels=5)


    def encode(self, s, a, iters=10):
        traj = torch.cat([s, a], axis=1)
        return self.glom(traj, iters=iters)

    def segment(self, z):
        # compute the relationship between adjacent hidden states
        hrcl_attn = self.glom.hierarchical_attention(z)

        # harden to 0 or 1
        # hard_attn = Bernoulli(hrcl_attn).sample()
        hard_attn = (hrcl_attn > hrcl_attn.mean()).float()

        # ensure that hierarchy is preserved
        hard_attn = 1-torch.cumprod((1-hard_attn).flip(0), axis=0).flip(0)

        plt.imshow(hrcl_attn)
        plt.show()
        plt.imshow(hard_attn)
        plt.show()

        extended_z = torch.zeros_like(z)
        shortened_z = []

        # wherever there is a zero, the two adjacent states are unrelated, 
        # this means it is the start of a new action
        # TODO(izzy): make this broadcastable
        for l in range(hard_attn.shape[0]):
            shortened_z_l = []
            for t in range(hard_attn.shape[1]):
                if hard_attn[l, t] == 1:
                    shortened_z_l.append(z[l,t])
                    extended_z[l, t] = z[l, t]
                if t > 0:
                    extended_z[l, t] = extended_z[l, t-1]

            shortened_z.append(torch.stack(shortened_z_l))


        for l, z_l in enumerate(shortened_z):
            print(l, z_l.shape)
        return extended_z, shortened_z


    def decode(self, z, g, s):
        pass

if __name__ == '__main__':
    gs = GLOMSegmenter(24, 4, 10, 6)

    for i, dataset in enumerate(get_datasets()):
        for j, (s, a) in enumerate(DataLoader(dataset, batch_size=50, shuffle=False)):
            with torch.no_grad():
                hidden = gs.encode(s, a)
                gs.segment(hidden)


            if j > 2:
                sys.exit(0)
