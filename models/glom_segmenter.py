from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Bernoulli, Normal

from experimental.glom import TrajectoryGLOM
from models.generic import GoalConditionedPolicyNet, DynamicsNet, FeedForward, GraphNetLayer


class GLOMSegmenter(nn.Module):
    def __init__(self, s_in, a_in, z_out, g_out):
        super(GLOMSegmenter, self).__init__()
        self.s_in = s_in
        self.a_in = a_in
        self.z_out = z_out
        self.g_out = g_out

        self.h_dim = 32

        # map a low-level state and a high-level action goal to low-level actions
        self.pi = GoalConditionedPolicyNet(s_in, g_out, a_in)
        # predict next high level state
        self.f = DynamicsNet(z_out, g_out, 2*z_out)
        # map from low to high level state
        self.c = FeedForward(s_in, 2*z_out)

        self.glom = TrajectoryGLOM()


    def encode(self, s, a):
        pass