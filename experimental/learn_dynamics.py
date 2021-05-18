import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

from model import FeedForward
from train import get_datasets


class DynamicsNet(FeedForward):
    def __init__(self, s_in, a_in, s_out, **kwargs):
        super(DynamicsNet, self).__init__(s_in + a_in, s_out, **kwargs)

class TransitionNet(FeedForward):
    def __init__(self, s_in, a_in, s_out, **kwargs):
        super(TransitionNet, self).__init__(s_in + a_in, s_out, **kwargs)

class ActionNet(FeedForward):
    def __init__(self, s_in, sp_in, a_out, **kwargs):
        super(ActionNet, self).__init__(s_in + sp_in, a_out, **kwargs)

class TrajectoryEncoder(nn.Module):
	def __init__(self, s_in, a_in, z_dim):
		super(TrajectoryEncoder, self).__init__()

		self.dynamics = DynamicsNet(s_in, a_in, s_in)
		self.transition = TransitionNet(s_in, a_in, s_in)
		self.action = ActionNet(s_in, s_in, a_in)

    def likelihood(self, s, a):
        

