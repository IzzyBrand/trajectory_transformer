import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Bernoulli


class FeedForward(nn.Module):
    def __init__(self, d_in, d_out, h_dims=[32, 32], final_batchnorm=False):
        super(FeedForward, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.h_dims = h_dims
        all_dims = [d_in] + list(h_dims) + [d_out]

        # create a linear layer and nonlinearity for each hidden dim
        modules = []
        for i in range(len(all_dims) - 1):
            modules.append(nn.Linear(all_dims[i], all_dims[i+1]))
            modules.append(nn.LeakyReLU())

        modules.pop(-1)  # don't include the last nonlinearity
        if final_batchnorm:
            modules.append(nn.BatchNorm1d(all_dims[-1]))
        self.layers = nn.Sequential(*modules)  # add modules to net

    def forward(self, *xs):
        x = torch.cat(xs, axis=1)
        return self.layers(x)


class GoalConditionedPolicyNet(FeedForward):
    def __init__(self, s_in, g_in, a_out, **kwargs):
        super(GoalConditionedPolicyNet, self).__init__(s_in + g_in, a_out, **kwargs)


class DynamicsNet(FeedForward):
    def __init__(self, s_in, a_in, s_out, **kwargs):
        super(DynamicsNet, self).__init__(s_in + a_in, s_out, **kwargs)


class GraphNetLayer(nn.Module):
    def __init__(self, n_in, n_out):
        super(GraphNetLayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.h_dim = 32

        # compute input and output dims for sub-networks
        self.edge_fn_in = self.n_in * 2 + 1
        self.edge_fn_out = self.n_out
        self.node_fn_in = self.edge_fn_out
        self.node_fn_out = self.n_out

        self.edge_fn_layers = FeedForward(self.edge_fn_in,
                                          self.edge_fn_out,
                                          h_dims=[self.h_dim, self.h_dim],
                                          final_batchnorm=True)

        self.node_fn_layers = FeedForward(self.node_fn_in,
                                          self.node_fn_out,
                                          h_dims=[self.h_dim, self.h_dim],
                                          final_batchnorm=True)

    def edge_fn(self, x):
        N, T, _ = x.shape
        x = x.view(-1, self.edge_fn_in)
        x = self.edge_fn_layers(x)
        return x.view(N, T, self.edge_fn_out)

    def node_fn(self, x):
        N, T, _ = x.shape
        x = x.view(-1, self.node_fn_in)
        x = self.node_fn_layers(x)
        return x.view(N, T, self.node_fn_out)

    def forward(self, x):
        # [batch_size x graph_size x node_size]
        N, T, D = x.shape
        edge_direction = torch.ones(N, T, 1)
        zero_pad = torch.zeros(N, 1, self.n_out)

        # NOTE: this graph corresponds to a markov chain -- we only have
        # connectivity between adjacent nodes. For this reason we do not
        # have to construct an entire connectivity matrix (O(T^2), but can
        # instead simply create the band-diagonal connections (O(T))

        # message from a node to itself
        paired_inplace_in = torch.cat([x, x, 0 * edge_direction], axis=2)
        paired_inplace_out = self.edge_fn(paired_inplace_in)
        # message from a node to the next timestep
        paired_forward_in = torch.cat(
            [x[:, 1:], x[:, :-1], edge_direction[:, :-1]], axis=2
        )
        paired_forward_out = torch.cat(
            [zero_pad, self.edge_fn(paired_forward_in)], axis=1
        )
        # message from a node to the previous timestep
        paired_backward_in = torch.cat(
            [x[:, :-1], x[:, 1:], -edge_direction[:, :-1]], axis=2
        )
        paired_backward_out = torch.cat(
            [self.edge_fn(paired_backward_in), zero_pad], axis=1
        )

        return self.node_fn(
            paired_inplace_out + paired_forward_out + paired_backward_out
        )



