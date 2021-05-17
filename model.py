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


class TemporalTransformer(nn.Module):
    def __init__(self, s_in, a_in, s_out, g_out):
        super(TemporalTransformer, self).__init__()
        self.s_in = s_in
        self.a_in = a_in
        self.s_out = s_out
        self.g_out = g_out

        self.h_dim = 32

        # map a low-level state and a high-level action goal to low-level actions
        self.pi = GoalConditionedPolicyNet(s_in, g_out, a_in)
        # predict next high level state
        self.f = DynamicsNet(s_out, g_out, s_out)
        # map from low to high level state
        self.c = FeedForward(s_in, s_out)

        # given a low level state-action trajectory, predict a high level
        # state-action trajectory with an extra bit to indicate if each
        # timestep is a continuation of a previous high level action
        self.gnn = nn.Sequential(
            GraphNetLayer(s_in + a_in, self.h_dim),
            nn.Tanh(),
            GraphNetLayer(self.h_dim, self.h_dim),
            nn.Tanh(),
            GraphNetLayer(self.h_dim, s_out + g_out + 1),
        )

    def hard(self, alpha):
        # sample hard values (0 or 1) using the "straight-thru" trick
        alpha_hard = Bernoulli(alpha).sample()
        return alpha_hard - alpha.detach() + alpha

    def extend_goals(self, g, alpha):
        """extend goals for their duraction along the trajectory

        given a goal for each timestep and a mask, alpha, repeat each
        goal g_i with alpha_i=1 forward in time until we hit alpha_j=1

        g_i = alpha_i * g_i + sum_{j=0}^{i-1} ( g_j * alpha_j * prod_{k=j}^{i} (1-alpha_k) )

        Arguments:
            g {torch.Tensor} -- [N x T x D]
            alpha {torch.Tensor} -- [N x T x 1]

        Returns:
            torch.Tensor -- [N x T x D]
        """

        N, T, _ = alpha.shape
        alpha_block = torch.tile(alpha, [1, 1, T])
        alpha_block = torch.tril(alpha_block, diagonal=-1)
        alpha_block = torch.cumprod(1 - alpha_block, dim=1)
        alpha_block = torch.tril(alpha_block, diagonal=0)

        g_block = torch.einsum("nhd, nth -> nthd", (g * alpha), alpha_block)
        return g_block.sum(axis=2)

    def extend_goals_hard(self, g, alpha):
        N, T, _ = g.shape
        assert N == 1
        g = g[0]
        alpha = alpha[0]

        new_g = []
        for t in range(T):
            if t == 0:
                new_g.append(torch.where(alpha[t] > 0, g[t], torch.zeros_like(g[t])))
            else:
                new_g.append(torch.where(alpha[t] > 0, g[t], new_g[-1]))

        return torch.stack(new_g, axis=0)[None, ...]

    def encode(self, s, a):
        x = torch.cat([s, a], axis=2)
        x = self.gnn(x)
        s_new, g, alpha = torch.split(x, [self.s_out, self.g_out, 1], dim=2)
        return s_new, g, torch.sigmoid(alpha)

    def decode(self, s, g):
        N, T, _ = g.shape
        g = g.view(-1, self.g_out)
        s = s.view(-1, self.s_in)
        return self.pi(s, g).view(N, T, -1)

    def nll(self, s, g, alpha):
        """compute the likelihood of the embedding

        The embedded high-level state and high-level action
        should be likely

        Arguments:
            s {[type]} -- [description]
            g {[type]} -- [description]
            alpha {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        indecision = alpha * (1 - alpha)
        distribution = ((s ** 2).mean(axis=2, keepdim=True) + (g ** 2).mean(axis=2, keepdim=True)) * alpha
        prior = alpha
        return prior


if __name__ == "__main__":
    # specify dimensions of inputs and outputs
    s_in = 5
    a_in = 2
    s_out = 4
    g_out = 3

    tt = TemporalTransformer(s_in, a_in, s_out, g_out)

    # show how extend goals works
    g = torch.tile(torch.Tensor(np.arange(10) + 1)[None, :, None], [1, 1, g_out])
    alpha = torch.Tensor(np.array([1.0, 0, 0, 1, 0, 1, 1, 0, 0, 0]))[None, :, None]
    g_new = tt.extend_goals(g, alpha)
    print(g)
    print(alpha)
    print(g_new)

    # sanity check to make sure output deminsions all seem correct
    T = 100
    s = torch.rand(1, T, s_in)
    a = torch.rand(1, T, a_in)
    s_new, g, alpha = tt.encode(s, a)
    a_recon = tt.decode(s, g, alpha)
    print(f"High level state expects dim {s_out} and has shape {s_new.shape}")
    print(f"High level goal expects dim {g_out} and has shape {g.shape}")
    print(f"High level mask expects dim {1} and has shape {alpha.shape}")
    print(f"Reconstructed action expects dim {a_in} and has shape {a_recon.shape}")
