"""
early idea in which we explicitly predict an "alpha" that determines whether
each timestep corresponds to the start of a new high level transition.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Bernoulli

from models.generic import GoalConditionedPolicyNet, DynamicsNet, FeedForward, GraphNetLayer


class AlphaSegmenter(nn.Module):
    def __init__(self, s_in, a_in, s_out, g_out):
        super(AlphaSegmenter, self).__init__()
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

    def forward(self, s, a,
                use_prior = False,
                train_dynamics = False,
                train_correspondence = False,
                return_loss_breakdown = False):

        # encode the batch of trajectories
        z, g, alpha = self.encode(s, a)

        # harden and save the log probabilities for REINFORCE
        p_alpha = Bernoulli(alpha)
        with torch.no_grad():
            hard_alpha = p_alpha.sample()

        reinforce_log_probs = p_alpha.log_prob(hard_alpha).mean()

        # NOTE: I couldn't think of a way to vectorize this along the
        # batch dimension. I think that's ok because we might always
        # end up training this with a batch size of 1 (where each batch
        # is one trajectory)
        i = 0 # the first trajectory in the batch
        mask = hard_alpha[i, :, 0].bool()
        short_z = z[i, mask]
        short_g = g[i, mask]

        if not self.training:
            return short_z, short_g, hard_alpha

        else:
            bce_loss = nn.BCEWithLogitsLoss(reduce=False)
            # the prior loss is the number of alpha == 1 (to encourage sparsity)
            prior_losses = hard_alpha * use_prior

            # compute losses for dynamics and correspondence functions
            dynamics_loss = torch.Tensor([0])
            correspondence_loss = torch.Tensor([0])

            if train_dynamics:
                short_z_recon = self.f(short_z[:-1], short_g[:-1])
                dynamics_loss += ((short_z_recon - short_z[1:]) ** 2).mean()

            if train_correspondence:
                short_z_recon = self.c(s[i, mask])
                correspondence_loss += ((short_z_recon - short_z) ** 2).mean()

            # use the alpha mask to extend the high level goals for their duration
            hard_g = self.extend_goals_hard(g, hard_alpha)
            # use the high level goals to reconstruct low level actions
            a_recon = self.decode(s, hard_g)
            # and the quality of the reconstruction
            recon_losses = bce_loss(a_recon, a)
            # recon_loss = recon_losses.mean()

            per_timestep_losses = prior_losses + recon_losses + dynamics_loss + correspondence_loss
            loss = per_timestep_losses.mean()

            # use REINFORCE to estimate the gradients of the alpha parameters
            reinforce_loss = (reinforce_log_probs * per_timestep_losses.detach()).mean()
            loss += reinforce_loss
            
            # return the latents and the losses
            if not return_loss_breakdown:
                return short_z, short_g, hard_alpha, loss
            else:
                loss_breakdown = [loss.item(),
                                  prior_losses.mean().item(),
                                  recon_losses.mean().item(),
                                  dynamics_loss.item(),
                                  reinforce_loss.item()]
                return short_z, short_g, hard_alpha, loss, loss_breakdown 


if __name__ == "__main__":
    # specify dimensions of inputs and outputs
    s_in = 5
    a_in = 2
    s_out = 4
    g_out = 3

    tt = AlphaSegmenter(s_in, a_in, s_out, g_out)

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
    z, g, alpha = tt.encode(s, a)
    hard_alpha = alpha > 0.5
    hard_g = tt.extend_goals_hard(g, hard_alpha)
    a_recon = tt.decode(s, hard_g)

    print(f"High level state expects dim {s_out} and has shape {z.shape}")
    print(f"High level goal expects dim {g_out} and has shape {g.shape}")
    print(f"High level mask expects dim {1} and has shape {alpha.shape}")
    print(f"Reconstructed action expects dim {a_in} and has shape {a_recon.shape}")