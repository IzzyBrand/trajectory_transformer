""" WORK IN PROGRESS

this is a half-baked idea to use the structure of the generative model in
concert with recognition nets to infer the latent distribution -- no iteration
"""

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Bernoulli, Normal

from models.generic import GoalConditionedPolicyNet, DynamicsNet, FeedForward, GraphNetLayer


def halve(x, dim=1):
    n = x.shape[dim]
    assert n % 2 == 0, "Expect even split."
    return torch.split(x, n // 2, dim=dim)

def split_and_sample(x, sample_shape=torch.Size([])):
    mu, logsigma = halve(x)
    q = Normal(mu, torch.exp(logsigma))
    return q.rsample(sample_shape=sample_shape)

def split_and_log_prob(x, v):
    mu, logsigma = halve(x)
    q = Normal(mu, torch.exp(logsigma))
    return q.log_prob(v)


class PredictiveSegmenter(nn.Module):
    def __init__(self, s_in, a_in, z_out, g_out):
        super(PredictiveSegmenter, self).__init__()
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
        # action recognition
        self.r = FeedForward(s_in + a_in, 2*g_out)


    def encode(self, s, a):
        T = s.shape[0]

        # recognize latent actions
        g = self.r(s, a)
        g_sample = split_and_sample(g)
        # g[:,3:] = 1
        # g_sample = torch.split(g, g.shape[1] // 2, dim=1)[0] # the mean

        # recognize latent states
        z = self.c(s)
        z_sample = split_and_sample(z)

        # predict next latent state
        z_next = self.f(z_sample, g_sample)

        # A_ij is the probability that a transition from timestep i to j took place.
        # this probability consists of
        # 
        # 1. the likelihood under the dynamics function: f(z_j | z_i, g_i)
        # 2. the likelihood that g_i persists until step j-1 under the policy
        #.   recognition function: sum k=i to j-1 of r(g_i | s_k, a_k)

        # get transition log likelihoods
        dynamics_log_likelihoods = split_and_log_prob(z_next[:-1], z_sample[1:, None, :]).sum(axis=2)
        dynamics_log_likelihoods = torch.triu(dynamics_log_likelihoods)

        # get recognition likelihoods
        policy_recogntiton_log_likelihoods = split_and_log_prob(g[:-1], g_sample[:-1, None, :]).sum(axis=2).T
        policy_recogntiton_log_likelihoods = torch.cumsum(torch.triu(policy_recogntiton_log_likelihoods), axis=1)

        # combine to get transition matrix
        transition_log_likelihoods = dynamics_log_likelihoods + policy_recogntiton_log_likelihoods

        # axes are [f(z,g), z_p]
        # transition_probs = torch.triu(torch.exp(transition_log_likelihoods))

        return transition_log_likelihoods



if __name__ == "__main__":
    # specify dimensions of inputs and outputs
    s_in = 5
    a_in = 2
    s_out = 4
    g_out = 3

    ps = PredictiveSegmenter(s_in, a_in, s_out, g_out)

    # sanity check to make sure output deminsions all seem correct
    T = 100
    s = torch.rand(T, s_in)
    a = torch.rand(T, a_in)
    plt.imshow(ps.encode(s, a).detach().numpy())
    plt.xlabel('ending timestep')
    plt.ylabel('starting timestep')
    plt.title('Log Probability of transition from i to j')
    plt.show()
