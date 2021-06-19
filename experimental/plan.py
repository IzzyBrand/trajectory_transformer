import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F


def stupid_planner(s, g, f, a_shape,
                   eps=1e-2,
                   dist_fn=nn.MSELoss(),
                   max_plan_len=10,
                   max_iters=1000):
    """ use gradient descent to optimize a shooting-based plan

    tries to plan for increasingly long action sequences, up to max_plan_len.

    Arguments:
        s {torch.Tensor} -- start state
        g {torch.Tensor} -- goal state
        f {function} -- differentiable dynamics fn: f(s, a) -> s'
        a_shape {tuple} -- dims of action space

    Keyword Arguments:
        eps {number} -- goal-reaching criterion (default: {1e-2})
        dist_fn {function} -- state-space distance (default: {nn.MSELoss()})
        max_plan_len {number} -- number of actions (default: {10})
        max_iters {number} -- number of iterations (default: {1000})

    Returns:
        torch.Tensor, torch.Tensor -- states, actions (None if no plan found)
    """

    # unroll a sequence of actions according to the dynamics function
    def unroll(a_traj):
        s_traj = [s]
        for a in a_traj:
            s_prev = s_traj[-1]
            s_next = f(s_prev, a)
            s_traj.append(s_next)

        return s_traj

    # optimizer a length-H sequence of actions using gradient descent
    def opt(H):
        a_traj = torch.Parameter(torch.randn(H, *a_shape)) # init actions
        optimizer = optim.Adam(a_traj, lr=1e-3)

        for i in range(max_iters):
            optimizer.zero_grad()

            s_traj = unroll(a_traj) # roll out the action sequence
            s_final = s_traj[-1] # final state
            l = dist_fn(s_final, g) # distance to goal

            # if we've reached the goal, return the plan
            if l.item() < eps:
                return torch.stack([s_traj]),
                       torch.stack([a_traj])
            # otherwise, gradient descent and continue
            else:
                l.backward()
                optimizer.step()

        return None

    # for each plan length, see if we can find a plan
    for H in range(1, max_plan_len):
        result = opt(H)
        if result is not None:
            return result

    return None


def sequential_stupid_planner(s, g, fs, cs, a_shape,
                              eps=1e-2,
                              dist_fn=nn.MSELoss()):
    """ apply the stupid planner at each level of abstraction

    plan to g at the top level. the first state of the resulting plan is the
    goal state for the level below

    Arguments:
        s {torch.Tensor} -- start state
        g {torch.Tensor} -- goal state
        fs {list(function)} -- dynamics function for each level
        cs {list(function)} -- correspondence function for each level
        a_shape {tuple} -- dims of action space

    Keyword Arguments:
        eps {number} -- goal-reaching criterion (default: {1e-2})
        dist_fn {function} -- state-space distance (default: {nn.MSELoss()})

    Returns:
        list(torch.Tensor), list(torch.Tensor) -- s, a traj at each level
    """

    L = len(fs) - 1 # number of level of hierarchy (minus 1)

    # use the correspondence function to get starting and end states at
    # the highest level of abstraction
    ss = [s]
    gs = [g]
    for c in cs:
        ss.append(c(ss[-1]))
        gs.append(c(gs[-1]))

    # the top-level goal is the first subgoal
    subgoal = gs[L]
    s_trajs = []
    a_trajs = []

    # for each level of hierarchy, working from high to low
    for l in range(L, -1, -1):
        if ss[l] - gs[l]
        # use stupid_planner to find a plan at level l
        result = stupid_planner(ss[l], subgoal, f[l])

        if result is None:
            print(f"Planning failed at level {l}")
            return None

        # the next subgoal is the first state this plan
        s_traj, a_traj = result
        subgoal = s_traj[1]
        # save the trajectories to return
        s_trajs.append(s_traj)
        a_trajs.append(a_traj)

    # return the action trajectories for each level. we reverse because we
    # planned from L to 0
    return s_trajs[::-1], a_trajs[::-1]


