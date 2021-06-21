import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from models.generic import FeedForward


def stupid_planner(s, g_dist, f, a_shape,
                   eps=1e-2,
                   max_plan_len=10,
                   max_iters=1000,
                   use_diff_opt=True):
    """ use gradient descent to optimize a shooting-based plan

    tries to plan for increasingly long action sequences, up to max_plan_len.

    Arguments:
        s {torch.Tensor} -- start state
        g_dist {function} -- goal function
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

    a_shape = (a_shape,) if isinstance(a_shape, int) else a_shape # make tuple

    # unroll a sequence of actions according to the dynamics function
    def unroll(a_traj):
        s_traj = [s]
        for a in a_traj:
            s_prev = s_traj[-1]
            s_next = f(s_prev[None, ...], a[None, ...])[0]
            s_traj.append(s_next)

        return s_traj

    # optimizer a length-H sequence of actions using gradient descent
    def diff_opt(H):
        a_traj = nn.Parameter(torch.randn(H, *a_shape)) # init actions
        optimizer = optim.Adam([a_traj], lr=1e-2) # init optimizer

        for _ in range(max_iters):
            optimizer.zero_grad()

            s_traj = unroll(a_traj) # roll out the action sequence
            s_final = s_traj[-1] # final state
            l = g_dist(s_final) # distance to goal

            # if we've reached the goal, return the plan
            if l.item() < eps:
                return torch.stack(s_traj), a_traj
            # otherwise, gradient descent and continue
            else:
                l.backward(retain_graph=True)
                optimizer.step()

        return None

    # optimizer a length-H sequence of actions by guessing
    def guess_opt(H):
        a_traj = torch.randn(H, *a_shape) # init actions

        for _ in range(max_iters):
            with torch.no_grad():
                s_traj = unroll(a_traj) # roll out the action sequence
                s_final = s_traj[-1] # final state
                l = g_dist(s_final) # distance to goal

            # if we've reached the goal, return the plan
            if l.item() < eps:
                return torch.stack(s_traj), a_traj

        return None

    # if we are already at the goal, return an empty plan
    if g_dist(s) < eps:
        print(f"[stupid_planner] Warning, planning not required.")
        return s[None, ...], torch.zeros(0, *a_shape)

    # set the optimizaion function
    opt = diff_opt if use_diff_opt else guess_opt

    # for each plan length, see if we can find a plan
    for H in range(1, max_plan_len):
        result = opt(H)
        if result is not None:
            print(f"[stupid_planner] Found length {H} plan!")
            return result

    return None


def sequential_stupid_planner(s, g, fs, cs, a_shapes,
                              eps=5e-2,
                              use_diff_opt=True):
    """ apply the stupid planner at each level of abstraction

    plan to g at the top level. the first state of the resulting plan is the
    goal state for the level below

    Arguments:
        s {torch.Tensor} -- start state
        g {torch.Tensor} -- goal state
        fs {list(function)} -- dynamics function for each level
        cs {list(function)} -- correspondence function for each level
        a_shapes {list(tuple)} -- dims of action spaces

    Keyword Arguments:
        eps {number} -- goal-reaching criterion (default: {1e-2})
        dist_fn {function} -- state-space distance (default: {nn.MSELoss()})

    Returns:
        list(torch.Tensor), list(torch.Tensor) -- s, a traj at each level
    """

    L = len(fs) # number of level of hierarchy

    # use the correspondence function to get starting and end states at
    # the highest level of abstraction
    ss = [s]
    gs = [g]
    with torch.no_grad():
        for l in range(L-1):
            c = cs[l]
            ss.append(c(ss[l][None, ...])[0])
            gs.append(c(gs[l][None, ...])[0])


    # first, we need to check if we are already at the goal at some level of
    # the heirarchy. Working up from the bottom:
    L_plan = 0
    for l in range(L):
        L_plan = l
        if F.mse_loss(ss[l], gs[l]) < eps:
            print(f"Already at goal on level {l}.")
            break

    # g_dist is None when there is no subgoal coming from a higher level plan
    g_dist = None

    s_trajs = []
    a_trajs = []

    # for each level of hierarchy, working from high to low
    for l in range(L_plan - 1, -1, -1):
        print(f"Level {l}.")

        # g_dist is None where there is no subgoal coming from a higher level plan
        # in this case we just plan to the terminal goal for this level
        if g_dist is None:
            g_dist = lambda x: F.mse_loss(x, gs[l])

        # # use stupid_planner to find a plan at level l
        print(f'Planning...')
        result = stupid_planner(ss[l], g_dist, fs[l], a_shapes[l],
                                eps=eps,
                                use_diff_opt=use_diff_opt)

        if result is None:
            print(f"Planning failed.")
            return None

        s_traj, a_traj = result
        # save the trajectories to return
        s_trajs.append(s_traj)
        a_trajs.append(a_traj)

        # we found a trajectory, so we set the subgoal
        if l > 0 and len(a_traj) > 1:
            subgoal = s_traj[1]
            c = cs[l-1]
            g_dist = lambda x: F.mse_loss(c(x[None,...])[0], subgoal)
        # otherwise we set g_dist to None. This should only happen when we
        # found a plan at the level above, and then at this level we realized
        # we already satisfy that plan, so at the level below we need to plan
        # to the terminal goal
        else:
            g_dist = None


    # return the action trajectories for each level. we reverse because we
    # planned from L to 0
    return s_trajs[::-1], a_trajs[::-1]


def test_sequential_stupid_planner():
    s_dims = [10, 9, 8, 7]
    a_dims = [6, 5, 4 ,3]

    L = len(s_dims)

    # start and goal state
    s = torch.randn(s_dims[0])
    g = torch.randn(s_dims[0])

    # dynamics and correspondence for each level
    fs = []
    cs = []
    for l in range(L):
        f_l = FeedForward(s_dims[l] + a_dims[l], s_dims[l], h_dims=[20], nonlinearity=nn.Tanh)
        fs.append(f_l)
        if l + 1 < L:
            c_l = FeedForward(s_dims[l], s_dims[l+1], h_dims=[20], nonlinearity=nn.Tanh)
            cs.append(c_l)

    print('Planning with differentiable optimization')
    result = sequential_stupid_planner(s, g, fs, cs, a_dims, use_diff_opt=True)
    if result is not None:
        print('Planning Success!')
        for i, (s_traj, a_traj) in enumerate(zip(*result)):
            print(f'Level {i}. Plan length {a_traj.shape[0]}.')

    print('Planning with random optimization')
    result = sequential_stupid_planner(s, g, fs, cs, a_dims, use_diff_opt=False)
    if result is not None:
        print('Planning Success!')
        for i, (s_traj, a_traj) in enumerate(zip(*result)):
            print(f'Level {i}. Plan length {a_traj.shape[0]}.')



if __name__ == '__main__':
    test_sequential_stupid_planner()