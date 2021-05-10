import os
import h5py
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn, optim
from torch.distributions import Bernoulli
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from model import TemporalTransformer


def train(model, dataset, optimizer, batch_size=128):
    losses = []
    alphas = []
    bce_loss = nn.BCEWithLogitsLoss(reduce=False)

    train_dynamics = True

    for s, a in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        optimizer.zero_grad()
        # implemented the model to have a batch dimension (N) in addition a
        # trajectory dimensions (T). for now our batches are trajectories,
        # so the batch dimension is 1 [N x T x D]
        s = s[None, ...]
        a = a[None, ...]
        # encode the batch of trajectories
        s_new, g, alpha = model.encode(s, a)

        # harden and save the log probabilities for REINFORCE
        p_alpha = Bernoulli(alpha)
        with torch.no_grad():
            hard_alpha = p_alpha.sample()
        log_probs = p_alpha.log_prob(hard_alpha).mean()

        prior_losses = model.nll(s_new, g, hard_alpha)

        dynamics_loss = torch.Tensor([0])
        if train_dynamics:
            # NOTE: I couldn't think of a way to vectorize this along the
            # batch dimension. I think that's ok because we might always
            # end up training this with a batch size of 1 (where each batch
            # is one trajectory)
            for i in range(s.shape[0]):
                mask = hard_alpha[i, :, 0].bool()
                if mask.sum() < 3:
                    break  # we need multiple steps
                short_s = s_new[i, mask]
                short_g = g[i, mask]
                short_s_recon = model.f(short_s[:-1], short_g[:-1])
                dynamics_loss += ((short_s_recon - short_s[1:]) ** 2).mean()

        # use the alpha mask to extend the high level goals for their duration
        g = model.extend_goals_hard(g, hard_alpha)
        # use the high level goals to reconstruct low level actions
        a_recon = model.decode(s, g)
        # and the quality of the reconstruction
        recon_losses = bce_loss(a_recon, a) * 5
        # recon_loss = recon_losses.mean()

        per_timestep_losses = prior_losses + recon_losses + dynamics_loss
        loss = per_timestep_losses.mean()

        # use REINFORCE to estimate the gradients of the alpha parameters
        reinforce_loss = (log_probs * per_timestep_losses.detach()).mean()
        loss += reinforce_loss

        loss.backward()
        optimizer.step()

        if alpha.shape[1] == batch_size:
            losses.append([loss.item(),
                           prior_losses.mean().item(),
                           recon_losses.mean().item(),
                           dynamics_loss.item(),
                           reinforce_loss.item()])

            alphas.append(alpha.detach()[0, :, 0].numpy())

    return losses, alphas


def get_datasets(folder="data/BipedalWalker-v2"):
    # TODO: revisit improved dataloader
    # https://towardsdatascience.com/hdf5-datasets-for-pytorch-631ff1d750f5
    filenames = [f for f in os.listdir(folder) if f.endswith("h5")]

    for filename in filenames:
        with h5py.File(folder + "/" + filename) as h5f:
            state = torch.Tensor(h5f["state"])
            action = torch.Tensor(h5f["action"])

            if folder == "data/BipedalWalker-v2":
                # renormalize the action from [-1,1] to [0,1]
                action = (action + 1) / 2.0
            elif folder == "data/CarRacing-v0":
                # preprocess images down to 16 x 16 and flatten
                state = F.interpolate(state[None, ...], (16, 16)).view(-1, 16 ** 2)

            yield TensorDataset(state, action)


def main():
    tt = TemporalTransformer(24, 4, 2, 10)
    optimizer = optim.Adam(tt.parameters(), lr=1e-3)

    losses = []
    alphas = []
    try:
        for epoch_idx in range(3):
            for i, d in enumerate(get_datasets()):
                new_losses, new_alphas = train(tt, d, optimizer)
                losses += new_losses
                alphas += new_alphas
                print(f'Epoch {epoch_idx}:\t{losses[-1]}')
    except KeyboardInterrupt:
        print("Stopping Early!")

    print(np.array(losses).shape)
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    ax1.plot(np.array(losses), label=['loss', 'prior', 'recon', 'dynamics', 'reinforce'])
    ax1.legend()
    ax2.imshow(np.array(alphas).T)
    plt.show()


if __name__ == "__main__":
    main()
