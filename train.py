import os
import h5py
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn, optim
from torch.distributions import Bernoulli
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from temporal_transformer import TemporalTransformer


def train(model, dataset, optimizer, batch_size=256, ref_traj=None):
    losses = []
    alphas = []

    for s, a in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        model.train()
        optimizer.zero_grad()
        # implemented the model to have a batch dimension (N) in addition a
        # trajectory dimensions (T). for now our batches are trajectories,
        # so the batch dimension is 1 [N x T x D]
        s = s[None, ...]
        a = a[None, ...]

        short_z, short_g, hard_alpha, loss, loss_breakdown = model(s, a, return_loss_breakdown=True)

        loss.backward()
        optimizer.step()

        losses.append(loss_breakdown)


        if ref_traj is None and hard_alpha.shape[1] == batch_size:
            alphas.append(hard_alpha.detach()[0, :, 0].numpy())

        elif ref_traj is not None:
            model.eval()
            alphas.append(model.encode(ref_traj[0][None, :300, :],
                                       ref_traj[1][None, :300, :])[2][0, :, 0].detach().numpy())

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
    ref_traj = next(get_datasets()).tensors
    losses = []
    alphas = []
    try:
        for epoch_idx in range(3):
            for i, d in enumerate(get_datasets()):
                new_losses, new_alphas = train(tt, d, optimizer, ref_traj=ref_traj)
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
