import os
import h5py
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from model import TemporalTransformer


def train(model, dataset, optimizer):
    losses = []
    alphas = []
    bce_loss = nn.BCEWithLogitsLoss()

    train_dynamics = True

    for s, a in DataLoader(dataset, batch_size=64, shuffle=False):
        optimizer.zero_grad()
        # implemented the model to have a batch dimension (N) in addition a
        # trajectory dimensions (T). for now our batches are trajectories,
        # so the batch dimension is 1 [N x T x D]
        s = s[None, ...]
        a = a[None, ...]
        # encode the batch of trajectories
        s_new, g, alpha = model.encode(s, a)

        # encourage alphas to be either 0 or 1
        prior_loss = (alpha * (1 - alpha)).mean() + (alpha.mean() - 0.5) ** 2

        if train_dynamics:
            dynamics_loss = 0
            # NOTE: I couldn't think of a way to vectorize this along the
            # batch dimension. I think that's ok because we might always
            # end up training this with a batch size of 1 (where each batch
            # is one trajectory)
            for i in range(s.shape[0]):
                mask = alpha[i, :, 0] > 0.5
                if mask.sum() < 2:
                    break  # we need multiple steps
                short_s = s_new[i, mask]
                short_g = g[i, mask]
                short_s_recon = model.f(short_s[:-1], short_g[:-1])
                dynamics_loss += ((short_s_recon - short_s[1:]) ** 2).mean()

        # use the alpha mask to extend the high level goals for their duration
        g = model.extend_goals(g, alpha)
        # use the high level goals to reconstruct low level actions
        a_recon = model.decode(s, g)
        # and the quality of the reconstruction
        recon_loss = bce_loss(a_recon, a)
        # compute the mean across the batch and trajectory
        loss = recon_loss + prior_loss + dynamics_loss

        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if alpha.shape[1] == 64:
            alphas.append(alpha.detach()[0, :, 0].numpy())

    print(losses[-1])
    return alphas


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
    optimizer = optim.Adam(tt.parameters(), lr=1e-4)

    alphas = []
    try:
        for epoch_idx in range(1):
            for i, d in enumerate(get_datasets()):
                alphas += train(tt, d, optimizer)
                # print(f'Epoch {epoch_idx}:\t{losses[-1]}')
    except KeyboardInterrupt:
        print("Stopping Early!")

    plt.imshow(np.array(alphas).T)
    plt.show()


if __name__ == "__main__":
    main()
