import argparse
from matplotlib import pyplot as plt
import numpy as np
import os
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

# from models.glom_segmenter import GLOMSegmenter
from models.predictive_segmenter import PredictiveSegmenter
from models.alpha_segmenter import AlphaSegmenter
from util import get_datasets


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


def main(args):
    # set up the model. different environments need different params
    tt_params = { "BipedalWalker-v2": (24, 4, 2, 10),
                  "CarRacing-v0": (16**2, 3, 4, 3) }
    tt = AlphaSegmenter(*tt_params[args.env])

    # reference trajectory used to plot latents over time
    ref_traj = next(get_datasets(folder=f"data/{args.env}")).tensors

    optimizer = optim.Adam(tt.parameters(), lr=1e-3)
    losses = []
    alphas = []
    try:
        for epoch_idx in range(3):
            for i, d in enumerate(get_datasets(folder=f"data/{args.env}")):
                new_losses, new_alphas = train(tt, d, optimizer, ref_traj=ref_traj)
                losses += new_losses
                alphas += new_alphas
                print(f"Epoch {epoch_idx}:\t{losses[-1]}")
    except KeyboardInterrupt:
        print("Stopping Early!")

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    ax1.plot(np.array(losses), label=["loss", "prior", "recon", "dynamics", "reinforce"])
    ax1.legend()
    ax2.imshow(np.array(alphas).T)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="BipedalWalker-v2", choices=["BipedalWalker-v2", "CarRacing-v0"])
    args = parser.parse_args()
    main(args)
