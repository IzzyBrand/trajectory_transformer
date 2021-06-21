import h5py
import os
import torch
from torch.utils.data import DataLoader, TensorDataset


class Log:
    def __init__(self, env):
        self.env = env
        self.reset()

    def reset(self):
        self.folder = "../../data"
        self.env_folder = self.folder + "/" + self.env

        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        if not os.path.isdir(self.env_folder):
            os.mkdir(self.env_folder)

        self.s = []
        self.a = []

    def add(self, state, action):
        self.s.append(state)
        self.a.append(action)

    def get_filename(self):
        T = len(self.s)
        prefix = f"{T}steps"

        # check how many files already exist
        n = len([f for f in os.listdir(self.env_folder) if f.startswith(prefix)])
        # create a new filename
        return f"{self.env_folder}/{prefix}_{n}.h5"

    def save(self, dest=None):
        if dest is None:
            dest = self.get_filename()
        with h5py.File(dest, "w") as h5f:
            h5f.create_dataset("state", data=np.array(self.s))
            h5f.create_dataset("action", data=np.array(self.a))


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