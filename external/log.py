import h5py
import numpy as np
import os

class Log():
    def __init__(self, env):
        self.env = env
        self.reset()

    def reset(self):
        self.folder = '../../data'
        self.env_folder = self.folder + '/' + self.env

        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        if not os.path.isdir(self.env_folder):
            os.mkdir(self.env_folder)

        self.s = []
        self.a = []

    def add(self, state, action):
        self.s.append(state[0])
        self.a.append(action)

    def get_filename(self):
        T = len(self.s)
        prefix = f'{T}steps'

        # check how many files already exist
        n = len([f for f in os.listdir(self.env_folder) if f.startswith(prefix)])
        # create a new filename
        return f'{self.env_folder}/{prefix}_{n}.h5'

    def save(self, dest=None):
        if dest is None:
            dest = self.get_filename()
        with h5py.File(dest, 'w') as h5f:
            h5f.create_dataset('state', data=np.array(self.s))
            h5f.create_dataset('action', data=np.array(self.a))