import torch
from torch import nn
from pathlib import Path
import numpy as np
import seaborn as sns
import scipy.io as spio
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from modules import optimize_ML, optimize_ELBO
from modules.elbo import ELBO

import os

# load sample trajectory
sim_idx = 0 # which out of the 100 trajectories to use
data_path = Path('data')
f_name = 'sim_0134.mat'
S = spio.loadmat(Path(data_path) / f_name)['S']
S_list = [S[0, i] for i in range(S.shape[1])]  # convert to list of structs

# unpack data
n_corr_obs = S_list[sim_idx]['Pc_reshaped'][0, 0] * S_list[sim_idx]['num_trials_mat'][0, 0]
n_total_obs = S_list[sim_idx]['num_trials_mat'][0, 0]
n_frames = S_list[sim_idx]['Pc_reshaped'][0, 0].shape[0]
n_dim = S_list[sim_idx]['x'][0, 0].shape[0]

# optimize_ELBO(n_dim, n_corr_obs, n_total_obs, verbose=True, n_starts=3)
elbo = ELBO(n_dim, n_corr_obs, n_total_obs, n_starts=1, n_iterations=2)
elbo.optimize_ELBO_SGD()

