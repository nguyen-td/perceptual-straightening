import os
import sys
from pathlib import Path
import scipy.io as spio
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules import compute_curvature_pixel
import numpy as np
import imageio.v2 as imageio 
import matplotlib.pyplot as plt
from modules.elbo import ELBO

# load sample trajectory
sim_idx = 0 # which out of the 100 trajectories to use
data_path = Path('data') / 'sim'
f_name = 'sim_0125.mat'
S = spio.loadmat(Path(data_path) / f_name)['S']
S_list = [S[0, i] for i in range(S.shape[1])]  # convert to list of structs

# unpack data
np.fill_diagonal(S_list[sim_idx]['Pc_reshaped'][0, 0], 0.5) # manually set diagonals to 0.5 (chance level if A=X=B)
n_corr_obs = S_list[sim_idx]['Pc_reshaped'][0, 0] * S_list[sim_idx]['num_trials_mat'][0, 0] 
n_total_obs = S_list[sim_idx]['num_trials_mat'][0, 0]
n_frames = S_list[sim_idx]['Pc_reshaped'][0, 0].shape[0]
n_dim = 5
c_true = np.mean(np.rad2deg(S_list[sim_idx]['c'][0, 0]).flatten())

# run inference
elbo = ELBO(n_dim, n_corr_obs, n_total_obs, n_starts=1, n_iterations=10)
x, p, errors, kl_loss, ll_loss, c_prior, d_prior, l_prior, c_post, d_post, l_post, c_est = elbo.optimize_ELBO_SGD()