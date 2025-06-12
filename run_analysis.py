import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import os

from modules.elbo import ELBO
from modules import optimize_null, forward_simulation

# load data
subject = 'ryan'
category = 'natural'
eccentricity = 'periphery'
movie_id = 6
diameter = 24; # 6, 24, 36
dat_movie_name = 'DAM'   # movie name as in the data file name
stim_movie_name = 'carnegie-dam' # movie name as in the stimulus file name
n_bootstraps = 1

dat = torch.load(Path('data') / 'yoon_data' / f'{subject}_{category}_{eccentricity}_{movie_id:02d}_{dat_movie_name}.pt')
stim_folder = os.path.join('data', 'yoon_stimulus', f'diameter_{diameter:02d}_deg', f'movie{movie_id:02d}-{stim_movie_name}')

# create trial matrices
n_frames = min(max(dat[:, 1]), max(dat[:, 2]))
n_trials = len(dat[:, 0])

n_total_obs = np.zeros((n_frames, n_frames))
n_corr_obs = np.zeros((n_frames, n_frames))

for iboot in range(n_bootstraps):
    # compute null models
    for itrial in range(n_trials):
        a_frame = dat[itrial, 1]
        b_frame = dat[itrial, 2]
        true_frame = a_frame if dat[itrial, 5] == 1 else b_frame
        pred_frame = a_frame if dat[itrial, 6] == 1 else b_frame

        n_total_obs[a_frame-1, b_frame-1] += 1
        n_corr_obs[a_frame-1, b_frame-1] += 1 if true_frame == pred_frame else 0

    prob_corr = np.divide(n_corr_obs, n_total_obs, out=np.zeros_like(n_corr_obs), where=n_total_obs!=0)

    is_natural = True if category == 'natural' else False
        
    n_dim = n_frames - 1
    x_null, c_pixel, c_null, prob_corr_human, prob_corr_null = optimize_null(stim_folder, n_corr_obs, n_total_obs, n_dim=n_dim, n_starts=10, n_iter=10000, n_frames=n_frames, is_natural=is_natural, version=1, disp=False)

    # synthesize data from null models
    n_reps = 10
    prob_corr_null_sim, n_total_obs_null_sim = forward_simulation(x_null.squeeze(), n_reps)

    n_corr_obs_null = np.round(n_total_obs_null_sim * prob_corr_null_sim)

    elbo_null = ELBO(n_dim, n_corr_obs_null, n_total_obs_null_sim, n_starts=10, n_iterations=80000)
    x_null_elbo, p_null_elbo, errors_null_elbo, kl_loss_null, ll_loss_null, c_prior_null, d_prior_null, l_prior_null, _, _, _, c_est_null = elbo_null.optimize_ELBO_SGD()