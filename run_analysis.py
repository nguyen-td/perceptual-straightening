import torch
import numpy as np
from pathlib import Path
import os
import time
from numpy import genfromtxt

from modules.elbo import ELBO
from modules import optimize_null, forward_simulation

# # set device (CPU or GPU)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f'Device (CPU or GPU): ', device)

# data settings
subject = 'yb'
category = 'synthetic'
eccentricity = 'periphery'
movie_id = 6
diameter = 36; # 6, 24, 36
dat_movie_name = 'DAM'   # movie name as in the data file name
stim_movie_name = 'carnegie-dam' # movie name as in the stimulus file name

# file where data will be stored
save_path = Path('data') / 'yoon_results'
save_path.mkdir(parents=True, exist_ok=True)
f_name = Path(save_path) / f'curvatures_{subject}_{category}_{eccentricity}_{movie_id:02d}_{dat_movie_name}.csv'

# optimization settings
n_bootstraps = 100
n_iterations = 50000
n_starts = 10 # multistarts for MLE, used for initializing posterior of the hierarchical variational inference algorithm

# load data
dat = torch.load(Path('data') / 'yoon_data' / f'{subject}_{category}_{eccentricity}_{movie_id:02d}_{dat_movie_name}.pt')
stim_folder = os.path.join('data', 'yoon_stimulus', f'diameter_{diameter:02d}_deg', f'movie{movie_id:02d}-{stim_movie_name}')

# create trial matrices
n_frames = min(max(dat[:, 1]), max(dat[:, 2]))
n_trials = len(dat[:, 0])
n_dim = n_frames - 1

n_total_obs = np.zeros((n_frames, n_frames))
n_corr_obs = np.zeros((n_frames, n_frames))

for itrial in range(n_trials):
    a_frame = dat[itrial, 1]
    b_frame = dat[itrial, 2]
    true_frame = a_frame if dat[itrial, 5] == 1 else b_frame
    pred_frame = a_frame if dat[itrial, 6] == 1 else b_frame

    n_total_obs[a_frame-1, b_frame-1] += 1
    n_corr_obs[a_frame-1, b_frame-1] += 1 if true_frame == pred_frame else 0

# create bootstraps
curvatures = np.zeros((n_bootstraps, 2)) # 1st column: c_null, 2nd colum: c_est; both columns are independent of each other and the order within columns does not matter
# curvatures = genfromtxt(f_name, delimiter=',')

# for iboot in range(37, n_bootstraps):
for iboot in range(n_bootstraps):
    t = time.perf_counter()

    print(f'Bootstrap: {iboot} \n')
    
    # compute null model
    print('Compute null model: ...')
    is_natural = True if category == 'natural' else False
    x_null, c_pixel, c_null, prob_corr_human, prob_corr_null = optimize_null(stim_folder, n_corr_obs, n_total_obs, n_dim=n_dim, n_starts=10, n_iter=10000, n_frames=n_frames, is_natural=is_natural, version=1, disp=False)

    # synthesize data from null model
    print('\nSynthesize data from null model: ...')
    n_reps = 10
    prob_corr_null_sim, n_total_obs_null_sim = forward_simulation(x_null.squeeze(), n_reps)

    print('\nEstimate curvature from null model observer data: ...')
    n_corr_obs_null = np.round(n_total_obs_null_sim * prob_corr_null_sim)
    elbo_null = ELBO(n_dim, n_corr_obs_null, n_total_obs_null_sim, n_starts=n_starts, n_iterations=n_iterations, verbose=False)
    _, _, _, _, _, _, _, _, _, _, _, c_est_null = elbo_null.optimize_ELBO_SGD()
    curvatures[iboot, 0] = torch.rad2deg(torch.mean(c_est_null)).detach().numpy()

    # run estimation on real data
    print('\nEstimate curvature from human observer data: ...')
    elbo = ELBO(n_dim, n_corr_obs, n_total_obs, n_starts=n_starts, n_iterations=n_iterations, verbose=False)
    _, _, _, _, _, _, _, _, _, _, _, c_est = elbo.optimize_ELBO_SGD()
    curvatures[iboot, 1] = torch.rad2deg(torch.mean(c_est)).detach().numpy()

    # save file
    np.savetxt(f_name , curvatures, delimiter=',')

    elapsed = time.perf_counter() - t
    print(f'Elapsed: {elapsed} seconds.')

    print('---------------------------------------------------------------------------------')