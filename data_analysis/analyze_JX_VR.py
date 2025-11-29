from pathlib import Path
import scipy.io as spio
import numpy as np
import os
import sys
import torch
import time
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from modules.elbo import ELBO
from modules import forward_simulation, construct_null_trajectory

# load data
data_path = Path('data') / 'JX_data_VR'
save_path = Path('data') / 'JX_results_VR'
save_path.mkdir(parents=True, exist_ok=True)
# out_csv = save_path / 'curvatures_pilot_JX_VR.csv'

n_bootstraps = 1
n_reps = 10
n_starts = 10
n_iterations = 1
# c_pixel = torch.Tensor([113.4483, 113.4134, 113.1497, 113.3503, 115.4514, 112.6135, 113.1197, 112.2594, 110.3757]) # pebbles
c_pixel = torch.Tensor([111.3045, 109.1879, 109.4946, 107.8206, 111.5625, 150.1991, 139.4365, 111.3499, 108.4367]) # park

data_files = sorted(os.listdir(data_path))

# run inference
t = time.perf_counter()
for i in range(n_bootstraps):
    print('-----------------------------------')
    print(f'BOOTSTRAP: {i+1}')
    for f_name in data_files:
        out_csv = Path(save_path) / f"{f_name.split('.')[0]}.csv"
        if not f_name.endswith('.mat'):
            continue

        S = spio.loadmat(Path(data_path) / f_name)
        n_total_obs = S['totalCounts']
        n_corr_obs = np.nan_to_num(n_total_obs * S['pctCorrect'])
        n_dim = n_total_obs.shape[0] - 1
        n_frames = n_total_obs.shape[0]

        try:
            elbo = ELBO(n_dim, n_corr_obs, n_total_obs, n_starts=n_starts, n_iterations=n_iterations)
            _, _, _, _, _, _, _, _, _, _, _, c_est = elbo.optimize_ELBO_SGD()
            c_est = torch.rad2deg(torch.mean(c_est)).detach().numpy()

            x_null, c_est_null, p_null = construct_null_trajectory('', n_dim, elbo._transform(elbo.mu_post_d, 'd'), elbo.mu_post_a, True, n_frames, c_pixel)

            prop_corr_null_sim, n_total_obs_null_sim = forward_simulation(x_null.detach().squeeze(), n_reps, var=1)
            n_corr_obs_null = np.round(n_total_obs_null_sim * prop_corr_null_sim)

            elbo_null = ELBO(n_dim, n_corr_obs_null, n_total_obs_null_sim, n_starts=n_starts, n_iterations=n_iterations, verbose=False)
            _, _, _, _, _, _, _, _, _, _, _, c_est_null = elbo_null.optimize_ELBO_SGD()
            c_est_null = torch.rad2deg(torch.mean(c_est_null)).detach().numpy()

            # save data
            with open(out_csv, 'a') as f:
                f.write(f"{c_est_null},{c_est}\n")
        except:
            print('Something went wrong')
        finally:
            continue

elapsed = time.perf_counter() - t
print(f"Finished in {elapsed:.2f} sec")