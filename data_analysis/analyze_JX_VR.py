from pathlib import Path
import scipy.io as spio
import numpy as np
import os
import sys
import torch
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from modules.elbo import ELBO

# load data
data_path = Path('data') / 'JX_VR'
save_path = Path('data') / 'JX_results'
save_path.mkdir(parents=True, exist_ok=True)
out_csv = save_path / 'curvatures_pilot_JX_VR.csv'

S = spio.loadmat(Path(data_path) / 'JX_pilot_normal_discriminability_mats.mat')
n_total_obs = S['totalCounts']
n_corr_obs = np.nan_to_num(n_total_obs * S['pctCorrect'])
n_dim = n_total_obs.shape[0] - 1

n_bootstraps = 10

# run inference
for i in range(n_bootstraps):
    try:
        elbo = ELBO(n_dim, n_corr_obs, n_total_obs, n_starts=10, n_iterations=40000)
        _, _, _, _, _, _, _, _, _, _, _, c_est = elbo.optimize_ELBO_SGD()

        # save data
        with open(out_csv, 'a') as f:
            f.write(f"{torch.rad2deg(torch.mean(c_est)).detach().numpy() }\n")
    except:
        print('Something went wrong')
    finally:
        continue