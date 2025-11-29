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
data_path = Path('data') / 'behavioral_data_09052025' / 'pt'
save_path = Path('data') / 'YB_results_unanalyzed'
save_path.mkdir(parents=True, exist_ok=True)
# out_csv = save_path / 'curvatures_pilot_JX_VR.csv'

n_bootstraps = 2
n_reps = 10
n_starts = 1
n_iterations = 1

data_files = sorted(os.listdir(data_path))

# run inference
t = time.perf_counter()
for iboot in range(n_bootstraps):
    print('-----------------------------------')
    print(f'BOOTSTRAP: {iboot+1}')
    for f_name in data_files:
        out_csv = Path(save_path) / f"{f_name.split('.')[0]}.csv"
        if not f_name.endswith('.pt'):
            continue

        subject, category, eccentricity, movie_id_str, dat_movie_name_ext = f_name.split('_')
        movie_id = int(movie_id_str)
        dat_movie_name = dat_movie_name_ext.split('.')[0]

        # eccentricity → diameter
        match eccentricity:
            case 'fovea': diameter = 6
            case 'parafovea': diameter = 24
            case _: diameter = 36

        # movie name mapping
        match dat_movie_name:
            case 'DAM': stim_movie_name = 'carnegie-dam'
            case 'PRAIRIE': stim_movie_name = 'prairie1'
            case 'EGOMOTION': stim_movie_name = 'egomotion'
            case _: continue  # skip unknown movies
        
        out_csv = save_path / f"curvatures_{subject}_{category}_{eccentricity}_{movie_id:02d}_{dat_movie_name}.csv"
        dat = torch.load(Path(data_path) / f_name)
        stim_folder = os.path.join('data', 'YB_stimulus_behavior', f'diameter_{diameter:02d}_deg', f'movie{movie_id:02d}-{stim_movie_name}')
        # stim_folder = os.path.join('/Users', 'tn22693', 'Dropbox', 'V1V2_straightening_2024', 'psychophysics', 'perceptual_results', 'stimulus', 'experiment', f'diameter_{diameter:02d}_deg', f'movie{movie_id:02d}-{stim_movie_name}')

        # create trial matrices
        n_frames = min(int(dat[:, 1].max()), int(dat[:, 2].max()))
        n_trials = dat.shape[0]
        n_dim = n_frames - 1

        n_total_obs = np.zeros((n_frames, n_frames))
        n_corr_obs = np.zeros((n_frames, n_frames))

        for i in range(n_trials):
            a = int(dat[i, 1])
            b = int(dat[i, 2])
            true_frame = a if dat[i, 5] == 1 else b
            pred_frame = a if dat[i, 6] == 1 else b

            n_total_obs[a - 1, b - 1] += 1
            n_corr_obs[a - 1, b - 1] += (true_frame == pred_frame)

        try:
            # run estimation on real data
            elbo = ELBO(n_dim, n_corr_obs, n_total_obs, n_starts=n_starts, n_iterations=n_iterations, verbose=False)
            _, _, _, _, _, _, _, _, _, _, _, c_est = elbo.optimize_ELBO_SGD()
            c_est_val = torch.rad2deg(torch.mean(c_est)).detach().numpy()

            # replace perceptual curvature with pixel-domain curvature
            is_natural = True if category == 'natural' else False

            x_null, c_est_null, p_null = construct_null_trajectory(stim_folder, n_dim, elbo._transform(elbo.mu_post_d, 'd'), elbo.mu_post_a, is_natural, n_frames)

            # synthesize data from null observer
            prop_corr_null_sim, n_total_obs_null_sim = forward_simulation(x_null.detach().squeeze(), n_reps, var=1)
            n_corr_obs_null = np.round(n_total_obs_null_sim * prop_corr_null_sim)

            # compute null model
            elbo_null = ELBO(n_dim, n_corr_obs_null, n_total_obs_null_sim, n_starts=n_starts, n_iterations=n_iterations, verbose=False)
            _, _, _, _, _, _, _, _, _, _, _, c_est_null = elbo_null.optimize_ELBO_SGD()
            c_est_null_val = torch.rad2deg(torch.mean(c_est_null)).detach().numpy()

            elapsed = time.perf_counter() - t
            print(f"Bootstrap {iboot} finished in {elapsed:.2f} sec")

            # save data
            with open(out_csv, 'a') as f:
                f.write(f"{c_est_null_val},{c_est_val}\n")
        except:
            print('Something went wrong')
        finally:
            continue