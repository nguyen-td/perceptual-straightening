from pathlib import Path
import os
import numpy as np
import torch

from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

# # set device (CPU or GPU)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f'Device (CPU or GPU): ', device)

def run_bootstrap(iboot, n_dim, n_corr_obs, n_total_obs, n_starts, n_iterations, category, stim_folder, n_reps, n_frames):
    """
    Function to run one entire iterations (one bootstrap). Needs to import packages separately because each process runs in isolation.
    """
    import torch
    import numpy as np
    import time
    from modules.elbo import ELBO
    from modules import forward_simulation, construct_null_trajectory

    try:
        t = time.perf_counter()
        print(f"Bootstrap: {iboot}")

        # run estimation on real data
        elbo = ELBO(n_dim, n_corr_obs, n_total_obs, n_starts=n_starts, n_iterations=n_iterations, verbose=False)
        _, _, _, _, _, _, _, _, _, _, _, c_est = elbo.optimize_ELBO_SGD()
        c_est_val = torch.rad2deg(torch.mean(c_est)).detach().numpy()

        # replace perceptual curvature with pixel-domain curvature
        is_natural = True if category == 'natural' else False
        x_null, c_est_null, p_null = construct_null_trajectory(
            stim_folder, n_dim, elbo._transform(elbo.mu_post_d, 'd'),
            elbo.mu_post_a, is_natural, n_frames
        )

        # synthesize data from null observer
        prop_corr_null_sim, n_total_obs_null_sim = forward_simulation(x_null.detach().squeeze(), n_reps, var=1)
        n_corr_obs_null = np.round(n_total_obs_null_sim * prop_corr_null_sim)

        # compute null model
        elbo_null = ELBO(n_dim, n_corr_obs_null, n_total_obs_null_sim, n_starts=n_starts, n_iterations=n_iterations, verbose=False)
        _, _, _, _, _, _, _, _, _, _, _, c_est_null = elbo_null.optimize_ELBO_SGD()
        c_est_null_val = torch.rad2deg(torch.mean(c_est_null)).detach().numpy()

        elapsed = time.perf_counter() - t
        print(f"Bootstrap {iboot} finished in {elapsed:.2f} sec")

        return iboot, c_est_null_val, c_est_val

    except Exception as e:
        print(f"Bootstrap {iboot} failed: {e}")
        traceback.print_exc()
        return iboot, np.nan, np.nan


f_path = Path('data') / 'behavioral_data_09052025' / 'pt'
data = sorted(os.listdir(f_path))

# for isubj in range(10, len(df)):
for f_name in data:
    if f_name.endswith('.pt'):

        subject = f_name.split('_')[0]
        category = f_name.split('_')[1]
        eccentricity = f_name.split('_')[2]
        movie_id = int(f_name.split('_')[3])
        dat_movie_name = f_name.split('_')[4].split('.')[0]

        match eccentricity:
            case 'fovea':
                diameter = 6
            case 'parafovea':
                diameter = 24
            case _:
                diameter = 36
        match dat_movie_name:
            case 'DAM':
                stim_movie_name = 'carnegie-dam'
            case 'PRAIRIE':
                stim_movie_name = 'prairie1'
            case 'EGOMOTION':
                stim_movie_name = 'egomotion'
            case _:
                continue

        # file where data will be stored
        save_path = Path('data') / 'yoon_results_unanalyzed'
        save_path.mkdir(parents=True, exist_ok=True)
        f_name = Path(save_path) / f'curvatures_{subject}_{category}_{eccentricity}_{movie_id:02d}_{dat_movie_name}.csv'

        # optimization settings
        n_bootstraps = 10
        n_iterations = 40000
        n_reps = 10
        n_starts = 10 # multistarts for MLE, used for initializing posterior of the hierarchical variational inference algorithm

        # load data
        dat = torch.load(Path(f_path) / f'{subject}_{category}_{eccentricity}_{movie_id:02d}_{dat_movie_name}.pt')
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

        if __name__ == "__main__":
            import multiprocessing
            multiprocessing.freeze_support()  

            curvatures = np.zeros((n_bootstraps, 2)) # 1st column: c_null, 2nd colum: c_est; both columns are independent of each other and the order within columns does not matter

            with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = [
                    executor.submit(
                        run_bootstrap, iboot, n_dim, n_corr_obs, n_total_obs, n_starts, n_iterations, category, stim_folder, n_reps, n_frames)
                    for iboot in range(n_bootstraps)
                ]

                for future in as_completed(futures):
                    iboot, c_null, c_est = future.result()
                    curvatures[iboot, 0] = c_null
                    curvatures[iboot, 1] = c_est
                    np.savetxt(f_name, curvatures, delimiter=',')
