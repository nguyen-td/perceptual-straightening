from pathlib import Path
import os
import numpy as np
import torch

from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback


def run_bootstrap(iboot, n_dim, n_corr_obs, n_total_obs, n_starts, n_iterations, category, stim_folder, n_reps, n_frames):
    """
    Runs one bootstrap iteration for one dataset. Executed inside a separate process.
    """
    import torch
    import numpy as np
    import time
    from modules.elbo import ELBO
    from modules import forward_simulation, construct_null_trajectory

    try:
        t = time.perf_counter()
        print(f"Bootstrap {iboot}: running")

        # run estimation on real data
        elbo = ELBO(n_dim, n_corr_obs, n_total_obs,
                    n_starts=n_starts, n_iterations=n_iterations,
                    verbose=False)

        _, _, _, _, _, _, _, _, _, _, _, c_est = elbo.optimize_ELBO_SGD()
        c_est_val = torch.rad2deg(torch.mean(c_est)).detach().numpy()

        # replace perceptual curvature with pixel-domain curvature
        is_natural = True if category == 'natural' else False

        x_null, c_est_null, p_null = construct_null_trajectory(
            stim_folder,
            n_dim,
            elbo._transform(elbo.mu_post_d, 'd'),
            elbo.mu_post_a,
            is_natural,
            n_frames
        )

        # synthesize data from null observer
        prop_corr_null_sim, n_total_obs_null_sim = forward_simulation(
            x_null.detach().squeeze(), n_reps, var=1
        )
        n_corr_obs_null = np.round(n_total_obs_null_sim * prop_corr_null_sim)

        # compute null model
        elbo_null = ELBO(n_dim, n_corr_obs_null, n_total_obs_null_sim,
                         n_starts=n_starts, n_iterations=n_iterations,
                         verbose=False)
        _, _, _, _, _, _, _, _, _, _, _, c_est_null = elbo_null.optimize_ELBO_SGD()
        c_est_null_val = torch.rad2deg(torch.mean(c_est_null)).detach().numpy()

        elapsed = time.perf_counter() - t
        print(f"Bootstrap {iboot} finished in {elapsed:.2f} sec")

        return iboot, c_est_null_val, c_est_val

    except Exception as e:
        print(f"Bootstrap {iboot} failed: {e}")
        traceback.print_exc()
        return iboot, np.nan, np.nan


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    f_path = Path('data') / 'behavioral_data_09052025' / 'pt'
    data_files = sorted(os.listdir(f_path))

    # global optimization settings
    n_bootstraps = 100
    n_iterations = 40000
    n_reps = 10
    n_starts = 10

    # prepare output directory
    save_dir = Path('data') / 'yoon_results_unanalyzed'
    save_dir.mkdir(parents=True, exist_ok=True)

    for iboot in range(n_bootstraps):
        futures = {}

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            for f_name in data_files:

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

                out_csv = save_dir / f"curvatures_{subject}_{category}_{eccentricity}_{movie_id:02d}_{dat_movie_name}.csv"

                dat = torch.load(Path(f_path) / f_name)

                stim_folder = os.path.join('data', 'yoon_stimulus', f'diameter_{diameter:02d}_deg', f'movie{movie_id:02d}-{stim_movie_name}')

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

                # schedule this dataset for bootstrap iteration
                fut = executor.submit(
                    run_bootstrap, iboot, n_dim, n_corr_obs, n_total_obs, n_starts, n_iterations, category, stim_folder, n_reps, n_frames)
                futures[fut] = out_csv

            # collect result for this bootstrap
            for future in as_completed(futures):
                out_csv = futures[future]
                iboot_res, c_null_val, c_est_val = future.result()

                # append ONE LINE to the CSV for this dataset
                with open(out_csv, 'a') as f:
                    f.write(f"{c_null_val},{c_est_val}\n")

        print(f"Bootstrap {iboot} completed for all datasets.\n")
