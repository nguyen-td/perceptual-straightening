from pathlib import Path
import scipy.io as spio
import numpy as np
import os
import sys
import torch
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from concurrent.futures import ProcessPoolExecutor, as_completed

from modules.elbo import ELBO

def process_sim(S_dat, save_path):
    try:
        # manually set diagonals to 0.5 (chance level if A=X=B)
        np.fill_diagonal(S_dat['Pc_reshaped'][0, 0], 0.5)

        n_corr_obs = S_dat['Pc_reshaped'][0, 0] * S_dat['num_trials_mat'][0, 0] 
        n_total_obs = S_dat['num_trials_mat'][0, 0]
        n_frames = S_dat['Pc_reshaped'][0, 0].shape[0]
        n_dim = 5
        c_true = np.mean(np.rad2deg(S_dat['c'][0, 0]).flatten())

        elbo = ELBO(n_dim, n_corr_obs, n_total_obs, n_starts=1, n_iterations=2)
        _, _, _, _, _, _, _, _, _, _, _, c_est = elbo.optimize_ELBO_SGD()

        # convert curvature estimates to degrees
        c_est_deg = torch.rad2deg(torch.mean(c_est)).detach().numpy() 
        c_true_deg = np.rad2deg(c_true)

        # save file
        n_reps = S_dat['generative_params'][0, 0]['n_reps'][0,0]
        n_frames = S_dat['generative_params'][0, 0]['n_frames'][0,0]

        f_name = Path(save_path) / f'curvatures_nframes-{int(n_frames[0, 0])}_nreps-{int(n_reps[0, 0])}.csv'
        with open(f_name, 'ab') as f:  # 'ab' = append in binary mode
            np.savetxt(f, np.array([[c_est_deg, c_true_deg]]), delimiter=',')

        return True

    except Exception as e:
        print(f'Something went wrong: {e}')
        return False


if __name__ == "__main__":
    # load sample trajectory
    data_path = Path('data') / 'sim' / 'jiaming'
    save_path = Path('data') / 'jiaming_results'
    save_path.mkdir(parents=True, exist_ok=True)

    data = sorted(os.listdir(data_path))

    for item in data:
        S = spio.loadmat(Path(data_path) / item)['S']
        S_list = [S[0, i] for i in range(S.shape[1])]  # convert to list of structs

        # unpack data in parallel
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(process_sim, S_dat, save_path) for S_dat in S_list]
            for f in as_completed(futures):
                f.result()  # trigger exception if any
