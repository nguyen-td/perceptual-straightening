from pathlib import Path
import scipy.io as spio
import numpy as np
import os
import sys
import torch
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from modules.elbo import ELBO

# load sample trajectory
data_path = Path('data') / 'sim' / 'jiaming'
save_path = Path('data') / 'jiaming_results'
save_path.mkdir(parents=True, exist_ok=True)

data = sorted(os.listdir(data_path))

for item in data:
    S = spio.loadmat(Path(data_path) / item)['S']
    S_list = [S[0, i] for i in range(S.shape[1])]  # convert to list of structs

    # unpack data
    curvatures = np.zeros((len(S_list), 2))
    sim_idx = 0
    print(item)
    try:
        np.fill_diagonal(S_list[sim_idx]['Pc_reshaped'][0, 0], 0.5) # manually set diagonals to 0.5 (chance level if A=X=B)
        n_corr_obs = S_list[sim_idx]['Pc_reshaped'][0, 0] * S_list[sim_idx]['num_trials_mat'][0, 0] 
        n_total_obs = S_list[sim_idx]['num_trials_mat'][0, 0]
        n_frames = S_list[sim_idx]['Pc_reshaped'][0, 0].shape[0]
        n_dim = 5
        c_true = np.mean(np.rad2deg(S_list[sim_idx]['c'][0, 0]).flatten())

        elbo = ELBO(n_dim, n_corr_obs, n_total_obs, n_starts=10, n_iterations=40000)
        _, _, _, _, _, _, _, _, _, _, _, c_est = elbo.optimize_ELBO_SGD()

        curvatures[sim_idx, 0] = torch.rad2deg(torch.mean(c_est)).detach().numpy() 
        curvatures[sim_idx, 1] = np.rad2deg(c_true)

        # save file
        n_reps = S_list[sim_idx]['generative_params'][0, 0]['n_reps'][0,0]
        n_frames = S_list[sim_idx]['generative_params'][0, 0]['n_frames'][0,0]

        f_name = Path(save_path) / f'curvatures_nframes-{int(n_frames[0, 0])}_nreps-{int(n_reps[0, 0])}.csv'
        with open(f_name, 'ab') as f:  # 'ab' = append in binary mode
            np.savetxt(f, curvatures, delimiter=',')

    except:
        print('Something went wrong')
    finally:
        continue
