import numpy as np

def forward_simulation(x, n_reps, var=0.1, abort_prob=0.1):
    """
    Simulate perceptual AXB responses given an observer model, i.e., a trajectory.  

    Inputs:
    ------
    x: (n_dim x n_frames) Numpy array
        Perceptual locations
    n_reps: Scalar
        Maximum number of repetitions per condition, e.g., 10 repetitions for each frame pair combination
    var: Float
        Variance. Default is 0.1 to encourage the synthesized data to be almost identical to the observer model. 
    abort_prob: Float
        Probability of aborted trials. Default is 0.1.

    Outputs:
    --------
    prop_corr_reshaped: (n_frames x n_frames) Numpy array
        Discriminability matrix containing probability of correct responses for each frame combination
    num_trials_mat_reshaped: (n_frames x n_frames) Numpy array
        Number of repetitions for each frame combination
    """

    n_dim = x.shape[0]
    n_frames = x.shape[-1]

    # get all pairwise combinations
    i, j = np.meshgrid(np.arange(0, n_frames), np.arange(0, n_frames))
    all_pairs = np.stack([j.ravel(), i.ravel()], axis=1)

    sigma = np.eye(n_dim) * var
    trial_mat = np.zeros((n_reps, all_pairs.shape[0]))
    n_pair_combs = all_pairs.shape[0] 

    for i_pair in range(n_pair_combs):
        for i_reps in range(n_reps):
            if np.random.rand() > abort_prob:
                sim_A = np.random.multivariate_normal(x[:, all_pairs[i_pair, 0]], sigma) # frame A TODO: try sim_A = x[]
                sim_B = np.random.multivariate_normal(x[:, all_pairs[i_pair, 1]], sigma) # frame B TODO: try sim_B = x[]
                if i_reps % 2 == 0:
                    sim_X = np.random.multivariate_normal(x[:, all_pairs[i_pair, 0]], sigma) # X = A
                    dist_AX = np.linalg.norm(sim_A - sim_X)
                    dist_BX = np.linalg.norm(sim_B - sim_X)
                    if dist_AX < dist_BX:
                        trial_mat[i_reps, i_pair] = 1 # correct response
                    else:
                        trial_mat[i_reps, i_pair] = 0 # incorrect response
                else:
                    sim_X = np.random.multivariate_normal(x[:, all_pairs[i_pair, 1]], sigma) # X = A
                    dist_AX = np.linalg.norm(sim_A - sim_X)
                    dist_BX = np.linalg.norm(sim_B - sim_X)
                    if dist_BX < dist_AX:
                        trial_mat[i_reps, i_pair] = 1 # correct response
                    else:
                        trial_mat[i_reps, i_pair] = 0 # incorrect response

    # calculate proportion correct and number of completed trials
    prop_corr = np.zeros((n_pair_combs))
    num_trials_mat = np.zeros((n_pair_combs))
    for i in range(n_pair_combs):
        prop_corr[i] = np.mean(trial_mat[:, i])
        num_trials_mat[i] = np.sum(trial_mat[:, i])
    
    prob_corr_reshaped = prop_corr.reshape((n_frames, n_frames))
    num_trials_mat_reshaped = num_trials_mat.reshape((n_frames, n_frames))

    return prob_corr_reshaped, num_trials_mat_reshaped