import torch
from torch import nn
from pathlib import Path
import numpy as np
import scipy

from modules import ELBO
from utils import load_sim_data, log_likelihood

import os
import matlab.engine

def compute_curvature(x, N):
    """
    Directly compute the curvature form the perceptual trajectories.

    Inputs:
    -------
    x: (N x (N - 1)) torch tensor
        Array corresponding to the inferred perceptual trajectory, where the second dimension corresponds to the number of dimensions.
    N: Scalar
        Number of frames

    Outputs:
    --------
    c: (N) numpy array
        Inferred average curvature
    """

    v_hat = torch.zeros(N-1, N-1) # first index is T, second index is the number of dimensions
    c = torch.zeros(N-2)
    
    for t in range(1, N):
        v = x.squeeze()[t] - x.squeeze()[t-1]
        v_hat[t-1] = v / torch.linalg.norm(v)
    
    for t in range(1, N-1):
        c[t-1] = torch.arccos(v_hat[t-1] @ v_hat[t])

    return c

def direct_estimation(sim_dir, n_traj, n_frames=4, n_dim=3, n_iterations=1000):
    """
    Two-step greedy curvature estimation using maximum likelihood estimation, where L(n,m | x) is the likelihood function.

    Inputs:
    -------
    sim_dir: String
        String containing the path were the MATLAB simulation_py.mat script is located
    n_traj: Scalar
        Number of simulated trajectories
    n_frames: Scalar
        Number of frames
    n_dim: Scalar
        Number of dimensions
    n_iterations: Scalar
        Number of iterations

    Outputs:
    --------
    c_est: (n_traj, ) torch tensor
        Estimated average curvatures for each simulated trajectory
    c_true: (n_traj, ) torch tensor
        Ground truth curvatures (angles) for each simulated trajectory
    """

    os.chdir(sim_dir)
    # call MATLAB function to create trajectory

    c_est = torch.zeros(n_traj)
    c_true = (torch.rand(n_traj) * (torch.pi)) * (180 / torch.pi) # in degrees

    for i_traj in range(n_traj):
        eng = matlab.engine.start_matlab()
        ExpParam, Data, _ = eng.simulation_py(c_true[i_traj].item(), n_frames, n_dim, nargout=3)

        # stop MATLAB engine
        eng.quit()

        # extract data matrices
        trial_mat = torch.tensor(Data['resp_mat'])
        pair_inds = torch.tensor(ExpParam['all_pairs'])

        # initialize parameters
        N = n_frames
        l = 0.06 # lambda
        x = nn.Parameter(torch.rand(1, N, N - 1))

        # set up optimizer
        lr = 1e-4
        optimizer = torch.optim.SGD([{'params': x}], lr=lr)

        # run optimization
        iterations = n_iterations
        loss = np.zeros(iterations)
        for i in range(iterations):
            log_ll = -torch.sum(log_likelihood(N, trial_mat, pair_inds, x, l))

            # gradient update
            log_ll.backward()
            optimizer.step()
            optimizer.zero_grad()

            # save error
            loss[i] = log_ll.item()

            # if not i % 50:
            #     print(f"Epoch: {i}, Loss: {loss[i]}")
        
        c_est[i_traj] = torch.mean(compute_curvature(x, N)) * (180 / torch.pi)

        if not i_traj % 10:
            print(f"Iteration: {i_traj}")

    return c_est, c_true