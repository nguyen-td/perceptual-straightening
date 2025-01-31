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
    c: (N-2) torch tensor 
        Inferred average curvature
    d: (N-1) torch tensor
        Distances/length of displacement vector
    a: (N-2, N-1) torch tensor
        Acceleration
    """

    v_hat = torch.zeros(N-1, N-1) # first index is T, second index is the number of dimensions
    c = torch.zeros(N-2)
    a = torch.zeros(N-2, N-1)
    d = torch.zeros(N-1)
    
    for t in range(1, N):
        v = x.squeeze()[t] - x.squeeze()[t-1] 
        v_hat[t-1] = v / torch.linalg.norm(v)
        d[t-1] = torch.linalg.norm(v)
    
    for t in range(1, N-1):
        c[t-1] = torch.arccos(v_hat[t-1] @ v_hat[t])
        a[t-1] = (v_hat[t] - torch.cos(c[t-1] * v_hat[t-1])) / torch.sin(c[t-1])

    return c, d, a

def direct_estimation(sim_dir, n_traj, n_frames=4, n_dim=3, n_iterations=1000, n_trials=1000, sim_curvature=None, isdegree=False):
    """
    Two-step greedy curvature estimation using maximum likelihood estimation for recovery analysis, where L(n,m | x) is the likelihood function. The ground truth curvatures are generated randomly. 

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
    n_trials: Scalar
        Number of trials
    sim_curvature: Scalar (degrees) or None (default: None)
        Simulated curvature in degrees. If None is passed, then the curvatures are randomly sampled.
    isdegree: Boolean (default: False)
        If True, curvature is returned in degrees. If False, curvature is returned in radians.

    Outputs:
    --------
    c_est: (n_traj, ) torch tensor
        Estimated average curvatures for each simulated trajectory
    c_true: (n_traj, ) torch tensor
        Ground truth curvatures (angles) for each simulated trajectory
    c: (n_traj, N-2) torch tensor
        Estimated curvature for each simulated trajectory
    d: (n_traj, N-1) torch tensor
        Distances/length of displacement vector
    a: (n_traj, N-2, N-2) torch tensor
        Acceleration
    ExpParam: Dictionary
        Contains experimental parameters
    Data: Dictionary
        Contains the trial matrix with correct (1) and incorrect (0) responses
    Pc_reshaped: (n_frames x n_frames) torch tensor
        Discriminability matrix
    """

    os.chdir(sim_dir)
    # call MATLAB function to create trajectory

    c_est = torch.zeros(n_traj)
    if not isinstance(sim_curvature, (int, float)):
        c_true = torch.rad2deg(torch.rand(1) * (torch.pi)) 
    else:
        c_true = torch.zeros(1) + sim_curvature
    c = torch.zeros(n_traj, n_frames-2)
    a = torch.zeros(n_traj, n_frames-2, n_frames-1)
    d = torch.zeros(n_traj, n_frames-1)

    eng = matlab.engine.start_matlab()
    ExpParam, Data, Pc_reshaped = eng.simulation_py(int(c_true.item()), n_frames, n_dim, n_trials, nargout=3)

    # stop MATLAB engine
    eng.quit()

    # extract data matrices and save them
    trial_mat = torch.tensor(Data['resp_mat'])
    pair_inds = torch.tensor(ExpParam['all_pairs'])

    torch.save(trial_mat, Path('..') / 'data' / 'simulations' / f'trial_mat_{int(c_true.item())}.pt')
    torch.save(torch.tensor(Pc_reshaped), Path('..') / 'data' / 'simulations' / f'Pc_reshaped{int(c_true.item())}.pt')

    print(f'Proportion of correct responses: {torch.sum(trial_mat) / torch.numel(trial_mat)}')

    for i_traj in range(n_traj):
        # initialize parameters
        N = n_frames
        l = 0.06 # lambda
        x = nn.Parameter(torch.rand(1, N, N - 1) + 1.5) # range: [1, 2.5]

        # set up optimizer
        lr = 1e-4
        optimizer = torch.optim.SGD([{'params': x}], lr=lr)

        # run optimization
        loss = np.zeros(n_iterations)
        for i in range(n_iterations):
            log_ll = -torch.sum(log_likelihood(N, trial_mat, pair_inds, x, l))

            # gradient update
            log_ll.backward()
            optimizer.step()
            optimizer.zero_grad()

            ## save error
            # loss[i] = log_ll.item()

            # if not i % 50:
            #     print(f"Epoch: {i}, Loss: {loss[i]}")
        
        # c[i_traj], a[i_traj], d[i_traj] = compute_curvature(x, N)
        c[i_traj], d[i_traj], a[i_traj] = compute_curvature(x, N)
        c_est[i_traj] = torch.mean(c[i_traj])
        print(f"Trajectory: {i_traj}")

    if isdegree:
        c_est = torch.rad2deg(c_est)
        c = torch.rad2deg(c)
    
    return c_est, c_true, c, d, a, ExpParam, Data, torch.tensor(Pc_reshaped)

def curvature_estimation(sim_dir, n_traj, n_frames, n_dim, n_iterations=1000, eps=1e-6, lr=1e-4, sim_curvature=None):
    """
    Run the curvature estimation for recovery analysis. The ground truth curvatures are generated randomly. 

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
    eps: Scalar (default: 1e-6)
        Regularization factor to ensure numerical stability for computing the Cholesky decomposition
    lr: Scalar (default: 1e-4)
        Learning rate of the optimizer
    sim_curvature: Scalar (degrees) or None (default: None)
        Simulated curvature in degrees. If None is passed, then the curvatures are randomly sampled.

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
    if not isinstance(sim_curvature, (int, float)):
        c_true = (torch.rand(n_traj) * (torch.pi)) * (180 / torch.pi) # in degrees
    else:
        c_true = torch.zeros(n_traj) + sim_curvature 

    for i_traj in range(n_traj):
        # # load data
        # eng = matlab.engine.start_matlab()
        # ExpParam, Data, _ = eng.simulation_py(c_true[i_traj].item(), n_frames, n_dim, nargout=3)

        # # stop MATLAB engine
        # eng.quit()

        # # extract data matrices
        # trial_mat = torch.tensor(Data['resp_mat'])
        # pair_inds = torch.tensor(ExpParam['all_pairs'])

        # run direct estimation once to get initial values for c and d
        _, _, c, d, _, ExpParam, Data = direct_estimation(sim_dir, n_traj=1, n_frames=n_frames, n_dim=n_frames-1, n_iterations=n_iterations, sim_curvature=sim_curvature)

         # extract data matrices
        trial_mat = torch.tensor(Data['resp_mat'])
        pair_inds = torch.tensor(ExpParam['all_pairs'])

        # create initial values
        d_post_init = d.squeeze()

        c_post_init = torch.zeros(n_frames-1)
        c_post_init[1:] = c

        # initialize model
        model = ELBO(n_frames, d_post_init, c_post_init, eps)

        # set up optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # optimize ELBO
        for i in range(n_iterations):
            log_ll, d, c, a = model.compute_likelihood(trial_mat, pair_inds)
            kl = model.kl_divergence()
            loss = model.compute_loss(log_ll, kl)

            # gradient update
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        c_est[i_traj] = torch.mean(c).detach().numpy() * (180/np.pi)
        print(f"Trajectory: {i_traj}")

    return c_est, c_true

