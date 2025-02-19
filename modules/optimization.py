import torch
from torch import nn
import numpy as np
from modules import compute_trajectory

def run_MLEfit(n_dim, n_corr_obs, n_total_obs, lr=1e-4, n_iter=1000):
    """
    Run maximum likelihood estimation (MLE) to estimate perceptual trajectories from perceptual observations. Uses polar coordinates.

    Inputs:
    -------
    n_dim: Scalar
        Number of dimensions
    n_corr_obs: (n_frames x n_frames) Numpy array
        Matrix where each entry corresponds to the number of correct observations/choices for the respective frame combination
    n_total_obs: (n_frames x n_frames) Numpy array
        Matrix where each entry corresponds to the number of completed trials for the respective frame combination
    lr: Scalar
        Learning rate
    n_iter: Scalar
        Number of iterations

    Outputs:
    --------
    loss: (n_runs, ) Numpy array
        Contains negative likelihood loss, n_runs is either the same as n_iter or less (stopping criterion: abs(loss(t) - loss(t-1)) < 1e-3)
    x: (n_dim x n_frames) Torch tensor
        Estimated perceptual locations
    c_est: (n_frames - 2) Torch tensor
        Estimated local curvatures
    p: (n_frames x n_frames) Torch tensor
        Estimated proportion correct
    """

    n_frames = n_corr_obs.shape[0]

    # initialize parameters
    c = nn.Parameter(torch.squeeze(torch.abs(torch.deg2rad(torch.normal(60, 10, size=(1, n_frames - 2))))))
    d = nn.Parameter(torch.squeeze(torch.abs(torch.normal(1, 0.5 , size=(1, n_frames - 1)))))
    a = nn.Parameter(torch.normal(0, 2, size=(n_dim, n_frames - 2)))
    l = 0.06; # guess rate

    # initialize optimizer
    optimizer = torch.optim.SGD([c, d, a], lr=lr)

    # run optimization
    iter = n_iter
    loss = np.zeros(iter)

    for i in range(iter):
        
        # compute perceptual locations
        x, _, c_est, _, _ = compute_trajectory(n_frames, n_dim, d, c, a)

        # get perceptual distances
        dist = torch.cdist(x.T, x.T, p=2)

        # compute hierarchical model
        normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0])) 
        p_axb = normal.cdf(dist / torch.sqrt(torch.tensor(2))) * normal.cdf(dist / torch.tensor(2)) + normal.cdf(- dist / torch.sqrt(torch.tensor(2))) * normal.cdf(-dist / torch.tensor(2))
        p = (1 - 2 * l) * p_axb + l
        NLL = -torch.sum(torch.tensor(n_corr_obs) * torch.log(p) + (torch.tensor(n_total_obs) - torch.tensor(n_corr_obs)) * torch.log(1 - p))

        # gradient update
        NLL.backward()
        optimizer.step()
        optimizer.zero_grad()

        # save error
        loss[i] = NLL.item()

        if not i % 100:
            print(f"Epoch: {i}, Loss: {loss[i]}")

        if i > 0:
            if (np.abs(loss[i] - loss[i-1])) < 1e-3:
                loss = loss[:i]
                break

    print(f'Estimated average global curvature: {torch.rad2deg(torch.mean(c_est))}')

    return loss, x, c_est, p