import torch
from torch import nn
import numpy as np
from scipy.spatial import distance
from scipy.stats import norm
from scipy.optimize import minimize
import torch.distributions as D
# from pybads import BADS

from modules import compute_trajectory_perceptual
from utils import make_positive_definite

def optimize_MLE(n_dim, n_corr_obs, n_total_obs, lr=1e-4, n_iter=1000, verbose=True, n_starts=10):
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
    verbose: Boolean
        If True, outputs progress bar.
    n_starts: Scalar
        Number of multistarts

    Outputs:
    --------
    loss: (n_runs, ) Numpy array
        Contains negative likelihood loss, n_runs is either the same as n_iter or less (stopping criterion: abs(loss(t) - loss(t-1)) < 1e-3)
    x: (n_dim x n_frames) Torch tensor
        Estimated perceptual locations
    c_est: (n_frames - 2) Torch tensor
        Estimated local curvatures
    p: (n_frames x n_frames) Numpy array
        Estimated proportion correct
    c: (n_frames - 2) Torch tensor
        Curvature vector
    d: (n_frames - 1) Torch tensor
        Distance vector
    a: (n_dim x n_frames - 2) Torch tensor
        Orthonormal acceleration vectors
    inv_hess: (n_params x n_params) Numpy array
        Inverse of the Hessian matrix evaluated at MLE, corresponds to Fisher Information Matrix because NLL is minimized
    """

    n_frames = n_corr_obs.shape[0]

    def func_NLL(vec):
        '''
        Returns negative log likelihood. 
        '''
        # get perceptual locations
        c_ = torch.tensor(vec[:n_frames - 2]).unsqueeze(0)
        d_ = torch.tensor(vec[n_frames-2:n_frames-2 + n_frames-1]).unsqueeze(0)
        a_ = torch.tensor(start_vec[n_frames-2 + n_frames-1:].reshape(n_dim, n_frames - 2)).unsqueeze(0)
        x, _, _, _, _ = compute_trajectory_perceptual(1, n_frames, n_dim, d_, c_, a_)

        # get perceptual distances
        dist = torch.cdist(torch.transpose(x, 1, 2), torch.transpose(x, 1, 2))

        # compute hierarchical model
        normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0])) # cdf of the standard normal
        p_axb = normal.cdf(dist / np.sqrt(2)) * normal.cdf(dist / 2) + normal.cdf(-dist / np.sqrt(2)) * normal.cdf(-dist / 2)
        p = (1 - 2 * l) * p_axb.clone() + l
        return -torch.sum((torch.tensor(n_corr_obs) * torch.log(p.clone())) + (torch.tensor(n_total_obs) - torch.tensor(n_corr_obs)) * torch.log(1 - p.clone())).detach().numpy()

    # run optimization using a multistart procedure
    old_loss = np.array([10e10])
    n_free_params = (n_frames - 2) + (n_frames - 1) + n_dim * (n_frames - 2)
    inv_hess = np.zeros((n_free_params, n_free_params))

    for i in range(n_starts):

        # initialize parameters
        c_init = np.deg2rad(np.abs(np.random.normal(60, 10, size=((n_frames - 2)))))
        d_init = np.abs(np.random.normal(1, 0.5, size=((n_frames - 1))))
        a_init = np.random.normal(0, 2, size=((n_dim, n_frames - 2)))
        l = 0.06; # guess rate
        start_vec = np.concatenate((c_init, d_init, a_init.flatten()))

        LB = np.zeros(np.size(start_vec))
        UB = np.zeros(np.size(start_vec))

        # bounds for c
        LB[:n_frames-2] = 0
        UB[:n_frames-2] = np.pi

        # bounds for d
        LB[n_frames-2:n_frames-2 + n_frames-1] = 0
        UB[n_frames-2:n_frames-2 + n_frames-1] = 3

        # bounds for a
        LB[n_frames-2 + n_frames-1:] = -100
        UB[n_frames-2 + n_frames-1:] = 100

        # set bounds
        bnds = np.zeros((np.size(LB), 2))
        bnds[:, 0] = LB
        bnds[:, 1] = UB

        res = minimize(func_NLL, start_vec, bounds=tuple(map(tuple, bnds)), options={'maxiter': n_iter, 'disp': False})
        if res['success']:
            new_loss = res.fun
            if verbose:
                print(f'Current loss: {res.fun}')

            if new_loss < old_loss:
                # reconstruct trajectory
                c = torch.tensor(res.x[:n_frames - 2]).unsqueeze(0)
                d = torch.tensor(res.x[n_frames-2:n_frames-2 + n_frames-1]).unsqueeze(0)
                a = torch.tensor(res.x[n_frames-2 + n_frames-1:].reshape(n_dim, n_frames - 2)).unsqueeze(0)
                x, _, c_est, _, _ = compute_trajectory_perceptual(1, n_frames, n_dim, d, c, a)

                # get perceptual distances
                dist = torch.cdist(torch.transpose(x, 1, 2), torch.transpose(x, 1, 2))

                # compute hierarchical model
                normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0])) # cdf of the standard normal
                p_axb = normal.cdf(dist / np.sqrt(2)) * normal.cdf(dist / 2) + normal.cdf(-dist / np.sqrt(2)) * normal.cdf(-dist / 2)
                p = (1 - 2 * l) * p_axb.clone() + l
                inv_hess = res.hess_inv.todense()

                old_loss = new_loss
                
                if verbose:
                    print('Loss updated')
        if verbose:
            print(f"Iteration {i+1} | Loss: {res.fun}")

    return x, c_est, p, c, d, a, inv_hess