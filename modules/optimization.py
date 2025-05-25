import torch
from torch import nn
import numpy as np
from scipy.spatial import distance
from scipy.stats import norm
from scipy.optimize import minimize
import torch.distributions as D
import imageio.v2 as imageio 
from pathlib import Path
import os

from modules import compute_trajectory_perceptual, compute_curvature_pixel
from utils import make_positive_definite

def optimize_MLE(n_dim, n_corr_obs, n_total_obs, n_iter=1000, verbose=True, n_starts=10, c_pixel=None, disp=False):
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
    n_iter: Scalar
        Number of iterations
    verbose: Boolean
        If True, outputs progress bar.
    n_starts: Scalar
        Number of multistarts
    c_pixel: (1 x n_frames - 2) Torch tensor or None
        If not None, c will not be a free parameter but will be set to 'c_pixel'. Can be used to compute null model where 'c' 
        is set to be the same as the pixel-domain curvature

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
        if c_pixel is None:
            c_ = torch.tensor(vec[:n_frames - 2]).unsqueeze(0)
            d_ = torch.tensor(vec[n_frames-2:n_frames-2 + n_frames-1]).unsqueeze(0)
            a_ = torch.tensor(start_vec[n_frames-2 + n_frames-1:].reshape(n_dim, n_frames - 2)).unsqueeze(0)
            x, _, _, _, _ = compute_trajectory_perceptual(1, n_frames, n_dim, d_, c_, a_)
        else:
            d_ = torch.tensor(vec[:n_frames-1]).unsqueeze(0)
            a_ = torch.tensor(vec[n_frames-1:].reshape(n_dim, n_frames - 2)).unsqueeze(0)
            x, _, _, _, _ = compute_trajectory_perceptual(1, n_frames, n_dim, d_, c_pixel, a_)

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
        if c_pixel is None:
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

        else: # c will not be a free parameter
            d_init = np.abs(np.random.normal(1, 0.5, size=((n_frames - 1))))
            a_init = np.random.normal(0, 2, size=((n_dim, n_frames - 2)))
            l = 0.06; # guess rate
            start_vec = np.concatenate((d_init, a_init.flatten()))

            LB = np.zeros(np.size(start_vec))
            UB = np.zeros(np.size(start_vec))

            # bounds for d
            LB[:n_frames-1] = 0
            UB[:n_frames-1] = 3

            # bounds for a
            LB[n_frames-1:] = -100
            UB[n_frames-1:] = 100

        # set bounds
        bnds = np.zeros((np.size(LB), 2))
        bnds[:, 0] = LB
        bnds[:, 1] = UB

        res = minimize(func_NLL, start_vec, bounds=tuple(map(tuple, bnds)), options={'maxiter': n_iter, 'disp': disp})
        if res['success']:
            new_loss = res.fun
            if verbose:
                print(f'Current loss: {res.fun}')

            if new_loss < old_loss:
                # reconstruct trajectory
                if c_pixel is None:
                    c = torch.tensor(res.x[:n_frames - 2]).unsqueeze(0)
                    d = torch.tensor(res.x[n_frames-2:n_frames-2 + n_frames-1]).unsqueeze(0)
                    a = torch.tensor(res.x[n_frames-2 + n_frames-1:].reshape(n_dim, n_frames - 2)).unsqueeze(0)
                    x, _, c_est, _, _ = compute_trajectory_perceptual(1, n_frames, n_dim, d, c, a)
                else:
                    d = torch.tensor(res.x[:n_frames-1]).unsqueeze(0)
                    a = torch.tensor(res.x[n_frames-1:].reshape(n_dim, n_frames - 2)).unsqueeze(0)
                    x, _, c_est, _, _ = compute_trajectory_perceptual(1, n_frames, n_dim, d, c_pixel, a)

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
        else:
            print(res.message)

        if verbose:
            print(f"Iteration {i+1} | Loss: {res.fun}")

    if c_pixel is None:
        return x, c_est, p, c, d, a, inv_hess
    else:
        return x, c_est, p

def optimize_null(stim_folder, n_corr_obs, n_total_obs, n_dim, n_starts=10, n_iter=1000, n_frames=11, is_natural=True, version=1, disp=False):
    """
    Compute null model where discriminabilities (i.e. perceptual distances) are the identical to those of the human observer and where curvatures are matched to pixel-domain curvatures. 
    
    Inputs:
    -------
    stim_folder: String
        String containing the location of the stimulus (i.e. movies)
    n_corr_obs: (n_frames x n_frames) Numpy array
        Number of correct responses for each frame combination
    n_total_obs: (n_frames x n_frames) Numpy array
        Numober of total responses for each frame combination
    n_dim: Scalar
        Number of dimensions
    n_starts: Scalar
        Number of multistarts. Default is 10.
    n_iter: Scalar
        Number of iterations. Default is 1000.
    n_frames: Scalar
        Number of frames. Default is 11.
    is_natural: Boolean
        Whether the stimulus is natural (True) or synthetic (False). Default is True.
    version: 1 or 2
        Which optimization version to use. Version 1 is based on minimizing the negative log likelihood (MLE), version 2 minimizes the MSE between the 
        human and estimated probabilities of correct responses. Default is 1.
    """

    im_category = 'natural' if is_natural else 'synthetic'
    prob_corr = np.divide(n_corr_obs, n_total_obs, out=np.zeros_like(n_corr_obs), where=n_total_obs!=0)

    # load videos 
    im = []
    for fname in sorted(os.listdir(stim_folder)):
        if not is_natural and (fname == f'natural01.png'): # first and last frames for synthetic videos are the same as for natural videos
            im.append(imageio.imread(Path(stim_folder) / fname))
        if im_category in fname:
            im_path = os.path.join(stim_folder, fname)
            im.append(imageio.imread(im_path))
    if not is_natural:
        im.append(imageio.imread(Path(stim_folder) / f'natural{n_frames:02d}.png')) # last frame

    # convert to 3D array and normalize to [0, 1]
    I = np.stack(im, axis=-1).astype(np.float64) / 255

    # compute pixel-domain curvature
    c_pixel = torch.tensor(compute_curvature_pixel(I)).unsqueeze(0)

    if version == 1:
        x, c_est, prob_est = optimize_MLE(n_dim, n_corr_obs, n_total_obs, n_iter, n_starts=n_starts, c_pixel=c_pixel, disp=disp)
    else:
        def func_binomial_prob(vec):
            # get perceptual locations
            d_ = torch.tensor(vec[:n_frames-1]).unsqueeze(0)
            a_ = torch.tensor(vec[n_frames-1:].reshape(n_dim, n_frames - 2)).unsqueeze(0)
            x, _, _, _, _ = compute_trajectory_perceptual(1, n_frames, n_dim, d_, c_pixel, a_)

            # get perceptual distances
            dist = torch.cdist(torch.transpose(x, 1, 2), torch.transpose(x, 1, 2))

            # compute hierarchical model
            normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0])) # cdf of the standard normal
            p_axb = normal.cdf(dist / np.sqrt(2)) * normal.cdf(dist / 2) + normal.cdf(-dist / np.sqrt(2)) * normal.cdf(-dist / 2)
            p = (1 - 2 * l) * p_axb.clone() + l
            # binomial = torch.distributions.Binomial(torch.tensor(n_corr_obs), probs=p)
            # prob_est = binomial.log_prob(torch.tensor(n_corr_obs)).exp()

            return (torch.tensor(prob_corr) - p.squeeze())**2 # this does not account for the frame-by-frame discriminability.. component-wise least-squares?
    
        n_frames = I.shape[-1]

        # initialization (c is taken from pixel domain)
        d_init = np.abs(np.random.normal(1, 0.5, size=((n_frames - 1))))
        a_init = np.random.normal(0, 2, size=((n_dim, n_frames - 2)))
        l = 0.06; # guess rate
        start_vec = np.concatenate((d_init, a_init.flatten()))

        LB = np.zeros(np.size(start_vec))
        UB = np.zeros(np.size(start_vec))

        # bounds for d
        LB[:n_frames-1] = -100
        UB[:n_frames-1] = 100

        # bounds for a
        LB[n_frames-1:] = -100
        UB[n_frames-1:] = 100

        # set bounds
        bnds = np.zeros((np.size(LB), 2))
        bnds[:, 0] = LB
        bnds[:, 1] = UB

        # optimize
        # res = minimize(func_binomial_prob, start_vec, method='L-BFGS-B', bounds=tuple(map(tuple, bnds)), options={'maxiter': n_iter, 'disp': False})
        res = minimize(func_binomial_prob, start_vec,  method='Powell', bounds=tuple(map(tuple, bnds)), options={'maxiter': n_iter, 'disp': disp})

        # reconstruct trajectory
        d = torch.tensor(res.x[:n_frames-1]).unsqueeze(0)
        a = torch.tensor(res.x[n_frames-1:].reshape(n_dim, n_frames - 2)).unsqueeze(0)
        x, _, c_est, _, _ = compute_trajectory_perceptual(1, n_frames, n_dim, d, c_pixel, a)

        # get perceptual distances
        dist = torch.cdist(torch.transpose(x, 1, 2), torch.transpose(x, 1, 2))

        # compute hierarchical model
        normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0])) # cdf of the standard normal
        p_axb = normal.cdf(dist / np.sqrt(2)) * normal.cdf(dist / 2) + normal.cdf(-dist / np.sqrt(2)) * normal.cdf(-dist / 2)
        p = (1 - 2 * l) * p_axb.clone() + l
        binomial = torch.distributions.Binomial(torch.tensor(n_corr_obs), probs=p)
        prob_est = binomial.log_prob(torch.tensor(n_corr_obs)).exp()

    return x, c_pixel, c_est, prob_corr, prob_est
