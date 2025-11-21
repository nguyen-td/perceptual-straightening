import numpy as np
import os
import torch

import imageio.v2 as imageio 
from pathlib import Path
from scipy.optimize import minimize

from modules import compute_trajectory_perceptual, compute_curvature_pixel, optimize_MLE, ELBO

def optimize_null(stim_folder, n_corr_obs, n_total_obs, n_dim, n_starts=10, n_iter=1000, n_frames=11, is_natural=True, version=1, disp=False, c_pixel=None):
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

    Outputs:
    --------
    x_null: (1 x n_dim x n_frames) Torch tensor
        Estimated perceptual locations 
    c_pixel: (n_frames - 2) Torch tensor
        Estimated local curvatures
    c_est: (n_frames - 2) Torch tensor
        Estimated local perceptual curvatures
    prop_corr: (n_frames x n_frames) Torch tensor
        Proportion correct as computed from human psychophysics data
    prop_est: (1 x n_frames x n_frames) Torch tensor
        Estimated proportion correct
    """

    im_category = 'natural' if is_natural else 'synthetic'

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

    prob_corr = np.divide(n_corr_obs, n_total_obs, out=np.zeros_like(n_corr_obs), where=n_total_obs!=0)

    if version == 1:
        x, c_est, prob_est = optimize_MLE(n_dim, n_corr_obs, n_total_obs, n_iter, n_starts=n_starts, c_pixel=c_pixel, disp=disp)
    elif version == 2:
        elbo = ELBO(n_dim, n_corr_obs, n_total_obs, n_starts=n_starts, n_iterations=n_iter, c_pixel=c_pixel)
        x, prob_est, _, _, _, _, _, _, _, c_est = elbo.optimize_ELBO_SGD()
        prob_est = prob_est.detach().numpy()
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

    return x, c_pixel, c_est, prob_corr, np.squeeze(prob_est)

def construct_null_trajectory(stim_folder, n_dim, d_perc, a_perc, is_natural=True, n_frames=11, c_pixel=None):
    """
    Construct a null trajectory by taking a most likely trajectory, estimated from human data, and 
    replace its curvatures with pixel-domain curvatures.

    Inputs:
    -------
    stim_folder: String
        String containing the location of the stimulus (i.e. movies)
    n_dim: Scalar
        Number of dimensions
    d_perc: (n_frames - 1) Torch tensor
        Most likely perceptual distance vector
    a_perc: (n_dim x n_frames) Torch tensor
        Most likely perceptual acceleration vectors
    n_frames: Scalar
        Number of frames. Default is 11.
    is_natural: Boolean
        Whether the stimulus is natural (True) or synthetic (False). Default is True.
    c_pixel: (n_samples x n_frames - 2) Torch tensor
        Pixel curvature. Default is None. If None, pixel curvature is being estimated from stim_folder.

    Output:
    -------
    x: (n_samples x n_dim x n_frames) Torch tensor
        Perceptual locations under the null model
    c_est: (n_frames - 2) Torch tensor
        Estimated local curvatures
    p: (n_frames x n_frames) Numpy array
        Estimated proportion correct
    """
    
    im_category = 'natural' if is_natural else 'synthetic'

    # load videos 
    if c_pixel is None:
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
        c_pixel = torch.tensor(compute_curvature_pixel(I))

    # construct null trajectory
    x, _, c_est, _, _ = compute_trajectory_perceptual(1, n_frames, n_dim, d_perc.unsqueeze(0), c_pixel.unsqueeze(0), a_perc.unsqueeze(0))

     # get perceptual distances
    dist = torch.cdist(torch.transpose(x, 1, 2), torch.transpose(x, 1, 2))

    # compute hierarchical model
    l = 0.06; # guess rate
    normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0])) # cdf of the standard normal
    p_axb = normal.cdf(dist / np.sqrt(2)) * normal.cdf(dist / 2) + normal.cdf(-dist / np.sqrt(2)) * normal.cdf(-dist / 2)
    p = (1 - 2 * l) * p_axb.clone() + l

    return x, c_est, p