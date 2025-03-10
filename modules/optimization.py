import torch
from torch import nn
import numpy as np
from scipy.spatial import distance
from scipy.stats import norm
from scipy.optimize import minimize
import torch.distributions as D
from pybads import BADS

from modules import compute_trajectory
from utils import make_positive_definite

def optimize_ML(n_dim, n_corr_obs, n_total_obs, lr=1e-4, n_iter=1000, verbose=True, n_starts=10):
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
    """

    n_frames = n_corr_obs.shape[0]

    def func_NLL(vec):
        # get perceptual locations
        c_ = torch.tensor(vec[:n_frames - 2]).unsqueeze(0)
        d_ = torch.tensor(vec[n_frames-2:n_frames-2 + n_frames-1]).unsqueeze(0)
        a_ = torch.tensor(start_vec[n_frames-2 + n_frames-1:].reshape(n_dim, n_frames - 2)).unsqueeze(0)
        x, _, _, _, _ = compute_trajectory(1, n_frames, n_dim, d_, c_, a_)

        # get perceptual distances
        dist = torch.cdist(torch.transpose(x, 1, 2), torch.transpose(x, 1, 2))

        # compute hierarchical model
        normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0])) # cdf of the standard normal
        p_axb = normal.cdf(dist / np.sqrt(2)) * normal.cdf(dist / 2) + normal.cdf(-dist / np.sqrt(2)) * normal.cdf(-dist / 2)
        p = (1 - 2 * l) * p_axb.clone() + l
        return -torch.sum((torch.tensor(n_corr_obs) * torch.log(p.clone())) + (torch.tensor(n_total_obs) - torch.tensor(n_corr_obs)) * torch.log(1 - p.clone())).detach().numpy()

    # run optimization using a multistart procedure
    old_loss = np.array([10e10])

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

        res = minimize(func_NLL, start_vec, bounds=tuple(map(tuple, bnds)), options={'maxiter': n_iter, 'disp': True})
        # inf_vec = np.zeros(np.size(start_vec)) + np.inf
        # bads = BADS(func_NLL, start_vec, -inf_vec, inf_vec, LB, UB, options={'max_iter': n_iter, 'display': 'iter'})
        # bads = BADS(func_NLL, start_vec, LB, UB, LB, UB, options={'max_iter': n_iter, 'display': 'iter'})
        # res = bads.optimize()
        # if res.success:
        if res['success']:
            new_loss = res.fun
            # new_loss = res['fval']
            if verbose:
                print(f'Current loss: {res.fun}')
                # print(f'Current loss: {res['fval']}')

            if new_loss < old_loss:
                # reconstruct trajectory
                c = torch.tensor(res.x[:n_frames - 2]).unsqueeze(0)
                d = torch.tensor(res.x[n_frames-2:n_frames-2 + n_frames-1]).unsqueeze(0)
                a = torch.tensor(res.x[n_frames-2 + n_frames-1:].reshape(n_dim, n_frames - 2)).unsqueeze(0)
                # c = torch.tensor(res['x'][:n_frames - 2]).unsqueeze(0)
                # d = torch.tensor(res['x'][n_frames-2:n_frames-2 + n_frames-1]).unsqueeze(0)
                # a = torch.tensor(res['x'][n_frames-2 + n_frames-1:].reshape(n_dim, n_frames - 2)).unsqueeze(0)
                x, _, c_est, _, _ = compute_trajectory(1, n_frames, n_dim, d, c, a)

                # get perceptual distances
                dist = torch.cdist(torch.transpose(x, 1, 2), torch.transpose(x, 1, 2))

                # compute hierarchical model
                normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0])) # cdf of the standard normal
                p_axb = normal.cdf(dist / np.sqrt(2)) * normal.cdf(dist / 2) + normal.cdf(-dist / np.sqrt(2)) * normal.cdf(-dist / 2)
                p = (1 - 2 * l) * p_axb.clone() + l

                old_loss = new_loss
                
                if verbose:
                    print('Loss updated')
        if verbose:
            print(f"Iteration {i+1} | Loss: {res.fun}")
            # print(f"Iteration {i+1} | Loss: {res['fval']}")

    return x, c_est, p, c, d, a

def optimize_ELBO(n_dim, n_corr_obs, n_total_obs, n_samples=100, lr=1e-4, eps=1e-6, n_iter=1000, verbose=True, n_starts=10):
    """
    Run optimizer to minimize ELBO to estimate perceptual trajectories from perceptual observations. Note that in all cases, sigma 
    denotes the variance/covariance matrix and NOT the standard deviation. Essentially, it is sigma**2.

    Inputs:
    -------
    n_dim: Scalar
        Number of dimensions
    n_corr_obs: (n_frames x n_frames) Numpy array
        Matrix where each entry corresponds to the number of correct observations/choices for the respective frame combination
    n_total_obs: (n_frames x n_frames) Numpy array
        Matrix where each entry corresponds to the number of completed trials for the respective frame combination
    n_samples: Scalar 
        Number of trajectories to sample
    lr: Scalar
        Learning rate
    eps: Scalar (default: 1e-6)
        Regularization factor to ensure numerical stability for computing the Cholesky decomposition
    n_iter: Scalar
        Number of iterations
    verbose: Boolean
        If True, outputs progress bar.
    n_starts: Scalar
        Number of multistarts for MLE

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
    """

    def prepare_ELBO(mu_prior_d, mu_prior_c, sigma_prior_d, sigma_prior_c, sigma_prior_a, mu_post_d, mu_post_c, mu_post_a, mu_post_l, sigma_post):

        # define means and covariances of the prior, extend the dimension of mu and sigma to match those of the posterior
        mu_prior = torch.cat((transform(mu_prior_d, 'd').repeat(n_frames - 1), 
                              mu_prior_c.repeat(n_frames - 2), 
                              mu_prior_a_init.repeat((n_frames - 2)), 
                              transform(mu_prior_l_init, 'l')), 0)
        sigma_prior = torch.block_diag(torch.diag(sigma_prior_d.repeat(n_frames - 1)), 
                                       torch.diag(sigma_prior_c.repeat(n_frames - 2)), 
                                       torch.diag(sigma_prior_a.repeat(n_frames - 2)), 
                                       torch.diag(sigma_prior_l_init))

        # define means and covariances of the posterior
        mu_post = torch.cat((transform(mu_post_d, 'd'), mu_post_c, mu_post_a.flatten(), transform(mu_post_l, 'l')))
        A_post, L_post = make_positive_definite(torch.diag(sigma_post), eps)
        posterior = D.MultivariateNormal(mu_post, scale_tril=L_post)
        # posterior = D.MultivariateNormal(mu_post, covariance_matrix=A_post)

        # define distributions
        A_prior, L_prior = make_positive_definite(sigma_prior, eps)
        prior = D.MultivariateNormal(mu_prior, scale_tril=L_prior)
        # prior = D.MultivariateNormal(mu_prior, scale_tril=A_prior)
       
        # use reparameterization trick (cf. Kingma and Welling, 2022) to sample from approximate distribution
        z_q = posterior.rsample(sample_shape=(n_samples, )) # shape: (n_samples x (n_frames - 1) + (n_frames - 2) + (n_dim * (n_frames - 2) + 1)) # TODO: something is wrong with z_q

        # define trajectory variables
        d_size = n_frames - 1
        c_size = n_frames - 2
        a_size = (n_dim) * (n_frames - 2)

        # extract variables
        d = transform(z_q[:, :d_size], 'd')
        c = z_q[:, d_size:d_size + c_size]
        a = z_q[:, d_size + c_size:d_size + c_size + a_size].reshape(-1, n_dim, n_frames - 2)
        l = transform(z_q[:, -1], 'l')

        # construct trajectory
        x, _, c_est, _, _ = compute_trajectory(n_samples, n_frames, n_dim, d, c, a)

        # get perceptual distances
        dist = torch.cdist(torch.transpose(x, 1, 2), torch.transpose(x, 1, 2))

        # compute hierarchical model
        normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0])) # cdf of the standard normal
        p_axb = normal.cdf(dist / np.sqrt(2)) * normal.cdf(dist / 2) + normal.cdf(-dist / np.sqrt(2)) * normal.cdf(-dist / 2)
        p = (1 - 2 * l[:, None, None]) * p_axb.clone() + l[:, None, None]

        p_eps = 1e-8  # small constant to prevent log(0)
        p = torch.clamp(p, p_eps, 1 - p_eps)
        log_ll = torch.mean(torch.sum((torch.tensor(n_corr_obs) * torch.log(p.clone())) + (torch.tensor(n_total_obs) - torch.tensor(n_corr_obs)) * torch.log(1 - p.clone()), dim=[1, 2]))

        # compute KL divergence
        kl = torch.sum(D.kl_divergence(posterior, prior))

        # compute ELBO
        elbo = -(log_ll - kl)

        return elbo, x, c_est, p, c, d, a


    def func_ELBO(vec):
        # unpack variables
        mu_prior_d    = torch.tensor(vec[0]).to(torch.float32)
        mu_prior_c    = torch.tensor(vec[1]).to(torch.float32)
        sigma_prior_d = torch.tensor(vec[2]).to(torch.float32)
        sigma_prior_c = torch.tensor(vec[3]).to(torch.float32)
        sigma_prior_a = torch.tensor(vec[4: 4 + len(sigma_prior_a_init)]).to(torch.float32)

        mu_post_d     = torch.tensor(vec[4 + len(sigma_prior_a_init) : 4 + len(sigma_prior_a_init) + len(mu_post_d_init)]).to(torch.float32)
        mu_post_c     = torch.tensor(vec[4 + len(sigma_prior_a_init) + len(mu_post_d_init) : 4 + len(sigma_prior_a_init) + len(mu_post_d_init) + len(mu_post_c_init)]).to(torch.float32)
        mu_post_a     = torch.tensor(vec[4 + len(sigma_prior_a_init) + len(mu_post_d_init) + len(mu_post_c_init) : 4 + len(sigma_prior_a_init) + len(mu_post_d_init) + len(mu_post_c_init) + len(mu_post_a_init.flatten())]).to(torch.float32)
        mu_post_l     = torch.tensor(vec[4 + len(sigma_prior_a_init) + len(mu_post_d_init) + len(mu_post_c_init) + len(mu_post_a_init.flatten()) : 4 + len(sigma_prior_a_init) + len(mu_post_d_init) + len(mu_post_c_init) + len(mu_post_a_init.flatten()) + 1]).to(torch.float32)
        sigma_post    = torch.tensor(vec[4 + len(sigma_prior_a_init) + len(mu_post_d_init) + len(mu_post_c_init) + len(mu_post_a_init.flatten()) + 1:]).to(torch.float32)

        elbo, _, _, _, _, _, _ = prepare_ELBO(mu_prior_d, mu_prior_c, sigma_prior_d, sigma_prior_c, sigma_prior_a, mu_post_d, mu_post_c, mu_post_a, mu_post_l, sigma_post)
        return elbo.detach().numpy()

    # run MLE to initialize posterior distribution
    print('Running MLE to initialize posterior..........................')
    x, _, p, c, d, a = optimize_ML(n_dim, n_corr_obs, n_total_obs, verbose=True, n_starts=n_starts)

    # create initial values
    n_frames = n_corr_obs.shape[0]

    mu_post_d_init = d.squeeze()
    mu_post_c_init = c.squeeze()
    mu_post_a_init = a.squeeze()
    mu_post_l_init = torch.tensor([0.0])
    mu_post_inits = torch.hstack((mu_post_d_init, mu_post_c_init, mu_post_a_init.flatten(), mu_post_l_init)).unsqueeze(1)
    sigma_post_init = torch.diag(mu_post_inits @ mu_post_inits.T) # try diagonal matrix, this is just a vector of diagonals!

    mu_prior_d_init = torch.tensor([1.0])
    mu_prior_c_init = torch.deg2rad(torch.tensor(60))
    mu_prior_a_init = torch.zeros(n_dim, requires_grad=False) 
    mu_prior_l_init = torch.tensor([0.0], requires_grad=False)

    d_size = n_frames - 1
    c_size = n_frames - 2
    a_size = n_dim * (n_frames - 2)
    sigma_prior_d_init = torch.var(mu_post_d_init, correction=False, keepdim=True) + torch.mean(sigma_post_init[:d_size])
    sigma_prior_c_init = torch.var(mu_post_c_init, correction=False, keepdim=True) + torch.mean(sigma_post_init[d_size:d_size + c_size])
    sigma_prior_a_init = torch.var(mu_post_a_init, dim=1, correction=False) + torch.mean(sigma_post_init[d_size + c_size:d_size + c_size + a_size])
    sigma_prior_l_init = torch.tensor([1.0], requires_grad=False)

    # create start vector with variables to optimize
    start_vec = torch.hstack((mu_prior_d_init, mu_prior_c_init, sigma_prior_d_init, sigma_prior_c_init, sigma_prior_a_init,
                                mu_post_d_init, mu_post_c_init, mu_post_a_init.flatten(), mu_post_l_init, sigma_post_init.flatten()))
    
    LB = torch.zeros(start_vec.size())
    UB = torch.zeros(start_vec.size())

    # bounds for mu_prior_d
    LB[0] = torch.tensor([0.0])
    UB[0] = torch.tensor([3.0])

    # bounds for mu_prior_c
    LB[1] = torch.tensor([0.0])
    UB[1] = torch.pi

    # bounds for sigma_prior_d
    LB[2] = torch.tensor([0.0])
    UB[2] = torch.tensor([5.0])

    # bounds for sigma_prior_c
    LB[3] = torch.tensor([0.0])
    UB[3] = torch.pi

    # bounds for sigma_prior_a
    LB[4 : 4 + len(sigma_prior_a_init)] = torch.tensor([0.0])
    UB[4 : 4 + len(sigma_prior_a_init)] = torch.tensor([5.0])

    # bounds for mu_post_d
    LB[4 + len(sigma_prior_a_init) : 4 + len(sigma_prior_a_init) + len(mu_post_d_init)] = torch.tensor([0.0])
    UB[4 + len(sigma_prior_a_init) : 4 + len(sigma_prior_a_init) + len(mu_post_d_init)] = torch.tensor([5.0])

    # bounds for mu_post_c
    LB[4 + len(sigma_prior_a_init) + len(mu_post_d_init) : 4 + len(sigma_prior_a_init) + len(mu_post_d_init) + len(mu_post_c_init)] = torch.tensor([0.0])
    UB[4 + len(sigma_prior_a_init) + len(mu_post_d_init) : 4 + len(sigma_prior_a_init) + len(mu_post_d_init) + len(mu_post_c_init)] = torch.pi

    # bounds for mu_post_a
    LB[4 + len(sigma_prior_a_init) + len(mu_post_d_init) + len(mu_post_c_init) : 4 + len(sigma_prior_a_init) + len(mu_post_d_init) + len(mu_post_c_init) + len(mu_post_a_init.flatten())] = torch.tensor([-5.0]) 
    UB[4 + len(sigma_prior_a_init) + len(mu_post_d_init) + len(mu_post_c_init) : 4 + len(sigma_prior_a_init) + len(mu_post_d_init) + len(mu_post_c_init) + len(mu_post_a_init.flatten())] = torch.tensor([-5.0])

    # bounds for mu_post_l
    LB[4 + len(sigma_prior_a_init) + len(mu_post_d_init) + len(mu_post_c_init) + len(mu_post_a_init.flatten()) : 4 + len(sigma_prior_a_init) + len(mu_post_d_init) + len(mu_post_c_init) + len(mu_post_a_init.flatten()) + 1] = torch.tensor([0.0]) 
    UB[4 + len(sigma_prior_a_init) + len(mu_post_d_init) + len(mu_post_c_init) + len(mu_post_a_init.flatten()) : 4 + len(sigma_prior_a_init) + len(mu_post_d_init) + len(mu_post_c_init) + len(mu_post_a_init.flatten()) + 1] = torch.tensor([5.0])

    # bounds for sigma_post
    LB_sigma_post = torch.zeros(len(sigma_post_init))
    UB_sigma_post = torch.zeros(len(sigma_post_init))

    LB_sigma_post[:len(mu_post_d_init)] = torch.tensor([0.0])
    UB_sigma_post[:len(mu_post_d_init)] = torch.tensor([5.0])

    LB_sigma_post[len(mu_post_d_init) : len(mu_post_d_init) + len(mu_post_c_init)] = torch.tensor([0.0]) 
    UB_sigma_post[len(mu_post_d_init) : len(mu_post_d_init) + len(mu_post_c_init)] = torch.pi

    LB_sigma_post[len(mu_post_d_init) + len(mu_post_c_init) : len(mu_post_d_init) + len(mu_post_c_init) + len(mu_post_a_init.flatten())] = torch.tensor([0.0]) 
    UB_sigma_post[len(mu_post_d_init) + len(mu_post_c_init) : len(mu_post_d_init) + len(mu_post_c_init) + len(mu_post_a_init.flatten())] = torch.tensor([5.0]) 

    LB_sigma_post[len(mu_post_d_init) + len(mu_post_c_init) + len(mu_post_a_init.flatten()):] = torch.tensor([0.0])  # lambda
    UB_sigma_post[len(mu_post_d_init) + len(mu_post_c_init) + len(mu_post_a_init.flatten()):] = torch.tensor([5.0])  # lambda

    LB[4 + len(sigma_prior_a_init) + len(mu_post_d_init) + len(mu_post_c_init) + len(mu_post_a_init.flatten()) + 1:] = LB_sigma_post
    UB[4 + len(sigma_prior_a_init) + len(mu_post_d_init) + len(mu_post_c_init) + len(mu_post_a_init.flatten()) + 1:] = UB_sigma_post

    # set bounds
    bnds = torch.zeros(len(LB), 2)
    bnds[:, 0] = LB
    bnds[:, 1] = UB

    print('')
    print('Start minimizing ELBO..........................')
    res = minimize(func_ELBO, start_vec, bounds=tuple(map(tuple, bnds)), options={'maxiter': n_iter, 'disp': verbose})

    # unpack variables
    mu_prior_d_    = torch.tensor(res.x[0]).to(torch.float32)
    mu_prior_c_    = torch.tensor(res.x[1]).to(torch.float32)
    sigma_prior_d_ = torch.tensor(res.x[2]).to(torch.float32)
    sigma_prior_c_ = torch.tensor(res.x[3]).to(torch.float32)
    sigma_prior_a_ = torch.tensor(res.x[4 : 4 + len(sigma_prior_a_init)]).to(torch.float32)
    mu_post_d_     = torch.tensor(res.x[4 + len(sigma_prior_a_init) : 4 + len(sigma_prior_a_init) + len(mu_post_d_init)]).to(torch.float32)
    mu_post_c_     = torch.tensor(res.x[4 + len(sigma_prior_a_init) + len(mu_post_d_init) : 4 + len(sigma_prior_a_init) + len(mu_post_d_init) + len(mu_post_c_init)]).to(torch.float32)
    mu_post_a_     = torch.tensor(res.x[4 + len(sigma_prior_a_init) + len(mu_post_d_init) + len(mu_post_c_init) : 4 + len(sigma_prior_a_init) + len(mu_post_d_init) + len(mu_post_c_init) + len(mu_post_a_init.flatten())]).to(torch.float32)
    mu_post_l_     = torch.tensor(res.x[4 + len(sigma_prior_a_init) + len(mu_post_d_init) + len(mu_post_c_init) + len(mu_post_a_init.flatten()) : 4 + len(sigma_prior_a_init) + len(mu_post_d_init) + len(mu_post_c_init) + len(mu_post_a_init.flatten()) + 1]).to(torch.float32)
    sigma_post_    = torch.tensor(res.x[4 + len(sigma_prior_a_init) + len(mu_post_d_init) + len(mu_post_c_init) + len(mu_post_a_init.flatten()) + 1:]).to(torch.float32)

    _, x, c_est, p, c, d, a = prepare_ELBO(mu_prior_d_, mu_prior_c_, sigma_prior_d_, sigma_prior_c_, sigma_prior_a_, mu_post_d_, mu_post_c_, mu_post_a_, mu_post_l_, sigma_post_)
    print('Done')

    return x, c_est, p, c, d, a, mu_prior_c_, mu_post_c_

def transform(x, var):
    """
    Feed variables through the respective transfer functions.

    Inputs:
    -------
    x: Torch tensor
        Variable to transform
    var: String
        Denotes which variable to transform, defined for "d" and "l". "c" is not transformed. For "a", the orthogonalization depends on the previous displacement vector and will therefore be computed during the trajectory construction.

    Output:
    -------
    x_trans: Torch tensor
        Transformed variable
    """

    def smooth_step_function(x, epsilon=1e-3):   
        """
        Differentiable approximation of the following piecewise function: 
        
        f(x) = {0 if x < 0
                pi if x >= pi
                x otherwise
                }
        """
        s1 = torch.sigmoid(x / epsilon)  # Approximates the step at x=0
        s2 = torch.sigmoid((torch.pi - x) / epsilon)  # Approximates the step at x=pi
        
        # Interpolating between the different regions
        result = s1 * x * s2 + torch.pi * (1 - s2)
        return result

    if var == 'd':
        f = nn.Softplus()
        y = f(x)
    elif var == 'l':
        f = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0])) 
        y = 0.06 * f.cdf(x)
    elif var == 'c':
        y = smooth_step_function(x)
    else:
        raise Exception("Unrecognized value for `var`.")

    return y