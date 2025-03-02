import numpy as np
import torch
from torch import nn
from pathlib import Path
from torchrl.modules.utils import inv_softplus
import torch.distributions as D
from scipy.spatial import distance
from scipy.stats import norm

from utils import make_positive_definite, make_positive_definite_batch, log_likelihood
from modules import compute_trajectory

class ELBO(nn.Module):
    """
    Class for optimizing the ELBO term. 
    
    The following only holds for a posterior that is *NOT* implemented within the mean-field approximation framework:

    The prior will contain global trajectory variables (one value for each variable: d*, c*, a*, l*), whereas the variational distribution will contain the local trajectory variables, i.e., a set of local variables for each node (or for N-1 or N-2 nodes, respectively: d(t), d(t+1)..., c(t), c(t+1), ..., a(t), a(t+1), ..., l). To match the dimension for computing the KL-divergence term, the prior will still have the same dimensions but with repeating values. For example, the variational distribution has N-1 values for the local 'd' parameter, the prior distribution has N-1 times the *same*  value for the 'd' parameter. 

    Input:
    ------
    n_frames: Scalar
        Number of nodes/frames
    n_dim: Scalar
        Number of dimensions
    mu_prior_d_init: Scalar torch tensor
        Initial value for the mean of the prior distribution around d in d'
    mu_prior_c_init: Scalar torch tensor
        Initial value for the mean of the posterior distribution around c in radians
    mu_prior_a_init: (N - 1) torch tensor
        Initial value for the mean of the posterior distribution around a in d'
    mu_prior_l_init: [1] torch tensor
        Initial value for the mean of the posterior distribution around lambda
    sigma_prior_d_init: [1] torch tensor
        Initial value for the variance of the prior distribution around d
    sigma_prior_c_init: [1] torch tensor
        Initial value for the variance of the prior distribution around c
    sigma_prior_a_init: (N - 1) torch tensor
        Initial value for the variance of the prior distribution around a
    sigma_prior_l_init: [1] torch tensor
        Initial value for the variance of the prior distribution around lambda
    mu_post_d_init: (N - 1) torch tensor 
        Initial value for the mean of the posterior distribution around d in d'
    mu_post_c_init: (N - 2) torch tensor
        Initial value for the mean of the posterior distribution around c in radians
    mu_post_a_init: (N - 2) x (N - 1) torch tensor
        Initial value for the mean of the posterior distribution around a in d'
    mu_post_l_init: [1] torch tensor
        Initial value for the mean of the posterior distribution around lambda
    sigma_post_init: (M x M) torch tensor, where M = (N - 1) + (N - 2) + (N - 2)*(N - 1) + 1
        Initial values for the covariance matrix of the posterior distribution
    eps: Scalar (default: 1e-6)
        Regularization factor to ensure numerical stability for computing the Cholesky decomposition
    """
    
    def __init__(self, 
                 n_frames, 
                 n_dim,
                 mu_prior_d_init, 
                 mu_prior_c_init, 
                 mu_prior_a_init, 
                 mu_prior_l_init, 
                 sigma_prior_d_init, 
                 sigma_prior_c_init, 
                 sigma_prior_a_init, 
                 sigma_prior_l_init, 
                 mu_post_d_init, 
                 mu_post_c_init, 
                 mu_post_a_init,
                 mu_post_l_init, 
                 sigma_post_init, 
                 eps=1e-6
                ):
        super(ELBO, self).__init__()

        self.n_frames = n_frames
        self.n_dim = n_dim
        self.eps = eps

        # initialize means of the prior
        # self.mu_prior_d = nn.Parameter(self._transform(mu_prior_d_init, 'd'))
        self.mu_prior_d = nn.Parameter(mu_prior_d_init)
        self.mu_prior_c = nn.Parameter(mu_prior_c_init)
        self.mu_prior_a = nn.Parameter(mu_prior_a_init, requires_grad=False)
        self.mu_prior_l = nn.Parameter(mu_prior_l_init, requires_grad=False)

        # initialize (diagonal) covariance matrices of the prior
        self.sigma_prior_d = nn.Parameter(sigma_prior_d_init)
        self.sigma_prior_c = nn.Parameter(sigma_prior_c_init)
        self.sigma_prior_a = nn.Parameter(sigma_prior_a_init)
        self.sigma_prior_l = nn.Parameter(sigma_prior_l_init, requires_grad=False)

        # initialize means of (N-1) independent posteriors
        self.mu_post_d = nn.Parameter(mu_post_d_init)
        self.mu_post_c = nn.Parameter(mu_post_c_init)
        self.mu_post_a = nn.Parameter(mu_post_a_init)
        self.mu_post_l = nn.Parameter(mu_post_l_init)

        # initialize (full) covariance matrix of posterior
        self.sigma_post = nn.Parameter(sigma_post_init) 

    def _make_prior_posterior(self):
        """
        Defines means and covariances for the prior and posterior according to the initialization scheme. 
        Then creates prior and posterior distributions.

        Outputs:
        --------
        prior: torch.distribution object
            Prior distribution p_{theta}(z)
        posterior: torch.distribution object
            Posterior distribution q_{phi}(z|x)
        
        """
        # define means and covariances of the prior, extend the dimension of mu and sigma to match those of the posterior
        mu_prior = torch.cat((self.mu_prior_d.repeat(self.n_frames - 1), 
                              self.mu_prior_c.repeat(self.n_frames - 2), 
                              self.mu_prior_a.repeat((self.n_frames - 2)), 
                              self.mu_prior_l), 0)
        sigma_prior = torch.block_diag(torch.diag(self.sigma_prior_d.repeat(self.n_frames - 1)), 
                                       torch.diag(self.sigma_prior_c.repeat(self.n_frames - 2)), 
                                       torch.diag(self.sigma_prior_a.repeat(self.n_frames - 2)), 
                                       torch.diag(self.sigma_prior_l))
        _, L_prior = make_positive_definite(sigma_prior, self.eps)
        prior = D.MultivariateNormal(mu_prior, scale_tril=L_prior)

        # define means and covariances of the posterior
        mu_post = torch.cat((self.mu_post_d, self.mu_post_c, self.mu_post_a.flatten(), self.mu_post_l))
        _, L_post = make_positive_definite(self.sigma_post, self.eps)
        posterior = D.MultivariateNormal(mu_post, scale_tril=L_post)
        
        return prior, posterior

    def kl_divergence(self):
        """
        Computes KL-divergence between posterior and prior.

        Output:
        -------
        kl: Scalar torch tensor
            KL-divergence term
        """

        prior, posterior = self._make_prior_posterior()
        kl = torch.sum(D.kl_divergence(posterior, prior))
        return kl

    def _transform(self, x, var):
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

    def compute_likelihood(self, n_corr_obs, n_total_obs, n_samples=100):
        """
        Compute the expected likelihood w.r.t. the posterior (first term of the objective function. Involves the computation of the trajectory.

        Inputs:
        -------
        n_corr_obs: (n_frames x n_frames) Numpy array
            Matrix where each entry corresponds to the number of correct observations/choices for the respective frame combination
        n_total_obs: (n_frames x n_frames) Numpy array
            Matrix where each entry corresponds to the number of completed trials for the respective frame combination
        n_samples: Scalar (default: 100)
            Number of trajectories to sample

        Outputs:
        --------
        log_ll: Scalar torch tensor 
            Contains the log likelihood over the entire dataset
        d: (n_samples x (n_frames - 1)) torch tensor
            Average transformed distance
        c: (n_samples x (n_frames - 2))
            Average curvature
        a: (n_samples x n_dim x n_frames)
            Average acceleration (direction of curvature) 
        p: (n_frames x n_frames) Torch tensor
            Average estimated proportion correct
        x: (n_dim x n_frames) Torch tensor
            Average perceptual locations 
        """

        _, posterior = self._make_prior_posterior()
        
        # use reparameterization trick (cf. Kingma and Welling, 2022) to sample from approximate distribution
        z_q = posterior.rsample(sample_shape=(n_samples, )) # shape: (n_samples x (N - 1) x (3 + N - 1))

        # define trajectory variables
        d_size = self.n_frames - 1
        c_size = self.n_frames - 2
        a_size = (self.n_dim) * (self.n_frames - 2)

        # extract variables
        d = z_q[:, :d_size]
        c = z_q[:, d_size:d_size + c_size]
        a = z_q[:, d_size + c_size:d_size + c_size + a_size].reshape(-1, self.n_dim, self.n_frames - 2)
        l = z_q[:, -1]

        # transform variables (note: a is not transformed here yet because it depends on previous displacement vector; 
        # will be transformed during trajectory generation)
        # d = self._transform(d, 'd')
        # l = self._transform(l, 'l')
        # c = self._transform(c, 'c')

        # construct trajectory
        x, _, _, _, _ = compute_trajectory(n_samples, self.n_frames, self.n_dim, d, c, a)

        # get perceptual distances
        dist = torch.cdist(torch.transpose(x, 1, 2), torch.transpose(x, 1, 2))

        # compute hierarchical model
        normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0])) # cdf of the standard normal
        p_axb = normal.cdf(dist / np.sqrt(2)) * normal.cdf(dist / 2) + normal.cdf(-dist / np.sqrt(2)) * normal.cdf(-dist / 2)
        p = (1 - 2 * l[:, None, None]) * p_axb.clone() + l[:, None, None]
        log_ll = torch.sum((torch.tensor(n_corr_obs) * torch.log(p.clone())) + (torch.tensor(n_total_obs) - torch.tensor(n_corr_obs)) * torch.log(1 - p.clone()), dim=[1, 2])

        return torch.mean(log_ll), torch.mean(d, dim=0), torch.mean(c, dim=0), torch.mean(a, dim=0), torch.mean(p, dim=0), torch.mean(x, dim=0)
    
    def compute_loss(self, log_ll, kl_divergence):
        """
        Returns the ELBO function. Minimizing the negative log likelihood is equivalent to maximizing the log likelihood.

        Inputs:
        -------
        log_ll: Scalar torch tensor 
            Contains the log likelihood over the entire dataset
        kl: Scalar torch tensor
            KL-divergence term

        Outputs:
        --------
        loss: Scalar torch tensor
            ELBO loss term
        """
        return -(log_ll - kl_divergence)
