import numpy as np
import torch
from torch import nn
import scipy
from pathlib import Path
from torchrl.modules.utils import inv_softplus
import torch.distributions as D

from utils import make_positive_definite, make_positive_definite_batch, log_likelihood

class ELBO(nn.Module):
    """
    Class for optimizing the ELBO term. Note that the prior will contain global trajectory variables (one value for each variable: d*, c*, a*, l*), whereas the variational distribution will contain the local trajectory variables, i.e., a set of local variables for each node (or for N-1 or N-2 nodes, respectively: d(t), d(t+1)..., c(t), c(t+1), ..., a(t), a(t+1), ..., l). To match the dimension for computing the KL-divergence term, the prior will still have the same dimensions but with repeating values. For example, the variational distribution has N-1 values for the local 'd' parameter, the prior distribution has N-1 times the *same*  value for the 'd' parameter.

    Input:
    ------
    N: Scalar
        Number of nodes
    d_post_init: (N-1) torch tensor
        Initial values for the mean of the posterior distributions around d
    c_post_init: (N-1) torch tensor
        Initial values for the means of the posterior distributions around c
    a_post_init: (N-1, N-1) torch tensor
        Initial values for the means of the posterior distributions around a
    eps: Scalar (default: 1e-6)
        Regularization factor to ensure numerical stability for computing the Cholesky decomposition
    """
    
    def __init__(self, N, d_post_init, c_post_init, a_post_init, eps=1e-6):
        super(ELBO, self).__init__()

        self.N = N
        self.eps = eps

        # initialize means of the prior
        self.mu_prior_d = nn.Parameter(torch.rand(1))
        # self.mu_prior_d = nn.Parameter(torch.mean(d_post_init, dim=0, keepdim=True)) 
        # self.mu_prior_d = nn.Parameter(torch.tensor([0.2]))
        self.mu_prior_c = nn.Parameter(torch.rand(1) * torch.pi/2)
        # self.mu_prior_c = nn.Parameter(torch.mean(c_post_init[:, 1:], dim=1))
        # self.mu_prior_c = nn.Parameter(torch.pi / torch.tensor([4.0]))
        # self.mu_prior_a = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.mu_prior_a = nn.Parameter(torch.tensor([0.0]))
        # self.mu_prior_a = nn.Parameter(torch.randn(1))
        # self.mu_prior_l = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.mu_prior_l = nn.Parameter(torch.tensor([0.0]))
        # self.mu_prior_l = nn.Parameter(torch.rand(1))

        # initialize (diagonal) covariance matrices of the prior
        self.sigma_prior_d = nn.Parameter(torch.tensor([2.0]))
        # self.sigma_prior_d = nn.Parameter(torch.rand(1))
        # self.sigma_prior_c = nn.Parameter(torch.randn(1)) 
        self.sigma_prior_c = nn.Parameter(torch.rand(1) * torch.pi)
        self.sigma_prior_a = nn.Parameter(torch.ones(N-1))
        # self.sigma_prior_a = nn.Parameter(torch.tensor([1.0]))
        # self.sigma_prior_l = nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        self.sigma_prior_l = nn.Parameter(torch.tensor([1.0]))

        # initialize means of (N-1) independent posteriors
        # self.mu_post_d = nn.Parameter(torch.randn(N-1))
        self.mu_post_d = nn.Parameter(d_post_init)
        # self.mu_post_c = nn.Parameter(torch.rand(N-1) * torch.pi)
        self.mu_post_c = nn.Parameter(c_post_init)
        # self.mu_post_a = nn.Parameter(torch.zeros(N-1, N-1), requires_grad=False)
        # self.mu_post_a = nn.Parameter(torch.zeros((N-2) * (N-1)), requires_grad=False)
        self.mu_post_a = nn.Parameter(a_post_init)
        self.mu_post_l = nn.Parameter(torch.zeros(N-1), requires_grad=False)
        # self.mu_post_l = nn.Parameter(torch.tensor([0.0]), requires_grad=False)

        # # initialize sigmas of (N-1) independent posteriors
        # self.sigma_post_d = nn.Parameter(torch.randn(N-1))
        # # self.sigma_post_d = nn.Parameter(torch.mean(d_post_init).repeat(N-1))
        # self.sigma_post_c = nn.Parameter(torch.randn(N-1))
        # # self.sigma_post_c = nn.Parameter(torch.tensor([5.0]).repeat(N-1))
        # self.sigma_post_a = nn.Parameter(torch.randn(N-1, N-1))
        # self.sigma_post_l = nn.Parameter(torch.ones(N-1), requires_grad=False)

        # initialize (full) covariance matrix of posterior
        # M = (N - 1) + (N - 2) + (N - 2) * (N - 1) + 1 # sum over the dimension of each variable (d, c, a, l)
        # self.sigma_post = nn.Parameter(torch.rand(M, M)) 
        self.sigma_post = torch.randn(self.N-1, 3 + (N-1), 3 + (N-1)) # 2nd and 3rd dimension: number of variables

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
        mu_prior = torch.cat((self.mu_prior_d, 
                              self.mu_prior_c, 
                              self.mu_prior_a.repeat(self.N-1), 
                              self.mu_prior_l), 0)
        _, L_prior = make_positive_definite(torch.diag(torch.cat((self.sigma_prior_d, 
                                                                  self.sigma_prior_c, 
                                                                  self.sigma_prior_a, 
                                                                  self.sigma_prior_l), 0)), self.eps)
        # mu_prior = torch.cat((self.mu_prior_d.repeat(self.N - 1), 
        #                       self.mu_prior_c.repeat(self.N - 2), 
        #                       self.mu_prior_a.repeat((self.N - 2) * (self.N - 1)), 
        #                       self.mu_prior_l))
        # sigma_prior = torch.block_diag(torch.diag(self.sigma_prior_d.repeat(self.N - 1)), 
        #                                torch.diag(self.sigma_prior_c.repeat(self.N - 2)), 
        #                                torch.diag(self.sigma_prior_a.repeat((self.N - 2) * (self.N - 1))), 
        #                                torch.diag(self.sigma_prior_l))
        # _, L_prior = make_positive_definite(sigma_prior, self.eps)
        prior = D.MultivariateNormal(mu_prior, scale_tril=L_prior)

        # define means and covariances of the posterior
        mu_post = torch.vstack((self.mu_post_d, self.mu_post_c, self.mu_post_a, self.mu_post_l)).T
        # mu_post = torch.cat((self.mu_post_d, self.mu_post_c, self.mu_post_a, self.mu_post_l))
        # sigma_post = torch.diag_embed(torch.vstack((self.sigma_post_d, self.sigma_post_c, self.sigma_post_a, self.sigma_post_l)).T)
        # _, L_post = make_positive_definite_batch(sigma_post, self.eps)
        _, L_post = make_positive_definite_batch(self.sigma_post, self.eps)
        # _, L_post = make_positive_definite(self.sigma_post, self.eps)
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
        # kl = torch.distributions.kl.kl_divergence(posterior, prior)
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

    def construct_trajectory(self, d, c, a, tol=1e-6):
        """
        Computation of the trajectory. M refers to the sum over all dimension of the variables (d, c, a, l)

        Inputs:
        -------
        d: (n_samples x (N - 1)) torch tensor
            Transformed distance
        c: (n_samples x (N - 2))
            Curvature
        a: (n_samples x (N - 2) x (N - 1))
            Acceleration (direction of curvature)
        tol: Scalar, default: 1e-6
            Tolerance for checking the orthogonalization. Analytically, the dot product between two vectors should be 0 if they are orthogonal. We can add a small tolerance to account for numerical errors.

        Output:
        -------
        x: (n_samples x N x (N - 1)) torch tensor
            Array corresponding to the inferred perceptual trajectory, where the third dimension corresponds to the number of dimensions. 
        """
        
        assert d.shape[0] == c.shape[0] == a.shape[0], (
            "All inputs should have the same first dimension size (n_samples)."
        )
        n_samples = d.shape[0]

        # initialize v_hat and x
        v_hat = torch.zeros(n_samples, self.N, self.N - 1) # last column is the number of dimensions D = N - 1
        v_hat[:, 0, :] = torch.nan # v_hat is only defined from 1, ..., N, but keep a 0-th dimension to avoid conflicts
        v_hat[:, 1, 0] = 1 # v1 lies in the direction of the first axis
        x = torch.zeros(v_hat.shape)

        # compute locations for remaining nodes (v0 and v1 were already initialized)
        for t in range(2, self.N):
            # orthogonalize a(t) w.r.t. v(t-1)
            a_hat_t = torch.zeros(n_samples, a.shape[2]) 
            for i_sample in range(n_samples):
                Q, _ = torch.linalg.qr(torch.stack([v_hat[i_sample, t-1, :], a[i_sample, t-2, :]], dim=1))
                a_hat_t[i_sample, :] = Q[:, 1]
                assert (v_hat[i_sample, t-1, :] @ a_hat_t[i_sample, :]).item() <= tol, ("Failed to orthogonalize a_t")

            # compute displacement vectors
            v_hat[:, t, :] = torch.cos(c[:, t-2].unsqueeze(-1)) * v_hat[:, t-1, :].clone() + torch.sin(c[:, t-2].unsqueeze(-1)) * a_hat_t.clone()

        for t in range(1, self.N): 
            # compute perceptual trajectories
            x[:, t, :] = x[:, t-1, :] + d[:, t-1].unsqueeze(-1) * v_hat[:, t, :]
        
        return x

    def compute_likelihood(self, trial_mat, pair_inds, n_samples=100):
        """
        Compute the expected likelihood w.r.t. the posterior (first term of the objective function. Involves the computation of the trajectory.

        Inputs:
        -------
        trial_mat: (n_trial x n_pairs) torch tensor
            Trial matrix of 1 (correct responses) and 0 (incorrect responses)
        pair_inds: (n_pairs x 2) torch tensor
            Matrix with index information for each pair
        n_samples: Scalar (default: 100)
            Number of trajectories to sample

        Outputs:
        --------
        log_ll: Scalar torch tensor 
            Contains the log likelihood over the entire dataset
        d: (n_samples x (N - 1)) torch tensor
            Transformed distance
        c: (n_samples x (N - 2))
            Curvature
        a: (n_samples x (N - 2) x (N - 1))
            Acceleration (direction of curvature)
        """

        _, posterior = self._make_prior_posterior()
        
        # use reparameterization trick (cf. Kingma and Welling, 2022) to sample from approximate distribution
        z_q = posterior.rsample(sample_shape=(n_samples, )) # shape: (n_samples x (N-1) x (3 + N - 1))

        # define trajectory variables
        d_size = self.N - 1
        c_size = self.N - 2
        a_size = (self.N - 1) * (self.N - 2)

        # extract variables
        d = z_q[:, :, 0]     # t = 1, ..., T
        c = z_q[:, 1:, 1]    # t = 2, ..., T
        a = z_q[:, 1:, 2:-1] # t = 2, ..., T
        l = torch.mean(z_q[:, :, -1])
        # d = z_q[:, :d_size]
        # c = z_q[:, d_size:d_size + c_size]
        # a = z_q[:, d_size + c_size:d_size + c_size + a_size].reshape(-1, self.N - 2, self.N - 1)
        # l = z_q[:, -1]

        # transform variables (note: a is not transformed here yet because it depends on previous displacement vector; 
        # will be transformed during trajectory generation)
        d = self._transform(d, 'd')
        l = self._transform(l, 'l')
        c = self._transform(c, 'c')

        # construct trajectory
        x = self.construct_trajectory(d, c, a)
        log_ll = log_likelihood(self.N, trial_mat, pair_inds, x, l)

        return torch.sum(log_ll), d, c, a
    
    def compute_loss(self, log_ll, kl_divergence):
        """
        Returns the ELBO function. Returns the negative function because the optimizers minimize.

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
