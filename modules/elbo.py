import numpy as np
import torch
from torch import nn
import scipy
from pathlib import Path
from torchrl.modules.utils import inv_softplus

from utils import make_positive_definite

class ELBO(nn.Module):
    """
    Class for optimizing the ELBO term. Note that the prior will contain global trajectory variables (one value for each variable: d*, c*, a*, l*), whereas the variational distribution will contain the local trajectory variables, i.e., a set of local variables for each node (or for N-1 or N-2 nodes, respectively: d(t), d(t+1)..., c(t), c(t+1), ..., a(t), a(t+1), ..., l). To match the dimension for computing the KL-divergence term, the prior will still have the same dimensions but with repeating values. For example, the variational distribution has N-1 values for the local 'd' parameter, the prior distribution has N-1 times the *same*  value for the 'd' parameter.

    Input:
    ------
    N: Scalar
        Number of nodes
    data_path: String
        Path where the (simulated) data is stored. The following MATLAB structures are loaded:
            Data.mat containing the (n_trial x pairs) 'resp_mat' matrix of 1 (correct response) and 0 (incorrect response)
            ExpParam.mat containing the (n_pairs x 2) 'all_pairs' matrix with index information for each pair
    eps: Scalar
        Regularization factor to ensure numerical stability for computing the Cholesky decomposition
    glob_curv: Scalar
        Simulated average global curvature (in radians), used to initialize the mean of the respective prior distribution
    """
    
    def __init__(self, N, data_path, eps=1e-6, glob_curv=(np.pi / 2)):
        super(ELBO, self).__init__()

        self.N = N
        self.data_path = data_path
        self.eps = eps

        # initialize means of the prior
        self.mu_d = nn.Parameter(torch.tensor([0.2])) 
        self.mu_c = nn.Parameter(torch.tensor([glob_curv]))
        self.mu_a = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.mu_l = nn.Parameter(torch.tensor([0.0]), requires_grad=False)

        # initialize (diagonal) covariance matrices of the prior
        self.sigma_d = nn.Parameter(torch.tensor([1.0]))
        self.sigma_c = nn.Parameter(torch.tensor([1.0])) 
        self.sigma_a = nn.Parameter(torch.tensor([1.0]))
        self.sigma_l = nn.Parameter(torch.tensor([1.0]), requires_grad=False)

        # means of d_t and c_t (local variables) of the variational posterior are parametrized by d* and c* (global)
        M = (N - 1) + (N - 2) + (N - 2) * (N - 1) + 1 # sum over the dimension of each variable (d, c, a, l)
        self.mu_posterior_al = nn.Parameter(torch.randn(M - (N-1) - (N - 2)))

        # initialize matrix covariance matrix for the variational posterior 
        self.A = nn.Parameter(torch.abs(torch.randn(M))) # variance cannot be negative

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
        mu_prior = torch.cat((self.mu_d.repeat(self.N - 1), 
                              self.mu_c.repeat(self.N - 2), 
                              self.mu_a.repeat((self.N - 2) * (self.N - 1)), 
                              self.mu_l))
        sigma_prior = torch.block_diag(torch.diag(self.sigma_d.repeat(self.N - 1)), 
                                       torch.diag(self.sigma_c.repeat(self.N - 2)), 
                                       torch.diag(self.sigma_a.repeat((self.N - 2) * (self.N - 1))), 
                                       torch.diag(self.sigma_l))
        _, L_prior = make_positive_definite(sigma_prior, self.eps)
        prior = torch.distributions.MultivariateNormal(mu_prior, scale_tril=L_prior)

        # define means and covariances of the posterior
        _, L_posterior = make_positive_definite(torch.diag(self.A), self.eps)
        mu_posterior = torch.cat((inv_softplus(self.mu_d.repeat(self.N - 1)), 
                                    self.mu_c.repeat(self.N - 2), 
                                    self.mu_posterior_al))
        posterior = torch.distributions.MultivariateNormal(mu_posterior, scale_tril=L_posterior)
        
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
        kl = torch.distributions.kl.kl_divergence(posterior, prior)
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
    
        if var == 'd':
            f = nn.Softplus()
            y = f(x)
        elif var == 'l':
            f = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0])) 
            y = 0.06 * f.cdf(x)
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

    def compute_likelihood(self, n_samples=100):
        """
        Compute the expected likelihood w.r.t. the posterior (first term of the objective function. Involves the computation of the trajectory.

        Inputs:
        -------
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
        z_q = posterior.rsample(sample_shape=(n_samples, )) # shape: (n_samples x M)

        # define trajectory variables
        d_size = self.N - 1
        c_size = self.N - 2
        a_size = (self.N - 1) * (self.N - 2)

        d = z_q[:, :d_size]
        c = z_q[:, d_size:d_size + c_size]
        a = z_q[:, d_size + c_size:d_size + c_size + a_size].reshape(-1, self.N - 2, self.N - 1)
        l = z_q[:, -1]

        # transform variables (note: a is not transformed here yet because it depends on previous displacement vector; 
        # will be transformed during trajectory generation)
        d = self._transform(d, 'd')
        l = self._transform(l, 'l')

        # construct trajectory
        x = self.construct_trajectory(d, c, a)
        
        # load trial information
        # (n_trial x n_pairs) matrix of 1 (correct response) and 0 (incorrect response)
        trial_mat = torch.from_numpy(scipy.io.loadmat(Path(self.data_path) / 'Data.mat')['Data']['resp_mat'][0][0]) 
        # (n_pairs x 2) matrix with index information for each pair
        pair_inds = torch.from_numpy(scipy.io.loadmat(Path(self.data_path) / 'ExpParam.mat')['ExpParam']['all_pairs'][0][0]) 

        n_trials = trial_mat.shape[0]
        n_pairs = trial_mat.shape[1]
        n_correct_mat = torch.zeros(n_pairs)

        # create array containing the number of correct responses for each frame pair
        for i_trial in range(n_pairs):
            n_correct_mat[i_trial] = torch.sum(trial_mat[:, i_trial])

        # define necessary distributions and functions    
        normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0])) # cdf of the standard normal
        p_axb = lambda d: normal.cdf(d / torch.sqrt(torch.tensor([2.0]))) * normal.cdf(d / torch.tensor([2.0])) + normal.cdf(-d / torch.sqrt(torch.tensor([2.0]))) * normal.cdf(-d / torch.tensor([2.0]))

        # compute log likelihood
        log_ll = torch.zeros(self.N - 1)
        for ij in range(self.N - 1): 
            distance = torch.linalg.norm(x[:, ij, :] - x[:, ij+1, :], dim=1)
            p_ij = (1 - 2 * l) * p_axb(distance) + l
            p_ij = torch.clamp(p_ij, min=1e-6, max=1.0 - 1e-6) # 0 or 1 would block gradient updates
            bool_ind_pair = (pair_inds == torch.tensor([ij+1, ij+2])).all(dim=1) # matlab starts from 0
            ind_pair = torch.where(bool_ind_pair)[0]

            binomial = torch.distributions.binomial.Binomial(n_trials, p_ij) 
            log_ll[ij] = torch.mean(binomial.log_prob(n_correct_mat[ind_pair])) # mean over samples

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
        return -(log_ll + kl_divergence)
