import numpy as np
import torch
from torch import nn
from pathlib import Path
import torch.distributions as D
from scipy.spatial import distance
from scipy.stats import norm

from utils import make_positive_definite
from modules import optimize_MLE, compute_hierarchical_ll

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
    n_corr_obs: (n_frames x n_frames) Numpy array
        Matrix where each entry corresponds to the number of correct observations/choices for the respective frame combination
    n_total_obs: (n_frames x n_frames) Numpy array
        Matrix where each entry corresponds to the number of completed trials for the respective frame combination
    lr: Scalar
        Learning rate for optimization algorithm
    n_iterations: Scalar
        Number of iterations for optimization
    n_starts: Scalar
        Number of multistarts of the maximum likelihood estimation for initializing the posterior distributions
    n_samples: Scalar
        Number of trajectories to sample to compute the expected value (in ELBO)
    eps: Scalar (default: 1e-6)
        Regularization factor to ensure numerical stability for computing the Cholesky decomposition
    verbose: Boolean
        If True, outputs progress bar.
    """
    
    def __init__(self,  
                 n_dim,
                 n_corr_obs,
                 n_total_obs,
                 lr=1e-4,
                 n_iterations=20000,
                 n_starts=10,
                 n_samples=100,
                 eps=1e-6,
                 verbose=True
                ):
        super(ELBO, self).__init__()

        self.n_dim = n_dim
        self.n_corr_obs = n_corr_obs
        self.n_total_obs = n_total_obs
        self.lr = lr
        self.n_iterations = n_iterations
        self.n_samples = n_samples
        self.n_starts = n_starts
        self.eps = eps
        self.verbose = verbose

    def optimize_ELBO_SGD(self):
        """
        Runs the complete algorithm to minimize the ELBO using SGD.

        Outputs:
        --------
        x: (n_samples x n_dim x n_frames) Torch tensor
            Most likely perceptual locations (determined by optimized posterior means)
        p: (n_frames x n_frames) Torch tensor
            Most likely estimation of proportion correct (determined by optimized posterior means)
        errors: (n_iterations, ) Torch tensor
            ELBO loss over iterations
        kl_loss: (n_iterations, ) Torch tensor
            KL-divergence over iterations
        c_prior: (n_iterations, ) Torch tensor
            Updates of mu_prior_c over iterations
        d_prior: (n_iterations, ) Torch tensor
            Updates of mu_prior_d over iterations
        l_prior: (n_iterations, ) Torch tensor
            Updates of mu_prior_l over iterations
        c_post: (n_frames - 2, n_iterations, ) Torch tensor
            Updates of mu_post_c over iterations
        d_post: (n_frames - 1, n_iterations, ) Torch tensor
            Updates of mu_post_d over iterations
        l_post: (n_iterations, ) Torch tensor
            Updates of mu_post_l over iterations
        c_est: (n_samples x n_frames - 2) Torch tensor
            Estimated curvature vector from generated trajectory; can be used to check if the estimated curvature is the same as the ground truth curvature used to generate the trajectory
        """

        # run MLE to initialize posterior distribution
        if self.verbose:
            print('Running MLE to initialize posterior..........................')
        _, _, _, c, d, a, inv_hess = optimize_MLE(self.n_dim, self.n_corr_obs, self.n_total_obs, verbose=self.verbose, n_starts=self.n_starts)

        # create initial values
        self.n_frames = self.n_corr_obs.shape[0]

        self.mu_post_d = nn.Parameter(self._transform(d.squeeze(), 'd'))
        self.mu_post_c = nn.Parameter(c.squeeze())
        self.mu_post_a = nn.Parameter(a.squeeze())
        self.mu_post_l = nn.Parameter(self._transform(torch.tensor([0.0]), 'l'))

        self.mu_post_inits = torch.hstack((self.mu_post_d, self.mu_post_c, self.mu_post_a.flatten(), self.mu_post_l))
        self.sigma_post = nn.Parameter(torch.eye(len(self.mu_post_inits)))

        self.mu_prior_d = nn.Parameter(self._transform(torch.tensor([1.0]), 'd'))
        self.mu_prior_c = nn.Parameter(torch.deg2rad(torch.tensor(60)))
        self.mu_prior_a = nn.Parameter(torch.zeros(self.n_dim), requires_grad=False) 
        self.mu_prior_l = nn.Parameter(self._transform(torch.tensor([0.0]), 'l'), requires_grad=False)

        d_size = self.n_frames - 1
        c_size = self.n_frames - 2
        a_size = self.n_dim * (self.n_frames - 2)

        # A = torch.hstack((self.mu_post_d, self.mu_post_c, self.mu_post_a.flatten(), self.mu_post_l))
        # sigma_post = torch.zeros(len(A), len(A))
        # sigma_post[:len(A)-1, :len(A)-1] = torch.tensor(inv_hess)
        # sigma_post[-1, -1] = 1.0 # variance for l
        # self.sigma_post = nn.Parameter(sigma_post)

        self.sigma_prior_d = nn.Parameter(torch.var(self.mu_post_d, correction=True, keepdim=True) + torch.mean(self.sigma_post[:d_size]))
        self.sigma_prior_c = nn.Parameter(torch.var(self.mu_post_c, correction=True, keepdim=True) + torch.mean(self.sigma_post[d_size:d_size + c_size]))
        self.sigma_prior_a = nn.Parameter(torch.var(self.mu_post_a, dim=1, correction=True) + torch.mean(self.sigma_post[d_size + c_size:d_size + c_size + a_size]))
        # self.sigma_prior_d = nn.Parameter(torch.tensor([0.001]))
        # self.sigma_prior_c = nn.Parameter(torch.tensor([0.5]))
        # self.sigma_prior_a = nn.Parameter(torch.tensor([5.0]).repeat(self.n_dim))
        self.sigma_prior_l = nn.Parameter(torch.tensor([1.0]), requires_grad=False)

        # initialize optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # initialize errors for storage
        errors = torch.zeros(self.n_iterations)
        kl_loss = torch.zeros(self.n_iterations)
        ll_loss = torch.zeros(self.n_iterations)

        # track free parameters
        c_prior = torch.zeros(self.n_iterations)
        d_prior = torch.zeros(self.n_iterations)
        l_prior = torch.zeros(self.n_iterations)

        c_post = torch.zeros(self.n_frames - 2, self.n_iterations)
        d_post = torch.zeros(self.n_frames - 1, self.n_iterations)
        l_post = torch.zeros(self.n_iterations)

        # run optimization
        for i in range(self.n_iterations):

            # clear gradients
            optimizer.zero_grad()

            # compute ELBO
            log_ll = self.compute_likelihood(self.n_corr_obs, self.n_total_obs, n_samples=self.n_samples)
            kl = self.kl_divergence()
            loss = self.compute_loss(log_ll, kl)

            # gradient update
            loss.backward()
            optimizer.step()

            # store errors
            errors[i] = loss.item()
            kl_loss[i] = kl.item()
            ll_loss[i] = log_ll.item()

            c_prior[i] = torch.rad2deg(self._transform(self.mu_prior_c, 'c').detach())
            d_prior[i] = self._transform(self.mu_prior_d, 'd').detach()
            l_prior[i] = self._transform(self.mu_prior_l, 'l').detach()

            c_post[:, i] = torch.rad2deg(self._transform(self.mu_post_c, 'c').detach())
            d_post[:, i] = self._transform(self.mu_post_d, 'd').detach()
            l_post[i] = self._transform(self.mu_post_l, 'l').detach()

            # print progress
            if self.verbose:
                if not i % 250:
                    print(f"Epoch: {i}, Loss: {loss.item()}")

            # early stopping based on convergence of prior
            if (i > 0) and (i % 5 == 0):
                if (np.abs(c_prior[i] - c_prior[i-3]) < 1e-5) and (np.abs(kl_loss[i] - kl_loss[i-3]) < 1e-4):
                    errors = errors[:i]
                    kl_loss = kl_loss[:i]
                    ll_loss = ll_loss[:i]

                    c_prior = c_prior[:i]
                    d_prior = d_prior[:i]
                    l_prior = l_prior[:i]

                    c_post = c_post[:, :i]
                    d_post = d_post[:, :i]
                    l_post = l_post[:i]
                    break

        x, p, _, c_est = compute_hierarchical_ll(1, self.n_frames, self.n_dim, self.n_corr_obs, self.n_total_obs, self._transform(self.mu_post_d, 'd').unsqueeze(0), self.mu_post_c.unsqueeze(0), self.mu_post_a.unsqueeze(0), self._transform(self.mu_post_l, 'l'))
        return x, p, errors, kl_loss, ll_loss, c_prior, d_prior, l_prior, c_post, d_post, l_post, c_est

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
        mu_prior = torch.cat((self._transform(self.mu_prior_d, 'd').repeat(self.n_frames - 1), 
                              self._transform(self.mu_prior_c, 'c').repeat(self.n_frames - 2), 
                              self.mu_prior_a.repeat((self.n_frames - 2)), 
                              self._transform(self.mu_prior_l, 'l')), 0)
        sigma_prior = torch.block_diag(torch.diag(self.sigma_prior_d.repeat(self.n_frames - 1)), 
                                       torch.diag(self.sigma_prior_c.repeat(self.n_frames - 2)), 
                                       torch.diag(self.sigma_prior_a.repeat(self.n_frames - 2)), 
                                       torch.diag(self.sigma_prior_l))
        _, L_prior = make_positive_definite(sigma_prior, self.eps)
        prior = D.MultivariateNormal(mu_prior, scale_tril=L_prior)

        # define means and covariances of the posterior
        mu_post = torch.cat((self._transform(self.mu_post_d, 'd'), self._transform(self.mu_post_c, 'c'), self.mu_post_a.flatten(), self._transform(self.mu_post_l, 'l')))
        _, L_post = make_positive_definite(self.sigma_post, self.eps)
        posterior = D.MultivariateNormal(mu_post.to(torch.float32), scale_tril=L_post)
        
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
            # return x
        
        # def inv_softplus(x):
        #     return x + torch.log(-torch.expm1(-x))

        if var == 'd':
            f = nn.Softplus(beta=1000) # closely approximate ReLu
            # f = nn.ReLU()
            y = f(x)
        # elif var == 'd_inv':
        #     y = inv_softplus(x)
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
        x: (n_dim x n_frames) Torch tensor
            Average perceptual locations 
        p: (n_frames x n_frames) Torch tensor
            Average estimated proportion correct
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
        d = self._transform(d, 'd')
        l = self._transform(l, 'l')
        c = self._transform(c, 'c')

        _, _, log_ll, _ = compute_hierarchical_ll(n_samples, self.n_frames, self.n_dim, n_corr_obs, n_total_obs, d, c, a, l)

        return torch.mean(log_ll)
    
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
