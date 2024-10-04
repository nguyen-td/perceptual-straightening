import torch

def log_likelihood(N, trial_mat, pair_inds, x, l):
    """
    Computes the log likelihood of the binomial distribution given the number of correct and incorrect answers.

    Inputs:
    -------
    N: Scalar
        Number of nodes
    trial_mat: (n_trial x n_pairs) torch tensor
        Trial matrix of 1 (correct responses) and 0 (incorrect responses)
    pair_inds: (n_pairs x 2) torch tensor
        Matrix with index information for each pair
     x: (n_samples x N x (N - 1)) torch tensor
        Array corresponding to the inferred perceptual trajectory, where the third dimension corresponds to the number of dimensions.
    l: Scalar torch tensor
        Lapse rate
    """

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
    log_ll = torch.zeros(N - 1)
    for ij in range(N - 1): 
        distance = torch.linalg.norm(x[:, ij, :] - x[:, ij+1, :], dim=1)
        p_ij = (1 - 2 * l) * p_axb(distance) + l
        p_ij = torch.clamp(p_ij, min=1e-6, max=1.0 - 1e-6) # 0 or 1 would block gradient updates
        bool_ind_pair = (pair_inds == torch.tensor([ij+1, ij+2])).all(dim=1) # matlab starts from 0
        ind_pair = torch.where(bool_ind_pair)[0]

        binomial = torch.distributions.binomial.Binomial(n_trials, p_ij) 
        log_ll[ij] = torch.mean(binomial.log_prob(n_correct_mat[ind_pair])) # mean over samples

    return log_ll