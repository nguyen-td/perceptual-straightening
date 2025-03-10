import torch

def project_to_positive_definite(matrix):
    """
    Project the matrix to the nearest positive-definite matrix.

    Input:
    ------
    matrix: (N x N) torch tensor
        Input matrix

    Output:
    -------
    matrix: (N x N) torch tensor
        Positive-definite putput matrix
    """
    eigvals, eigvecs = torch.linalg.eigh(matrix)
    # ensure eigenvalues are non-negative
    eigvals = torch.clamp(eigvals, min=1e-6)  
    return eigvecs @ torch.diag(eigvals) @ eigvecs.t()


def make_positive_definite(A, eps):
    """
    Perform the Cholesky decomposition to ensure the matrix becomes positive-definite.

    Inputs:
    -------
    A: (N x N) torch tensor
        Input matrix
    eps: Scalar
        Regularization factor to ensure numerical stability for computing the Cholesky decomposition
    
    Outputs:
    --------
    A_new: (N x N) torch tensor
        Positive-definite matrix
    L: (N x N) torch tensor
        Lower triangular matrix such that A = LL^H
    """

    A = A @ A.T.conj() + eps * torch.eye(A.shape[0]) 
    # A = A @ A.mT + eps 
    L, info = torch.linalg.cholesky_ex(A)
    if info != 0:
        raise ValueError("Cholesky decomposition failed, matrix may not be positive definite.")
    assert torch.all(torch.diag(L) > 0), "Cholesky factor has non-positive diagonal entries."

    return A, L

def make_positive_definite_batch(A, eps):
    """
    Perform the Cholesky decomposition to ensure the matrices in a batch (first dimension) become positive-definite.

    Inputs:
    -------
    A: (*, N x N) torch tensor
        Input matrices, where the first dimension corresponds to the batch size
    eps: Scalar
        Regularization factor to ensure numerical stability for computing the Cholesky decomposition

    Outputs:
    --------
    A_new: (*, N x N) torch tensor
        Positive-definite matrices, where the first dimension corresponds to the batch size
    L: (*, N x N) torch tensor
        Lower triangular matrices such that A = LL^H, where the first dimensions correspond to the batch size
    """

    A = torch.bmm(A, torch.transpose(A, 2, 1).conj()) * eps
    L, info = torch.linalg.cholesky_ex(A)
    if torch.sum(info) != 0:
        raise ValueError("Cholesky decomposition failed, matrix may not be positive definite.")
    for batch in range(L.shape[0]):
        assert torch.all(torch.diag(L[batch]) > 0), "Cholesky factor has non-positive diagonal entries."

    return A, L
