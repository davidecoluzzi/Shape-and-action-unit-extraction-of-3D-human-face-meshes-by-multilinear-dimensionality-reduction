import numpy as np


def symmetric(A, p):
    """ Permute symmetrically a matrix.

    Parameters
    ----------
    A : numpy.ndarray
        The matrix that should be permuted.
        It must have the shape (m, m).
    p : numpy.ndarray
        The permutation vector.
        It must have the shape (m,).

    Returns
    -------
    numpy.ndarray
        The matrix `A` symmetrically permuted by the permutation vector `p`.
        For the returned matrix `B` holds for all i, j in range(m):
        B[i,j] == A[p[i],p[j]]
        It has the shape (m, m).
    """

    if p is not None:
        A = A[p[:, np.newaxis], p[np.newaxis, :]]
    return A
