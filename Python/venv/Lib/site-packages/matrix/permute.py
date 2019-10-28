import numpy as np

import matrix.dense.permute
import matrix.sparse.permute
import matrix.sparse.util

import matrix.constants


def permutation_vector(A, method=None):
    if method is not None:
        method = method.lower()
    supported_methods = matrix.constants.PERMUTATION_METHODS
    if method not in supported_methods:
        raise ValueError('Permutation method {} is unknown. Only the following methods are supported {}.'.format(method, supported_methods))

    if method in matrix.constants.NO_PERMUTATION_METHODS:
        return None
    else:
        d = A.diagonal()
        if isinstance(d, np.matrix):
            d = d.A1
        if method in (matrix.constants.INCREASING_ABSOLUTE_DIAGONAL_VALUES_PERMUTATION_METHOD, matrix.constants.DECREASING_ABSOLUTE_DIAGONAL_VALUES_PERMUTATION_METHOD):
            d = np.abs(d)
        if method in (matrix.constants.DECREASING_DIAGONAL_VALUES_PERMUTATION_METHOD, matrix.constants.DECREASING_ABSOLUTE_DIAGONAL_VALUES_PERMUTATION_METHOD):
            d = -d
        p = np.argsort(d)
        return p


def invert_permutation_vector(p):
    p_inverse = np.empty_like(p)
    for i in range(len(p)):
        p_inverse[p[i]] = i
    return p_inverse


def symmetric(A, p):
    """ Permute symmetrically a matrix.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix (with shape (m, m))
        The matrix that should be permuted.
    p : numpy.ndarray (with shape (m,))
        The permutation vector.

    Returns
    -------
    numpy.ndarray (with shape (m, m))
        The matrix `A` symmetrically permuted by the permutation vector `p`.
        For the returned matrix `B` holds for all i, j in range(m):
        B[i,j] == A[p[i],p[j]]
    """
    if p is not None:
        if matrix.sparse.util.is_sparse(A):
            return matrix.sparse.permute.symmetric(A, p)
        else:
            return matrix.dense.permute.symmetric(A, p)
    else:
        return A
