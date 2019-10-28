import warnings

import scipy.sparse

import matrix.permute


def _indices_of_compressed_matrix(A, p):
    if p is not None:
        # chose same dtype of indices and indptr
        try:
            p = p.astype(A.indices.dtype, casting='safe')
        except TypeError:
            A.indptr = A.indptr.astype(p.dtype, casting='safe')

        # apply permutation
        p_inverse = matrix.permute.invert_permutation_vector(p)
        A.indices = p_inverse[A.indices]
        A.has_sorted_indices = False
    return A


def rows(A, p, inplace=False, warn_if_wrong_format=True):
    if p is not None:
        if not scipy.sparse.isspmatrix_csc(A):
            if warn_if_wrong_format:
                warnings.warn('CSC matrix format is required. Converting to CSC matrix format.', scipy.sparse.SparseEfficiencyWarning)
            A = scipy.sparse.csc_matrix(A)
        elif not inplace:
            A = A.copy()
        A = _indices_of_compressed_matrix(A, p)
    return A


def colums(A, p, inplace=False, warn_if_wrong_format=True):
    if p is not None:
        if not scipy.sparse.isspmatrix_csr(A):
            if warn_if_wrong_format:
                warnings.warn('CSR matrix format is required. Converting to CSC matrix format.', scipy.sparse.SparseEfficiencyWarning)
            A = scipy.sparse.csr_matrix(A)
        elif not inplace:
            A = A.copy()
        A = _indices_of_compressed_matrix(A, p)
    return A


def symmetric(A, p, inplace=False, warn_if_wrong_format=True):
    """ Permute symmetrically a matrix.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        The matrix that should be permuted.
        It must have the shape (m, m).
    p : numpy.ndarray
        The permutation vector.
        It must have the shape (m,).
    inplace : bool
        Whether the permutation should be done inplace or not.
        optional, default : False
    warn_if_wrong_format : bool
        Whether the warn if the matrix `A` is not in the needed sparse format.
        optional, default : True

    Returns
    -------
    numpy.ndarray
        The matrix `A` symmetrically permuted by the permutation vector `p`.
        For the returned matrix `B` holds for all i, j in range(m):
        B[i,j] == A[p[i],p[j]]
        It has the shape (m, m).
    """

    if p is not None:
        if scipy.sparse.isspmatrix_csc(A):
            A = rows(A, p, inplace=inplace, warn_if_wrong_format=False)
            A = colums(A, p, inplace=True, warn_if_wrong_format=False)
        else:
            if warn_if_wrong_format and not scipy.sparse.isspmatrix_csr(A):
                warnings.warn('CSC or CSR matrix format is required. Converting to needed matrix format.', scipy.sparse.SparseEfficiencyWarning)
            A = colums(A, p, inplace=inplace, warn_if_wrong_format=False)
            A = rows(A, p, inplace=True, warn_if_wrong_format=False)
    return A
