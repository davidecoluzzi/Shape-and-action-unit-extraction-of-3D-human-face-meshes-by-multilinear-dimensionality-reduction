import warnings

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

import matrix.errors
import matrix.sparse.util


def is_sparse(A):
    return scipy.sparse.issparse(A)


def equal(A, B):
    return A.shape == B.shape and (A != B).nnz == 0


def is_finite(A):
    return np.all(np.isfinite(A.data))


def check_finite(A, check_finite=True):
    if check_finite and not is_finite(A):
        raise matrix.errors.MatrixNotFiniteError(matrix=A)


def convert_to_csc(A, warn_if_wrong_format=True, sort_indices=False, eliminate_zeros=False):
    if not scipy.sparse.isspmatrix_csc(A):
        if warn_if_wrong_format:
            warnings.warn('CSC matrix format is required. Converting to CSC matrix format.', scipy.sparse.SparseEfficiencyWarning)
        A = scipy.sparse.csc_matrix(A)
    A = matrix.sparse.util.sort_indices(A, sort_indices=sort_indices)
    A = matrix.sparse.util.eliminate_zeros(A, eliminate_zeros=eliminate_zeros)
    return A


def convert_to_csr(A, warn_if_wrong_format=True, sort_indices=False, eliminate_zeros=False):
    if not scipy.sparse.isspmatrix_csr(A):
        if warn_if_wrong_format:
            warnings.warn('CSR matrix format is required. Converting to CSR matrix format.', scipy.sparse.SparseEfficiencyWarning)
        A = scipy.sparse.csr_matrix(A)
    A = matrix.sparse.util.sort_indices(A, sort_indices=sort_indices)
    A = matrix.sparse.util.eliminate_zeros(A, eliminate_zeros=eliminate_zeros)
    return A


def sort_indices(A, sort_indices=True):
    if sort_indices:
        A.sort_indices()
    return A


def eliminate_zeros(A, eliminate_zeros=True):
    if eliminate_zeros:
        A.eliminate_zeros()
    return A


def convert_index_dtype(A, dtype):
    if not (scipy.sparse.isspmatrix_csc(A) or scipy.sparse.isspmatrix_csr(A)):
        raise NotImplementedError("Only CSR and CSC are supported yet.")
    A.indices = np.asanyarray(A.indices, dtype=dtype)
    A.indptr = np.asanyarray(A.indptr, dtype=dtype)
    return A


def set_diagonal(A, diagonal_value):
    assert A.ndim == 2
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', scipy.sparse.SparseEfficiencyWarning)
        for i in range(min(A.shape)):
            A[i, i] = diagonal_value


def solve_triangular(A, b, lower=True, unit_diagonal=False, overwrite_b=False, check_finite=True):
    if check_finite:
        check_finite(A)
        check_finite(b)
    A = A.tocsr(copy=True)
    if unit_diagonal:
        matrix.sparse.util.set_diagonal(A, 1)
    b = b.astype(np.result_type(A.data, b, np.float), copy=not overwrite_b)  # this has to be done due to a bug in scipy (see pull reqeust #7449)
    return scipy.sparse.linalg.spsolve_triangular(A, b, lower=lower, overwrite_A=True, overwrite_b=overwrite_b)
