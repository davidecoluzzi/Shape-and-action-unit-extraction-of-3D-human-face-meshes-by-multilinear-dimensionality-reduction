import scipy.sparse

import matrix.dense.util
import matrix.sparse.util

import matrix.errors


def equal(A, B):
    A_is_sparse = matrix.sparse.util.is_sparse(A)
    B_is_sparse = matrix.sparse.util.is_sparse(B)
    if A_is_sparse != B_is_sparse:
        return False
    if A_is_sparse:
        assert B_is_sparse
        return matrix.sparse.util.equal(A, B)
    else:
        assert not B_is_sparse
        return matrix.dense.util.equal(A, B)


def check_square_matrix(A):
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise matrix.errors.MatrixNotSquareError(matrix=A)


def is_finite(A):
    if scipy.sparse.issparse(A):
        return matrix.sparse.util.is_finite(A)
    else:
        return matrix.dense.util.is_finite(A)


def check_finite(A, check_finite=True):
    if check_finite and not is_finite(A):
        raise matrix.errors.MatrixNotFiniteError(matrix=A)


def solve_triangular(A, b, lower=True, unit_diagonal=False, overwrite_b=False, check_finite=True):
    if scipy.sparse.issparse(A):
        return matrix.sparse.util.solve_triangular(
            A, b,
            lower=lower, unit_diagonal=unit_diagonal,
            overwrite_b=overwrite_b, check_finite=check_finite)
    else:
        return matrix.dense.util.solve_triangular(
            A, b,
            lower=lower, unit_diagonal=unit_diagonal,
            overwrite_b=overwrite_b, check_finite=check_finite)
