import numpy as np
import scipy.linalg

import matrix.errors


def equal(A, B):
    return A.shape == B.shape and not np.any(A != B)


def is_finite(A):
    return np.all(np.isfinite(A))


def check_finite(A, check_finite=True):
    if check_finite and not is_finite(A):
        raise matrix.errors.MatrixNotFiniteError(matrix=A)


def solve_triangular(A, b, lower=True, unit_diagonal=False, overwrite_b=False, check_finite=True):
    return scipy.linalg.solve_triangular(A, b, lower=lower, unit_diagonal=unit_diagonal, overwrite_b=overwrite_b, check_finite=check_finite)
