import warnings

import numpy as np
import scipy.sparse

import matrix.dense.calculate

import matrix.sparse.calculate
import matrix.sparse.util

import matrix.constants
import matrix.errors
import matrix.permute
import matrix.util


def decompose(A, permutation_method=None, check_finite=True, return_type=None):
    """
    Computes a decomposition of a matrix.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix
        Matrix to be decomposed.
        It is assumed, that A is Hermitian.
        The matrix must be a squared matrix.
    permutation_method : str
        The symmetric permutation method that is applied to the matrix before
        it is decomposed. It has to be a value in
        :const:`matrix.PERMUTATION_METHODS`.
        If `A` is sparse, it can also be a value in
        :const:`matrix.SPARSE_PERMUTATION_METHODS`.
        optional, default: no permutation
    check_finite : bool
        Whether to check that the input matrix contains only finite numbers.
        Disabling may result in problems (crashes, non-termination)
        if the inputs do contain infinities or NaNs.
        Disabling gives a performance gain.
        optional, default: True
    return_type : str
        The type of the decomposition that should be calculated.
        It has to be a value in :const:`matrix.DECOMPOSITION_TYPES`.
        If return_type is None the type of the returned decomposition is
        chosen by the function itself.
        optional, default: the type of the decomposition is chosen by the function itself

    Returns
    -------
    matrix.decompositions.DecompositionBase
        A decompostion of `A` of type `return_type`.

    Raises
    ------
    matrix.errors.MatrixNoDecompositionPossibleError
        If the decomposition of `A` is not possible.
    matrix.errors.MatrixNotSquareError
        If `A` is not a square matrix.
    matrix.errors.MatrixNotFiniteError
        If `A` is not a finte matrix and `check_finite` is True.
    """

    if matrix.sparse.util.is_sparse(A):
        return matrix.sparse.calculate.decompose(A, permutation_method=permutation_method, check_finite=check_finite, return_type=return_type)
    else:
        return matrix.dense.calculate.decompose(A, permutation_method=permutation_method, check_finite=check_finite, return_type=return_type)


def approximate(A, t=None, min_diag_value=None, max_diag_value=None, min_abs_value=None, permutation_method=None, check_finite=True, return_type=None, callback=None):
    """
    Computes an approximative decomposition of a matrix.

    If `A` is decomposable in a decomposition of type `return_type`, this decomposition is returned.
    Otherwise a decomposition of type `return_type` is retuned which represents an approximation
    of `A`.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix
        The matrix that should be approximated by a decomposition.
        It is assumed, that A is Hermitian.
        The matrix must be a squared matrix.
    t : numpy.ndarray
        The targed vector used for the approximation. For each i in range(M)
        `min_diag_value <= t[i] <= max_diag_value` must hold.
        `t` and `A` must have the same length.
        optional, default : The diagonal of `A` is used as `t`.
    min_diag_value : float
        Each component of the diagonal of the matrix `D` in an returned `LDL` decomposition
        is forced to be greater or equal to `min_diag_value`.
        optional, default : 0.
    max_diag_value : float
        Each component of the diagonal of the matrix `D` in an returned `LDL` decomposition
        is forced to be lower or equal to `max_diag_value`.
        optional, default : No maximal value is forced.
    min_abs_value : float
        Absolute values below `min_abs_value` are considered as zero.
        optional, default : The resolution of the underlying data type is used.
    permutation_method : str
        The symmetric permutation method that is applied to the matrix before
        it is decomposed. It has to be a value in
        :const:`matrix.PERMUTATION_METHODS`.
        If `A` is sparse, it can also be a value in
        :const:`matrix.SPARSE_PERMUTATION_METHODS`.
        optional, default: No permutation is done.
    check_finite : bool
        Whether to check that the input matrix contains only finite numbers.
        Disabling may result in problems (crashes, non-termination)
        if the inputs do contain infinities or NaNs.
        Disabling gives a performance gain.
        optional, default: True
    return_type : str
        The type of the decomposition that should be calculated.
        It has to be a value in :const:`matrix.DECOMPOSITION_TYPES`.
        optional, default : The type of the decomposition is chosen by the function itself.
    callback : callable
        In each iteration `callback(i, r)` is called where `i` is the index of
        the row and column where components of `A` are reduced by the factor `r`.
        optional, default : No callback function is called.

    Returns
    -------
    matrix.decompositions.DecompositionBase
        An approximative decompostion of `A` of type `return_type`.

    Raises
    ------
    matrix.errors.MatrixNotSquareError
        If `A` is not a square matrix.
    matrix.errors.MatrixNotFiniteError
        If `A` is not a finte matrix and `check_finite` is True.
    """

    # convert input matrix A to needed type
    is_sparse = matrix.sparse.util.is_sparse(A)
    if not is_sparse:
        A = np.asanyarray(A)
    A_dtype = np.result_type(A.dtype, np.float)
    if is_sparse:
        A = A.astype(A_dtype)
    else:
        A = A.astype(A_dtype, copy=False)

    # check input matrix A
    matrix.util.check_square_matrix(A)
    matrix.util.check_finite(A, check_finite=check_finite)

    # check target vector t
    n = A.shape[0]
    if t is not None:
        t = np.asanyarray(t)
        if t.ndim != 1:
            raise ValueError('t has to be a one-dimensional array.')
        if len(t) != n:
            raise ValueError('The length of t {} must have the same length as the dimensions of A {}.'.format(len(t), n))

    # determine max_reduction_factor
    dtype_resolution = np.finfo(A.dtype).resolution
    max_reduction_factor = 1 - dtype_resolution * 10**2
    min_diag_value_LL = dtype_resolution * 10**2

    # check min_abs_value
    if min_abs_value is None:
        min_abs_value = dtype_resolution
    else:
        if min_abs_value < 0:
            raise ValueError('min_abs_value {} has to be greater or equal zero.'.format(min_abs_value))
        if min_abs_value < dtype_resolution:
            warnings.warn('Setting min_abs_value to resolution {} of matrix data type {}.'.format(dtype_resolution, A.dtype))
            min_abs_value = dtype_resolution

    # apply min_abs_value
    if min_abs_value > 0:
        if is_sparse:
            A.data[np.abs(A.data) < min_abs_value] = 0
            A.eliminate_zeros()
        else:
            A[np.abs(A) < min_abs_value] = 0

    # check min_diag_value and max_diag_value
    if min_diag_value is None:
        min_diag_value = min_diag_value_LL
    else:
        if return_type == matrix.constants.LL_DECOMPOSITION_TYPE:
            if min_diag_value < 0:
                raise ValueError('If return type is {}, min_diag_value {} has to be greater or equal zero .'.format(return_type, min_diag_value))
            elif min_diag_value < min_diag_value_LL:
                warnings.warn('Setting min_diag_value to resolution {} of matrix data type {}.'.format(min_diag_value, A.dtype))
        else:
            if min_diag_value <= 0:
                raise ValueError('Only min_diag_values greater zero are supported.')
            elif min_diag_value < min_diag_value_LL:
                warnings.warn('Setting min_diag_value to resolution {} of matrix data type {}.'.format(min_diag_value, A.dtype))

    if max_diag_value is None:
        max_diag_value = np.inf
    if min_diag_value > max_diag_value:
        raise ValueError('min_diag_value {} has to be lower or equal to max_diag_value {}.'.format(min_diag_value, max_diag_value))

    # check return type
    supported_return_types = matrix.constants.DECOMPOSITION_TYPES
    if return_type not in supported_return_types:
        raise ValueError('Unkown return type {}. Only values in {} are supported.'.format(return_type, supported_return_types))

    # check permutation method
    if permutation_method is not None:
        permutation_method = permutation_method.lower()
    if is_sparse:
        supported_permutation_methods = matrix.sparse.constants.PERMUTATION_METHODS
    else:
        supported_permutation_methods = matrix.dense.constants.PERMUTATION_METHODS
    if permutation_method not in supported_permutation_methods:
        raise ValueError('Permutation method {} is unknown. Only the following methods are supported {}.'.format(permutation_method, supported_permutation_methods))

    # apply permutation
    if permutation_method in matrix.constants.PERMUTATION_METHODS:
        p_first = matrix.permute.permutation_vector(A)
        A = matrix.permute.symmetric(A, p_first)
        decomposition_permutation_method = None
    else:
        p_first = None
        decomposition_permutation_method = permutation_method
        assert is_sparse and permutation_method in matrix.sparse.constants.CHOLMOD_PERMUTATION_METHODS

    # convert input matrix
    if is_sparse:
        A = matrix.sparse.util.convert_to_csc(A, sort_indices=True, eliminate_zeros=True)
    assert A.dtype == A_dtype

    # calculate approximation of A
    finished = False
    while not finished:

        # try to compute decomposition
        try:
            decomposition = decompose(A, permutation_method=decomposition_permutation_method, check_finite=False)
        except matrix.errors.MatrixNoDecompositionPossibleError as e:
            decomposition = e.subdecomposition
            bad_index = e.problematic_leading_principal_submatrix_index
        except matrix.errors.MatrixNoDecompositionPossibleTooManyEntriesError as e:
            if is_sparse and (A.indices.dtype != np.int64 or A.indptr != np.int64):
                warnings.warn('Problem to large for index type {}, index type is switched to long.'.format(e.matrix_index_type))
                A = matrix.sparse.util.convert_index_dtype(A, np.int64)
                return approximate(A, t=t, min_diag_value=min_diag_value, max_diag_value=max_diag_value, min_abs_value=min_abs_value, permutation_method=permutation_method, check_finite=False, return_type=return_type, callback=callback)
            else:
                raise
        else:
            bad_index = n

        # get diagonal values of current (sub-)decomposition
        decomposition = decomposition.to_any(matrix.constants.LDL_DECOMPOSITION_TYPE, matrix.constants.LDL_DECOMPOSITION_COMPRESSED_TYPE)
        decomposition._apply_previous_permutation(p_first)
        d = decomposition.d

        # get lowest index where decomposition is not possible
        bad_indices_mask = np.logical_or(d[:bad_index] < min_diag_value, d[:bad_index] > max_diag_value)
        bad_indices = np.where(bad_indices_mask)[0]
        if len(bad_indices) > 0:
            bad_index = np.min(bad_indices)
        del bad_indices

        # if not all diagonal entries okay, reduce
        finished = bad_index >= n
        if not finished:
            # apply permutation
            i_permuted = bad_index
            i = decomposition.p[i_permuted]

            # get A[i,i]
            if is_sparse:
                A_i_start_index = A.indptr[i]
                A_i_stop_index = A.indptr[i + 1]
                assert A_i_stop_index >= A_i_start_index
                A_ii_index_mask = np.where(A.indices[A_i_start_index:A_i_stop_index] == i)[0]
                if len(A_ii_index_mask) == 1:
                    A_ii_index = A_i_start_index + A_ii_index_mask[0]
                    assert A_i_start_index <= A_ii_index and A_i_stop_index >= A_ii_index
                    A_ii = A.data[A_ii_index]
                else:
                    assert len(A_ii_index_mask) == 0
                    A_ii_index = None
                    A_ii = 0
            else:
                A_ii = A[i, i]

            # get and check t[i]
            if t is None:
                t_i = A[i, i]
            else:
                t_i = t[i]
            if t_i < min_diag_value:
                raise ValueError('Each entry in the target vector t has to be greater or equal to min_diag_value {}. But its {}-th diagonal entry is {}.'.format(min_diag_value, i, t_i))
            if t_i > max_diag_value:
                raise ValueError('Each entry in the target vector t has to be lower or equal to max_diag_value {}. But its {}-th diagonal entry is {}.'.format(max_diag_value, i, t_i))

            # get L or LD
            if decomposition.is_type(matrix.constants.LDL_DECOMPOSITION_TYPE):
                L_or_LD = decomposition.L
            elif decomposition.is_type(matrix.constants.LDL_DECOMPOSITION_COMPRESSED_TYPE):
                L_or_LD = decomposition.LD
            else:
                assert False
            L_or_LD_row_i = L_or_LD[i_permuted]
            del decomposition, L_or_LD

            # get needed part of L and d
            if is_sparse:
                L_or_LD_row_i = L_or_LD_row_i.tocsr()
                L_or_LD_row_i_columns = L_or_LD_row_i.indices
                L_or_LD_row_i_data = L_or_LD_row_i.data
                assert len(L_or_LD_row_i_data) == len(L_or_LD_row_i_columns) >= 1
                assert L_or_LD_row_i_columns[-1] == i_permuted

                L_row_i_until_column_i = L_or_LD_row_i_data[:-1]
                d_until_i = d[L_or_LD_row_i_columns[:-1]]
            else:
                L_or_LD_row_i = L_or_LD_row_i.A1
                L_row_i_until_column_i = L_or_LD_row_i[:i_permuted]
                d_until_i = d[:i_permuted]

            # calculate reduction factor
            d_i_unmodified = A_ii - np.sum(L_row_i_until_column_i**2 * d_until_i)

            if d_i_unmodified < min_diag_value:
                reduction_factor = ((t_i - min_diag_value) / (t_i - d_i_unmodified))**(0.5)
            elif d_i_unmodified > max_diag_value:
                reduction_factor = ((max_diag_value - t_i) / (d_i_unmodified - t_i))**(0.5)
            elif np.isclose(d_i_unmodified, min_diag_value) or np.isclose(d_i_unmodified, max_diag_value):
                reduction_factor = max_reduction_factor
            elif d_i_unmodified == 0:
                if A_ii == 0 and t_i == 0:
                    reduction_factor = 0
                else:
                    reduction_factor = max_reduction_factor
            else:
                assert False
            assert 0 <= reduction_factor <= 1

            if reduction_factor > max_reduction_factor:
                reduction_factor = max_reduction_factor
            assert 0 <= reduction_factor < 1

            # apply reduction factor
            if t is None:
                A_ii_new = A_ii    # reduces rounding errors
            else:
                A_ii_new = (1 - reduction_factor**2) * t_i + reduction_factor**2 * A_ii

            if is_sparse:
                # set column
                A.data[A_i_start_index:A_i_stop_index] *= reduction_factor

                # apply min_abs_value in column
                set_to_zero_indices = np.where(np.abs(A.data[A_i_start_index:A_i_stop_index]) < min_abs_value)[0]
                set_to_zero_indices += A_i_start_index
                A.data[set_to_zero_indices] = 0
                del set_to_zero_indices

                # set row
                A_i_data = A.data[A_i_start_index:A_i_stop_index]
                A_i_rows = A.indices[A_i_start_index:A_i_stop_index]
                for j, A_ji in zip(A_i_rows, A_i_data):
                    if i != j:
                        A[i, j] = A_ji
                del A_i_data, A_i_rows

                # set diagonal entry
                if A_ii_index is not None:
                    A.data[A_ii_index] = A_ii_new
                elif A_ii_new != 0:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', scipy.sparse.SparseEfficiencyWarning)
                        A[i, i] = A_ii_new

                # eliminate zeros
                A.eliminate_zeros()
            else:
                # set column
                A[:, i] *= reduction_factor

                # apply min_abs_value in column
                A[np.abs(A[:, i].A1) < min_abs_value, i] = 0

                # set row
                A[i, :] = A[:, i].T

                # set diagonal entry
                A[i, i] = A_ii_new

            # call callback
            if callback is not None:
                callback(i, reduction_factor)

    # return
    assert np.all(d >= min_diag_value)
    assert np.all(d <= max_diag_value)

    decomposition = decomposition.to(return_type)
    return decomposition


def is_positive_semi_definite(A, check_finite=True):
    """
    Returns whether the passed matrix is positive semi-definite.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix
        The matrix that should be checked.
        It is assumed, that A is Hermitian.
        The matrix must be a squared matrix.
    check_finite : bool
        Whether to check that `A` contain only finite numbers.
        Disabling may result in problems (crashes, non-termination)
        if they contain infinities or NaNs.
        Disabling gives a performance gain.
        optional, default: True

    Returns
    -------
    bool
        Whether `A` is positive semi-definite.

    Raises
    ------
    matrix.errors.MatrixNotFiniteError
        If `A` is not a finte matrix and `check_finite` is True.
    """

    try:
        decomposition = decompose(A, permutation_method=matrix.constants.INCREASING_DIAGONAL_VALUES_PERMUTATION_METHOD, check_finite=check_finite)
    except (matrix.errors.MatrixNoDecompositionPossibleError, matrix.errors.MatrixNotSquareError):
        return False
    else:
        return decomposition.is_positive_semi_definite()


def is_positive_definite(A, check_finite=True):
    """
    Returns whether the passed matrix is positive definite.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix
        The matrix that should be checked.
        It is assumed, that A is Hermitian.
        The matrix must be a squared matrix.
    check_finite : bool
        Whether to check that `A` contain only finite numbers.
        Disabling may result in problems (crashes, non-termination)
        if they contain infinities or NaNs.
        Disabling gives a performance gain.
        optional, default: True

    Returns
    -------
    bool
        Whether `A` is positive definite.

    Raises
    ------
    matrix.errors.MatrixNotFiniteError
        If `A` is not a finte matrix and `check_finite` is True.
    """

    try:
        decomposition = decompose(A, permutation_method=matrix.constants.INCREASING_DIAGONAL_VALUES_PERMUTATION_METHOD, check_finite=check_finite)
    except (matrix.errors.MatrixNoDecompositionPossibleError, matrix.errors.MatrixNotSquareError):
        return False
    else:
        return decomposition.is_positive_definite()


def is_invertible(A, check_finite=True):
    """
    Returns whether the passed matrix is an invertible matrix.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix
        The matrix that should be checked.
        It is assumed, that A is Hermitian.
        The matrix must be a squared matrix.
    check_finite : bool
        Whether to check that `A` contain only finite numbers.
        Disabling may result in problems (crashes, non-termination)
        if they contain infinities or NaNs.
        Disabling gives a performance gain.
        optional, default: True

    Returns
    -------
    bool
        Whether `A` is invertible.

    Raises
    ------
    matrix.errors.MatrixNotFiniteError
        If `A` is not a finte matrix and `check_finite` is True.
    """

    try:
        decomposition = decompose(A, check_finite=check_finite)
    except matrix.errors.MatrixNotSquareError:
        return False
    else:
        return decomposition.is_invertible()


def solve(A, b, overwrite_b=False, check_finite=True):
    """
    Solves the equation `A x = b` regarding `x`.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix
        The matrix that should be checked.
        It is assumed, that A is Hermitian.
        The matrix must be a squared matrix.
    b : numpy.ndarray
        Right-hand side vector or matrix in equation `A x = b`.
        Ii must hold `b.shape[0] == A.shape[0]`.
    overwrite_b : bool
        Allow overwriting data in `b`.
        Enabling gives a performance gain.
        optional, default: False
    check_finite : bool
        Whether to check that `A` and b` contain only finite numbers.
        Disabling may result in problems (crashes, non-termination)
        if they contain infinities or NaNs.
        Disabling gives a performance gain.
        optional, default: True

    Returns
    -------
    numpy.ndarray
        An `x` so that `A x = b`.
        The shape of `x` matches the shape of `b`.

    Raises
    ------
    matrix.errors.MatrixNotSquareError
        If `A` is not a square matrix.
    matrix.errors.MatrixNotFiniteError
        If `A` is not a finte matrix and `check_finite` is True.
    matrix.errors.MatrixSingularError
        If `A` is singular.
    """

    decomposition = decompose(A, check_finite=check_finite)
    try:
        return decomposition.solve(b, overwrite_b=overwrite_b, check_finite=False)
    except matrix.errors.MatrixDecompositionSingularError as e:
        raise matrix.errors.MatrixSingularError from e
