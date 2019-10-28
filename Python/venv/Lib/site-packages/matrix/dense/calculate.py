import numpy as np
import scipy.linalg.lapack
import scipy.linalg.misc

import matrix.dense.constants
import matrix.dense.permute
import matrix.dense.util

import matrix.decompositions
import matrix.errors
import matrix.permute
import matrix.util


def _decompose(A, permutation_method=None, check_finite=True, return_type=None, overwrite=False, clean=True):
    """
    Computes a decomposition of a dense matrix.

    Parameters
    ----------
    A : numpy.ndarray
        Matrix to be decomposed.
        It is assumed, that A is Hermitian.
        The matrix must be a squared matrix.
    permutation_method : str
        The symmetric permutation method that is applied to the matrix before
        it is decomposed. It has to be a value in
        :const:`matrix.dense.constants.PERMUTATION_METHODS`.
        optional, default: no permutation
    check_finite : bool
        Whether to check that the input matrix contains only finite numbers.
        Disabling may result in problems (crashes, non-termination)
        if the inputs do contain infinities or NaNs.
        Disabling gives a performance gain.
        optional, default: True
    return_type : str
        The type of the decomposition that should be calculated.
        It has to be a value in :const:`matrix.constants.DECOMPOSITION_TYPES`.
        If return_type is None the type of the returned decomposition is
        chosen by the function itself.
        optional, default: the type of the decomposition is chosen by the function itself
    overwrite : bool
        Whether to overwrite data in `A`.
        (enabling may improve performance)
        optional, default: False
    clean : bool
        Whether to set the zero entries in the triangular Cholesky to zero.
        Disabling gives a performance gain.
        optional, default: True

    Returns
    -------
    matrix.decompositions.DecompositionBase
        A decompostion of `A`. If `return_type` is not None, the decomposition
        is of this type.

    Raises
    ------
    matrix.errors.MatrixNoDecompositionPossibleError
        If the decomposition of `A` is not possible.
    matrix.errors.MatrixNotSquareError
        If `A` is not a square matrix.
    matrix.errors.MatrixNotFiniteError
        If `A` is not a finte matrix and `check_finite` is True.
    """

    # convert matrix to array
    A_original = A
    A = np.asanyarray(A_original)
    overwrite = overwrite or scipy.linalg.misc._datacopied(A, A_original)
    del A_original

    # check matrix A
    matrix.util.check_square_matrix(A)
    matrix.dense.util.check_finite(A, check_finite=check_finite)

    # check and apply permutation_method
    if permutation_method is not None:
        permutation_method = permutation_method.lower()
    supported_permutation_methods = matrix.dense.constants.PERMUTATION_METHODS
    if permutation_method not in supported_permutation_methods:
        raise ValueError('Permutation method {} is unknown. Only the following methods are supported {}.'.format(permutation_method, supported_permutation_methods))

    p = matrix.permute.permutation_vector(A, permutation_method)
    A = matrix.dense.permute.symmetric(A, p)
    overwrite = overwrite or p is not None

    # check return type
    supported_return_type = matrix.dense.constants.DECOMPOSITION_TYPES
    if return_type is not None and return_type not in supported_return_type:
        raise ValueError('Unkown decomposition type {}. Only values in {} are supported.'.format(
            return_type, supported_return_type))

    # call lapack Cholesky function
    potrf = scipy.linalg.lapack.get_lapack_funcs('potrf', (A,))
    L, exit_code = potrf(A, lower=True, overwrite_a=overwrite, clean=clean)

    # make decomposition
    decomposition = matrix.decompositions.LL_Decomposition(L, p=p)

    # check exit code
    if exit_code > 0:
        bad_index = exit_code - 1
        decomposition.L[bad_index, bad_index] = np.nan
        raise matrix.errors.MatrixNoLLDecompositionPossibleError(
            A, problematic_leading_principal_submatrix_index=bad_index, subdecomposition=decomposition)
    if exit_code < 0:
        raise ValueError('The {}-th argument of the internal potrf function is invalid.'.format(-exit_code))

    # return decomposition
    assert exit_code == 0
    return decomposition.to(return_type)


def decompose(A, permutation_method=None, check_finite=True, return_type=None):
    """
    Computes a decomposition of a dense matrix.

    Parameters
    ----------
    A : numpy.ndarray
        Matrix to be decomposed.
        It is assumed, that A is Hermitian.
        The matrix must be a squared matrix.
    permutation_method : str
        The symmetric permutation method that is applied to the matrix before
        it is decomposed. It has to be a value in
        :const:`matrix.dense.constants.PERMUTATION_METHODS`.
        optional, default: no permutation
    check_finite : bool
        Whether to check that the input matrix contains only finite numbers.
        Disabling may result in problems (crashes, non-termination)
        if the inputs do contain infinities or NaNs.
        Disabling gives a performance gain.
        optional, default: True
    return_type : str
        The type of the decomposition that should be calculated.
        It has to be a value in :const:`matrix.constants.DECOMPOSITION_TYPES`.
        If return_type is None the type of the returned decomposition is
        chosen by the function itself.
        optional, default: the type of the decomposition is chosen by the function itself

    Returns
    -------
    matrix.decompositions.DecompositionBase
        A decompostion of `A`. If `return_type` is not None, the decomposition
        is of this type.

    Raises
    ------
    matrix.errors.MatrixNoDecompositionPossibleError
        If the decomposition of `A` is not possible.
    matrix.errors.MatrixNotSquareError
        If `A` is not a square matrix.
    matrix.errors.MatrixNotFiniteError
        If `A` is not a finte matrix and `check_finite` is True.
    """

    return _decompose(A, permutation_method=permutation_method, check_finite=check_finite, return_type=return_type)
