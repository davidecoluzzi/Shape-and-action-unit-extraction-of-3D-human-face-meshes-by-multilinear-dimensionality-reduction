import tempfile
import warnings

import numpy as np
import scipy.sparse
import pytest

import matrix
import matrix.constants
import matrix.decompositions
import matrix.permute


# *** random values *** #

def random_matrix(n, m, dense=True):
    random_state = 1234
    if dense:
        np.random.seed(random_state)
        A = np.random.rand(n, m)
        A = np.asmatrix(A)
    else:
        density = 0.1
        A = scipy.sparse.rand(n, m, density=density, random_state=random_state)
        A = A.tocsc()
    return A


def random_square_matrix(n, dense=True, positive_semi_definite=False, positive_definite=False, min_diag_value=None):
    A = random_matrix(n, n, dense=dense)
    A = A + A.H
    if positive_semi_definite or positive_definite:
        A = A @ A
    if min_diag_value is not None or positive_definite:
        if min_diag_value is None:
            min_diag_value = 0
        if positive_definite:
            min_diag_value = max(min_diag_value, 1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', scipy.sparse.SparseEfficiencyWarning)
            for i in range(n):
                A[i, i] += min_diag_value
    return A


def random_lower_triangle_matrix(n, dense=True):
    A = random_matrix(n, n, dense=dense)
    if dense:
        A = np.tril(A)
    else:
        A = scipy.sparse.tril(A).tocsc()
    return A


def random_vector(n):
    random_state = 1234
    np.random.seed(random_state)
    v = np.random.rand(n)
    return v


def random_permutation_vector(n):
    random_state = 1234
    np.random.seed(random_state)
    p = np.arange(n)
    np.random.shuffle(p)
    return p


def random_decomposition(decomposition_type, n, dense=True, finite=True, invertible=None):
    # make random decomposition
    LD = random_lower_triangle_matrix(n, dense=dense)
    p = random_permutation_vector(n)
    decomposition = matrix.decompositions.LDL_DecompositionCompressed(LD, p)
    # apply finite
    if not finite:
        decomposition = decomposition.to(matrix.constants.LDL_DECOMPOSITION_TYPE)
        decomposition.d[np.random.randint(n)] = np.nan
    # apply singular
    if invertible is not None:
        decomposition = decomposition.to(matrix.constants.LDL_DECOMPOSITION_TYPE)
        if invertible:
            decomposition.d = decomposition.d + 1
        else:
            decomposition.d[np.random.randint(n)] = 0
    # return correct type
    decomposition = decomposition.to(decomposition_type)
    return decomposition


# *** permute *** #

test_permute_setups = [
    (n, dense)
    for n in (100,)
    for dense in (True, False)
]


@pytest.mark.parametrize('n, dense', test_permute_setups)
def test_permute(n, dense):
    p = random_permutation_vector(n)
    A = random_square_matrix(n, dense=dense, positive_semi_definite=True)
    A_permuted = matrix.permute.symmetric(A, p)
    for i in range(n):
        for j in range(n):
            assert A[p[i], p[j]] == A_permuted[i, j]
    p_inverse = matrix.permute.invert_permutation_vector(p)
    np.testing.assert_array_equal(p[p_inverse], np.arange(n))
    np.testing.assert_array_equal(p_inverse[p], np.arange(n))


# *** equal *** #

test_equal_setups = [
    (n, dense, decomposition_type)
    for n in (100,)
    for dense in (True, False)
    for decomposition_type in matrix.constants.DECOMPOSITION_TYPES
]


@pytest.mark.parametrize('n, dense, decomposition_type', test_equal_setups)
def test_equal(n, dense, decomposition_type):
    decomposition = random_decomposition(decomposition_type, n, dense=dense)
    for (n_other, dense_other, decomposition_type_other) in test_equal_setups:
        decomposition_other = random_decomposition(decomposition_type_other, n_other, dense=dense_other)
        equal = n == n_other and dense == dense_other and decomposition_type == decomposition_type_other
        equal_calculated = decomposition == decomposition_other
        assert equal == equal_calculated


# *** convert *** #

test_convert_setups = [
    (n, dense, decomposition_type, copy)
    for n in (100,)
    for dense in (True, False)
    for decomposition_type in matrix.constants.DECOMPOSITION_TYPES
    for copy in (True, False)
]


@pytest.mark.parametrize('n, dense, decomposition_type, copy', test_convert_setups)
def test_convert(n, dense, decomposition_type, copy):
    decomposition = random_decomposition(decomposition_type, n, dense=dense)
    for convert_decomposition_type in matrix.constants.DECOMPOSITION_TYPES:
        converted_decomposition = decomposition.to(convert_decomposition_type, copy=copy)
        equal = decomposition_type == convert_decomposition_type
        equal_calculated = decomposition == converted_decomposition
        assert equal == equal_calculated


# *** decompose *** #

def supported_permutation_methods(dense):
    if dense:
        return matrix.PERMUTATION_METHODS
    else:
        return matrix.SPARSE_PERMUTATION_METHODS


test_decompose_setups = [
    (n, dense, permutation_method, check_finite, return_type)
    for n in (100,)
    for dense in (True, False)
    for permutation_method in supported_permutation_methods(dense)
    for check_finite in (True, False)
    for return_type in matrix.DECOMPOSITION_TYPES
]


@pytest.mark.parametrize('n, dense, permutation_method, check_finite, return_type', test_decompose_setups)
def test_decompose(n, dense, permutation_method, check_finite, return_type):
    A = random_square_matrix(n, dense=dense, positive_semi_definite=True)
    if dense:
        A_dense = A
    else:
        A_dense = A.todense()
    decomposition = matrix.decompose(A, permutation_method=permutation_method, check_finite=check_finite, return_type=return_type)
    A_composed = decomposition.composed_matrix
    if dense:
        A_composed_dense = A_composed
    else:
        A_composed_dense = A_composed.todense()
    np.testing.assert_array_almost_equal(A_dense, A_composed_dense)


# *** positive definite *** #

test_positive_definite_setups = [
    (n, dense)
    for n in (100,)
    for dense in (True, False)
]


@pytest.mark.parametrize('n, dense', test_positive_definite_setups)
def test_positive_definite(n, dense):
    A = random_square_matrix(n, dense=dense, positive_semi_definite=True)
    assert matrix.is_positive_semi_definite(A)
    assert not matrix.is_positive_semi_definite(-A)
    A = random_square_matrix(n, dense=dense, positive_definite=True)
    assert matrix.is_positive_semi_definite(A)
    assert not matrix.is_positive_semi_definite(-A)
    assert matrix.is_positive_definite(A)
    assert not matrix.is_positive_definite(-A)


# *** approximate *** #

test_approximate_setups = [
    (n, dense, permutation_method, check_finite, return_type, t, min_diag_value, max_diag_value_shift, min_abs_value)
    for n in (10,)
    for dense in (True, False)
    for permutation_method in supported_permutation_methods(dense)
    for check_finite in (True, False)
    for return_type in matrix.DECOMPOSITION_TYPES
    for min_abs_value in (None, 10**-8)
    for min_diag_value in (None, 10**-4)
    for max_diag_value_shift in (None, 1 + np.random.rand(1) * 10)
    for t in (None, np.random.rand(n) + 10**-4)
]


@pytest.mark.parametrize('n, dense, permutation_method, check_finite, return_type, t, min_diag_value, max_diag_value_shift, min_abs_value', test_approximate_setups)
def test_approximate(n, dense, permutation_method, check_finite, return_type, t, min_diag_value, max_diag_value_shift, min_abs_value):
    if t is None:
        if min_diag_value is None:
            A_min_diag_value = 10**-6
        else:
            A_min_diag_value = min_diag_value
    else:
        A_min_diag_value = None
        assert min_diag_value is None or min_diag_value <= t.min()

    A = random_square_matrix(n, dense=dense, min_diag_value=A_min_diag_value)

    if max_diag_value_shift is not None:
        if t is None:
            max_diag_value = A.diagonal().max()
        else:
            max_diag_value = t.max()
        max_diag_value = max_diag_value + max_diag_value_shift
    else:
        max_diag_value = None

    decomposition = matrix.approximate(A, t=t, min_diag_value=min_diag_value, max_diag_value=max_diag_value, min_abs_value=min_abs_value, permutation_method=permutation_method, check_finite=check_finite, return_type=return_type)
    if min_diag_value is not None or max_diag_value is not None:
        decomposition = decomposition.to(matrix.LDL_DECOMPOSITION_TYPE)
        if min_diag_value is not None:
            assert decomposition.d.min() >= min_diag_value
        if max_diag_value is not None:
            assert decomposition.d.max() <= max_diag_value


# *** save and load *** #

test_save_and_load_setups = [
    (n, dense, decomposition_type, filename_prefix)
    for n in (100,)
    for dense in (True, False)
    for decomposition_type in matrix.constants.DECOMPOSITION_TYPES
    for filename_prefix in (None, 'TEST')
]


@pytest.mark.parametrize('n, dense, decomposition_type, filename_prefix', test_save_and_load_setups)
def test_save_and_load(n, dense, decomposition_type, filename_prefix):
    decomposition = random_decomposition(decomposition_type, n, dense=dense)
    decomposition_other = type(decomposition)()
    with tempfile.TemporaryDirectory() as tmp_dir:
        decomposition.save(tmp_dir, filename_prefix=filename_prefix)
        decomposition_other.load(tmp_dir, filename_prefix=filename_prefix)
    assert decomposition == decomposition_other


# *** is finite *** #

test_is_finite_setups = [
    (n, dense, decomposition_type, finite)
    for n in (100,)
    for dense in (True, False)
    for decomposition_type in matrix.constants.DECOMPOSITION_TYPES
    for finite in (True, False)
]


@pytest.mark.parametrize('n, dense, decomposition_type, finite', test_is_finite_setups)
def test_is_finite(n, dense, decomposition_type, finite):
    # make random decomposition
    decomposition = random_decomposition(decomposition_type, n, dense=dense, finite=finite)
    # test
    assert decomposition.is_finite() == finite
    if not finite:
        with np.testing.assert_raises(matrix.errors.MatrixDecompositionNotFiniteError):
            decomposition.check_finite()
    else:
        decomposition.check_finite()


# *** is invertible *** #

test_is_invertible_setups = [
    (n, dense, decomposition_type, invertible)
    for n in (100,)
    for dense in (True, False)
    for decomposition_type in matrix.constants.DECOMPOSITION_TYPES
    for invertible in (True, False)
]


@pytest.mark.parametrize('n, dense, decomposition_type, invertible', test_is_finite_setups)
def test_is_invertible(n, dense, decomposition_type, invertible):
    # make random decomposition
    decomposition = random_decomposition(decomposition_type, n, dense=dense, invertible=invertible)
    # test
    assert decomposition.is_invertible() == invertible
    if not invertible:
        with np.testing.assert_raises(matrix.errors.MatrixDecompositionSingularError):
            decomposition.check_invertible()
    else:
        decomposition.check_invertible()


# *** solve *** #

test_solve_setups = [
    (n, dense, decomposition_type, invertible, b)
    for n in (10,)
    for dense in (True, False)
    for decomposition_type in matrix.constants.DECOMPOSITION_TYPES
    for invertible in (True, False)
    for b in (random_vector(n), np.zeros(n), np.arange(n))
]


@pytest.mark.parametrize('n, dense, decomposition_type, invertible, b', test_solve_setups)
def test_solve(n, dense, decomposition_type, invertible, b):
    # make random decomposition
    decomposition = random_decomposition(decomposition_type, n, dense=dense, finite=True, invertible=invertible)
    if invertible:
        # calculate solution
        x = decomposition.solve(b)
        # verify solution
        A = decomposition.composed_matrix
        y = A @ x
        if dense:
            y = y.A1
        assert np.all(np.isclose(b, y))
    else:
        with np.testing.assert_raises(matrix.errors.MatrixDecompositionSingularError):
            decomposition.solve(b)
