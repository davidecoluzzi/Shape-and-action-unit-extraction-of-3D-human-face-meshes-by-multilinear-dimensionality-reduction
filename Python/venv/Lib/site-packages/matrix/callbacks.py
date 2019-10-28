import numpy as np


def display():
    """ Returns a callback function for :func:`matrix.approximate` which prints each iteration.

    Returns
    -------
    callable
        A callback function for :func:`matrix.approximate` which prints each iteration.
    """

    def callback_function(i, reduction_factor):
        print('Row/column {} is reduced with reduction factor {}.'.format(i, reduction_factor))
    return callback_function


def save(reduction_factors_file, n):
    """ Returns a callback function for :const:`matrix.approximate` which saves each iteration in a :mod:`numpy` file.

    Parameters
    ----------
    reduction_factors_file : str
        The file where the reduction factors are saved.
    n : int
        The dimension of the squared matrix that is approximated.

    Returns
    -------
    callable
        A callback function for :func:`matrix.approximate` which saves each iteration in a :mod:`numpy` file.
    """

    try:
        reduction_factors = np.load(reduction_factors_file, mmap_mode='r+')
    except FileNotFoundError:
        reduction_factors = np.ones(n)
        np.save(reduction_factors_file, reduction_factors)
        reduction_factors = np.load(reduction_factors_file, mmap_mode='r+')
    else:
        if reduction_factors.shape != (n,):
            raise ValueError('The reduction factors file contains an array with shape {} but the expected shape is {}'.format(reduction_factors.shape, (n,)))

    def callback_function(i, reduction_factor):
        reduction_factors[i] = reduction_factors[i] * reduction_factor
        reduction_factors.flush()

    return callback_function
