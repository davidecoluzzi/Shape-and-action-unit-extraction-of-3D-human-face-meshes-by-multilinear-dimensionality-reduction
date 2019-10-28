# *** base exception *** #


class MatrixError(Exception):
    """ An exception related to a matrix.

    This is the base exception for all exceptions in this package.
    """

    def __init__(self, matrix=None, message=None):
        self.matrix = matrix
        if message is None:
            message = 'Error with matrix{matrix_decription}!'
        message = message.format(matrix_decription=self._matrix_decription)
        self.message = message
        super().__init__(message)

    def __str__(self):
        return self.message

    @property
    def _matrix_decription(self):
        if self.matrix is not None:
            return ' with shape {}'.format(self.matrix.shape)
        else:
            return ''


# *** matrix exceptions *** #

class MatrixNotSquareError(MatrixError):
    """ A matrix is not a square matrix although this is required. """

    def __init__(self, matrix=None):
        message = 'Matrix{matrix_decription} is not a square matrix.'
        super().__init__(matrix=matrix, message=message)


class MatrixNotFiniteError(MatrixError):
    """ A matrix has non-finite entries although a finite matrix is required. """

    def __init__(self, matrix=None):
        message = 'Matrix{matrix_decription} has not finite entries.'
        super().__init__(matrix=matrix, message=message)


class MatrixSingularError(MatrixError):
    """ A matrix is singular although an invertible matrix is required. """

    def __init__(self, matrix=None):
        message = 'Matrix{matrix_decription} is singular.'
        super().__init__(matrix=matrix, message=message)


# *** general decomposition exceptions *** #

class MatrixDecompositionNotFiniteError(MatrixError):
    """ A decomposition of a matrix has non-finite entries although a finite matrix is required. """

    def __init__(self, decomposition=None):
        # compose message
        message = 'Decomposition '
        if decomposition is not None:
            self.decomposition = decomposition
            message += str(decomposition) + ' '
        message += 'has non-finite entries.'
        super().__init__(message=message)


class MatrixDecompositionSingularError(MatrixError):
    """ A decomposition represents a singular matrix although a non-singular matrix is required. """

    def __init__(self, decomposition=None):
        # compose message
        message = 'Decomposition '
        if decomposition is not None:
            self.decomposition = decomposition
            message += str(decomposition) + ' '
        message += 'represents a singular matrix.'
        super().__init__(message=message)


# *** decomposition not calculable exceptions *** #

class MatrixNoDecompositionPossibleError(MatrixError):
    """ The matrix decomposition is not possible for this matrix. """

    def __init__(self, matrix=None, decomposition_decription=None, message=None):
        # compose message
        total_message = 'It was not possible to compute a '
        if decomposition_decription is not None:
            total_message += decomposition_decription + ' '
        total_message += 'decomposition for matrix{matrix_decription}.'
        if message is not None:
            total_message += message
        super().__init__(matrix=matrix, message=total_message)


class MatrixNoDecompositionPossibleTooManyEntriesError(MatrixNoDecompositionPossibleError):
    """ The matrix decomposition is not possible for this matrix because it would have too many entries. """

    def __init__(self, matrix_index_type=None):
        message = 'Matrix '
        if matrix_index_type is not None:
            message += 'of type {} '.format(matrix_index_type)
        message += 'can not store as many entries.'
        super().__init__(message=message)
        self.matrix_index_type = matrix_index_type


class MatrixNoDecompositionPossibleWithProblematicSubdecompositionError(MatrixNoDecompositionPossibleError):
    """ The desired matrix decomposition is not possible for this matrix. Only a subdecompostion could be calculated """

    def __init__(self, matrix=None, decomposition_decription=None, problematic_leading_principal_submatrix_index=None, subdecomposition=None):
        # compose message
        message = 'It was not possible to compute a '
        if decomposition_decription is not None:
            message += decomposition_decription + ' '
        message += 'decomposition for matrix{matrix_decription}.'
        if problematic_leading_principal_submatrix_index is not None:
            message += ' A non-decomposable leading principal submatrix is up to row/column {}.'.format(problematic_leading_principal_submatrix_index)
        super().__init__(matrix=matrix, message=message)
        # store infos
        if problematic_leading_principal_submatrix_index is not None:
            self.problematic_leading_principal_submatrix_index = problematic_leading_principal_submatrix_index
        if subdecomposition is not None:
            self.subdecomposition = subdecomposition


class MatrixNoLDLDecompositionPossibleError(MatrixNoDecompositionPossibleWithProblematicSubdecompositionError):
    """ A LDL decomposition is not possible for this matrix. """

    def __init__(self, matrix=None, problematic_leading_principal_submatrix_index=None, subdecomposition=None):
        super().__init__(matrix=matrix, decomposition_decription='LDL^H', problematic_leading_principal_submatrix_index=problematic_leading_principal_submatrix_index, subdecomposition=subdecomposition)


class MatrixNoLLDecompositionPossibleError(MatrixNoDecompositionPossibleWithProblematicSubdecompositionError):
    """ A LL decomposition is not possible for this matrix. """

    def __init__(self, matrix=None, problematic_leading_principal_submatrix_index=None, subdecomposition=None):
        super().__init__(matrix=matrix, decomposition_decription='LL^H', problematic_leading_principal_submatrix_index=problematic_leading_principal_submatrix_index, subdecomposition=subdecomposition)


class MatrixDecompositionNoConversionImplementedError(MatrixError):
    """ A decomposition conversion is not implemented for this type. """

    def __init__(self, original_decomposition=None, desired_decomposition_type=None):
        message = 'Decomposition'
        if original_decomposition is not None:
            message += ' {}'.format(type(original_decomposition).__name__)
        message += ' can not be converted'
        if desired_decomposition_type is not None:
            message += ' to decomposition {}'.format(desired_decomposition_type)
        message += '.'
        super().__init__(message=message)
