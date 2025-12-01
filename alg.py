import random
import scipy
import numpy

'''
OPERATIONS WITH MATRICES AND VECTORS
'''

# generate a matrix of size (_row * _col) with random uniform numbers in range (_min, _max)
def gen_random_matrix(_row : int, _col : int, _min : float, _max : float):
    matrix = [ [0.0] * _col for i in range(_row)]
    for i in range (_row):
        for j in range(_col):
            matrix[i][j] = random.uniform(_min, _max)
    return matrix

# generates a matrix of size (size * size) with base sugar contents in the column zero and with degradeted or ripped sugar content in the rest columns
def gen_random_base_matrix(_size: int, _min_a: float, _max_a: float, _matrix_of_coefficients : list):
    matrix = [ [0.0] * _size for i in range(_size)]
    for i in range(_size):
        matrix[i][0] = random.uniform(_min_a, _max_a) # only first column (base sugar content)
    for i in range(_size):
        for j in range(1, _size):
            matrix[i][j] = matrix[i][j - 1] * _matrix_of_coefficients[i][j]
    return matrix

# merge two matrices with the same amount of rows
def merge_matrices(_matrix_riping: list, _matrix_degradation: list):
    size = len(_matrix_riping)
    result_matrix = [ [0.0] * size for i in range(size)]
    size_riping = len(_matrix_riping[0])
    size_degradation = len(_matrix_degradation[0])
    for i in range(size):
        for j in range(size_riping):
            result_matrix[i][j] = _matrix_riping[i][j]
        for j in range(size_degradation):
            result_matrix[i][size_riping + j] = _matrix_degradation[i][j]
    return result_matrix

# shows matrix
def show_matrix(_matrix: list):
    _row = len(_matrix)
    _col = len(_matrix[0])
    for i in range(_row):
        for j in range(_col):
            print(_matrix[i][j], end = ' ')
        print()

# returns an integer value which is not in list _vector
def get_free_index(_vector: list, _size: int):
    for i in range(_size):
        if i not in _vector:
            return i
    return 0

'''
CREATING THE INORGANIC
'''

def gen_inorganic_matrix(_size :int, _minK : float, _maxK : float, _minNa : float, _maxNa : float, _minN : float, _maxN : float):
    matrix = [[0.0] * 3 for i in range(_size)]

    for i in range(_size):
        matrix[i][0] = random.uniform(_minK, _maxK)
        matrix[i][1] = random.uniform(_minNa, _maxNa)
        matrix[i][2] = random.uniform(_minN, _maxN)

    return matrix

def braunschweig(compos: list) -> float:
    K = compos[0]
    Na = compos[1]
    N = compos[2]
    return (0.12 * (K + Na) + 0.24 * N + 0.48) / 100

def get_inorganic(_matrix: list, _inorganic_matrix: list):
    size = len(_matrix)
    new_matrix = [ [0.0] * size for i in range(size)]

    for variety in range(size):
        braun_res = braunschweig(_inorganic_matrix[variety])
        for day in range(size):
            if braun_res < _matrix[day][variety]:
                new_matrix[day][variety] = _matrix[day][variety] - braun_res
            else:
                new_matrix[day][variety] = 0

    return new_matrix


'''
ALGORITHMS
'''

# greedy algorithm
# returns a result value and permutation
def greedy_algorithm(_matrix: list):
    result = list()
    permutation = []
    size = len(_matrix[0]) # amount of rows/columns in _matrix

    for j in range(size): # number of column
        local_max_index = get_free_index(permutation, size)
        local_max_value = _matrix[local_max_index][j]
        for i in range(size): # number of row
            if _matrix[i][j] > local_max_value:
                if i not in permutation:
                    local_max_value = _matrix[i][j]
                    local_max_index = i
        if j == 0: result.append(local_max_value)
        else: result.append(result[j-1] + local_max_value)
        permutation.append(local_max_index)
    return result, permutation

# thrifty algorithm
# returns a result value and permutation
def thrifty_algorithm(_matrix: list):
    result = list()
    permutation = []
    size = len(_matrix[0]) # amount of rows/columns in _matrix

    for j in range(size): # number of column
        local_min_index = get_free_index(permutation, size)
        local_min_value = _matrix[local_min_index][j]
        for i in range(size): # number of row
            if _matrix[i][j] < local_min_value:
                if i not in permutation:
                    local_min_value = _matrix[i][j]
                    local_min_index = i
        if (j == 0): result.append(local_min_value)
        else: result.append(result[j-1] + local_min_value)
        permutation.append(local_min_index)
    return result, permutation

# greedy_v_thrifty algorithm
# inputs matrix N*N and parameter v (period of riping)
# returns a result value and permutation
def greedy_v_thrifty_algorithm(_matrix: list, _v: int):
    result = list()
    permutation = []
    size = len(_matrix[0]) # amount of rows/columns in _matrix

    # while riping - greedy algorithm
    for j in range(_v): # number of column
        local_max_index = get_free_index(permutation, size)
        local_max_value = _matrix[local_max_index][j]
        for i in range(size): # number of row
            if _matrix[i][j] > local_max_value:
                if i not in permutation:
                    local_max_value = _matrix[i][j]
                    local_max_index = i
        if j == 0: result.append(local_max_value)
        else: result.append(result[j-1] + local_max_value)
        permutation.append(local_max_index)

    # while degrading - thrifty algorithm
    for j in range(_v, size): # number of column
        local_min_index = get_free_index(permutation, size)
        local_min_value = _matrix[local_min_index][j]
        for i in range(size): # number of row
            if _matrix[i][j] < local_min_value:
                if i not in permutation:
                    local_min_value = _matrix[i][j]
                    local_min_index = i
        if j == 0: result.append(local_min_value)
        else: result.append(result[j-1] + local_min_value)
        permutation.append(local_min_index)

    return result, permutation

# thrifty_v_greedy algorithm
# inputs matrix N*N and parameter v (period of riping)
# returns a result value and permutation
def thrifty_v_greedy_algorithm(_matrix: list, _v: int):
    result = list()
    permutation = []
    size = len(_matrix[0]) # amount of rows/columns in _matrix

    # while riping - thrifty algorithm
    for j in range(_v): # number of column
        local_min_index = get_free_index(permutation, size)
        local_min_value = _matrix[local_min_index][j]
        for i in range(size): # number of row
            if _matrix[i][j] < local_min_value:
                if i not in permutation:
                    local_min_value = _matrix[i][j]
                    local_min_index = i
        if j == 0: result.append(local_min_value)
        else: result.append(result[j-1] + local_min_value)
        permutation.append(local_min_index)

    # while degrading - greedy algorithm
    for j in range(_v, size): # number of column
        local_max_index = get_free_index(permutation, size)
        local_max_value = _matrix[local_max_index][j]
        for i in range(size): # number of row
            if _matrix[i][j] > local_max_value:
                if i not in permutation:
                    local_max_value = _matrix[i][j]
                    local_max_index = i
        if j == 0: result.append(local_max_value)
        else: result.append(result[j - 1] + local_max_value)
        permutation.append(local_max_index)

    return result, permutation

# hungarian_max algorithm
def hungarian_max_algorithm(_matrix: list):
    size = len(_matrix)

    # delete max element in row
    for i in range(size):
        max_value = _matrix[i][0]
        for j in range(size):
            if _matrix[i][j] > max_value:
                max_value = _matrix[i][j]
        for j in range(size):
            _matrix[i][j] = max_value - _matrix[i][j]

    copy_matrix = numpy.copy(_matrix)

    row_indices, col_indices = scipy.optimize.linear_sum_assignment(copy_matrix)


    result = [_matrix[row_indices[0]][col_indices[0]]]
    for i in range(1, len(row_indices)):
        result.append(result[i-1] + _matrix[row_indices[i]][col_indices[i]])

    for i in range(len(row_indices)):
        row_indices[col_indices[i]] = i
    return result, row_indices
