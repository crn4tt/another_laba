from __future__ import annotations  

import random
from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

Matrix = List[List[float]]


def _validate_dimensions(matrix: Sequence[Sequence[float]]) -> None:
    """Ensure the matrix is rectangular and not empty"""
    if not matrix:
        raise ValueError("Matrix must contain at least one row.")
    row_length = len(matrix[0])
    if row_length == 0:
        raise ValueError("Matrix rows must not be empty.")
    for row in matrix:
        if len(row) != row_length:
            raise ValueError("All matrix rows must have the same length.")


def random_matrix(rows: int, cols: int, min_value: float, max_value: float) -> Matrix:
    """Generate a matrix populated with random values in the given range"""
    if rows <= 0 or cols <= 0:
        raise ValueError("Matrix dimensions must be positive.")
    if min_value > max_value:
        raise ValueError("Minimum value cannot exceed maximum value.")

    return [[random.uniform(min_value, max_value) for _ in range(cols)] for _ in range(rows)]


def concentrated_matrix(
    rows: int, cols: int, min_val: float, max_val: float
) -> Matrix:
    """Generates a matrix with concentrated distribution"""
    if rows <= 0 or cols <= 0:
        raise ValueError("Matrix dimensions must be positive.")
    if min_val > max_val:
        raise ValueError("Minimum value cannot exceed maximum value.")

    base_delta = (max_val - min_val) / 4 if (max_val - min_val) > 0 else 0.0

    matrix: Matrix = []
    for _ in range(rows):
        delta_i = random.uniform(0, base_delta) if base_delta > 0 else 0.0

        if delta_i == 0:
            beta_i1 = beta_i2 = random.uniform(min_val, max_val)
        else:
            beta_i1 = random.uniform(min_val, max_val - delta_i)
            beta_i2 = beta_i1 + delta_i

        row = [random.uniform(beta_i1, beta_i2) for _ in range(cols)]
        matrix.append(row)

    return matrix


def base_sugar_matrix(
    size: int, min_sugar: float, max_sugar: float, coefficients: Matrix
) -> Matrix:
    """Generate the base sugar matrix using coefficient multipliers for each day"""
    if size <= 0:
        raise ValueError("Size must be positive.")
    _validate_dimensions(coefficients)
    if len(coefficients) != size:
        raise ValueError("Coefficient matrix must have the same number of rows as size.")
    if any(len(row) != size for row in coefficients):
        raise ValueError("Coefficient matrix must be square with dimensions equal to size.")
    if min_sugar > max_sugar:
        raise ValueError("Minimum sugar cannot exceed maximum sugar.")

    matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    for variety in range(size):
        matrix[variety][0] = random.uniform(min_sugar, max_sugar)
        for day in range(1, size):
            matrix[variety][day] = matrix[variety][day - 1] * coefficients[variety][day]
    return matrix


def merge_matrices(left: Matrix, right: Matrix) -> Matrix:
    """Merge matrices horizontally, ensuring they share the same row count"""
    _validate_dimensions(left)
    _validate_dimensions(right)
    if len(left) != len(right):
        raise ValueError("Matrices must have the same number of rows to merge.")

    merged: Matrix = []
    for left_row, right_row in zip(left, right):
        merged.append([*left_row, *right_row])
    return merged


def first_available_index(used_indices: Sequence[int], size: int) -> int:
    """Return the first index not present in ``used_indices`` within the given size"""
    for index in range(size):
        if index not in used_indices:
            return index
    raise ValueError("No available indices remain.")


def inorganic_matrix(
    size: int, min_k: float, max_k: float, min_na: float, max_na: float, min_n: float, max_n: float
) -> Matrix:
    """Generate inorganic compound measurements for each variety"""
    if size <= 0:
        raise ValueError("Size must be positive.")
    for label, min_value, max_value in (
        ("K", min_k, max_k),
        ("Na", min_na, max_na),
        ("N", min_n, max_n),
    ):
        if min_value > max_value:
            raise ValueError(f"Minimum {label} value cannot exceed its maximum.")

    return [
        [random.uniform(min_k, max_k), random.uniform(min_na, max_na), random.uniform(min_n, max_n)]
        for _ in range(size)
    ]


def braunschweig(composition: Sequence[float]) -> float:
    """Calculate sugar loss based on inorganic composition"""
    if len(composition) != 3:
        raise ValueError("Composition must contain exactly three values for K, Na, and N.")
    k_value, na_value, n_value = composition
    return (0.12 * (k_value + na_value) + 0.24 * n_value + 0.48) / 100


def adjust_for_inorganic(base_matrix: Matrix, losses_matrix: Matrix) -> Matrix:
    """Reduce sugar values by the Braunschweig calculation where applicable"""
    size = len(base_matrix)
    adjusted = [[0.0 for _ in range(size)] for _ in range(size)]
    for r in range(size):
        for c in range(size):
            val = base_matrix[r][c] - losses_matrix[r][c]
            adjusted[r][c] = max(val, 0.0)

    return adjusted


def _select_by_strategy(
    matrix: Matrix, column: int, permutation: List[int], pick_max: bool
) -> Tuple[float, int]:
    candidate_index = first_available_index(permutation, len(matrix))
    candidate_value = matrix[candidate_index][column]
    for row_index, row in enumerate(matrix):
        value = row[column]
        if row_index in permutation:
            continue
        if (pick_max and value > candidate_value) or (not pick_max and value < candidate_value):
            candidate_value = value
            candidate_index = row_index
    return candidate_value, candidate_index


def greedy_algorithm(matrix: Matrix) -> Tuple[List[float], List[int]]:
    """Select the highest available value in each column without repeating rows"""
    _validate_dimensions(matrix)
    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError("Matrix must be square for the assignment algorithms.")

    result: List[float] = []
    permutation: List[int] = []
    for column in range(size):
        value, index = _select_by_strategy(matrix, column, permutation, pick_max=True)
        cumulative = value if column == 0 else result[column - 1] + value
        result.append(cumulative)
        permutation.append(index)
    return result, permutation


def thrifty_algorithm(matrix: Matrix) -> Tuple[List[float], List[int]]:
    """Select the lowest available value in each column without repeating rows"""
    _validate_dimensions(matrix)
    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError("Matrix must be square for the assignment algorithms.")

    result: List[float] = []
    permutation: List[int] = []
    for column in range(size):
        value, index = _select_by_strategy(matrix, column, permutation, pick_max=False)
        cumulative = value if column == 0 else result[column - 1] + value
        result.append(cumulative)
        permutation.append(index)
    return result, permutation


def greedy_then_thrifty(matrix: Matrix, ripening_period: int) -> Tuple[List[float], List[int]]:
    """Use greedy selection during ripening, then thrifty selection"""
    _validate_dimensions(matrix)
    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError("Matrix must be square for the assignment algorithms.")
    if ripening_period < 0 or ripening_period > size:
        raise ValueError("Ripening period must be between 0 and the matrix size.")

    result: List[float] = []
    permutation: List[int] = []

    for column in range(ripening_period):
        value, index = _select_by_strategy(matrix, column, permutation, pick_max=True)
        cumulative = value if column == 0 else result[column - 1] + value
        result.append(cumulative)
        permutation.append(index)

    for column in range(ripening_period, size):
        value, index = _select_by_strategy(matrix, column, permutation, pick_max=False)
        cumulative = value if column == 0 else result[column - 1] + value
        result.append(cumulative)
        permutation.append(index)

    return result, permutation


def thrifty_then_greedy(matrix: Matrix, ripening_period: int) -> Tuple[List[float], List[int]]:
    """Use thrifty selection during ripening, then greedy selection"""
    _validate_dimensions(matrix)
    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError("Matrix must be square for the assignment algorithms.")
    if ripening_period < 0 or ripening_period > size:
        raise ValueError("Ripening period must be between 0 and the matrix size.")

    result: List[float] = []
    permutation: List[int] = []

    for column in range(ripening_period):
        value, index = _select_by_strategy(matrix, column, permutation, pick_max=False)
        cumulative = value if column == 0 else result[column - 1] + value
        result.append(cumulative)
        permutation.append(index)

    for column in range(ripening_period, size):
        value, index = _select_by_strategy(matrix, column, permutation, pick_max=True)
        cumulative = value if column == 0 else result[column - 1] + value
        result.append(cumulative)
        permutation.append(index)

    return result, permutation


def hungarian_max_algorithm(matrix: Matrix) -> Tuple[List[float], List[int]]:
    """Use the Hungarian method to maximize the assignment cost"""
    _validate_dimensions(matrix)
    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError("Matrix must be square for the assignment algorithms.")

    costs = -np.array(matrix, dtype=float)
    row_indices, col_indices = linear_sum_assignment(costs)

    totals: List[float] = []
    running_total = 0.0
    for row, col in zip(row_indices, col_indices):
        running_total += matrix[row][col]
        totals.append(running_total)

    permutation = [0 for _ in range(size)]
    for _, (row, col) in enumerate(zip(row_indices, col_indices)):
        permutation[col] = row

    return totals, permutation


def calculate_losses_matrix(size: int, inorganic: Matrix, i0_min: float, i0_max: float) -> Matrix:
    """Sugar loss matrix by days"""
    losses = [[0.0 for _ in range(size)] for _ in range(size)]

    for row_idx in range(size):
        K, Na, N = inorganic[row_idx]
        I0 = random.uniform(i0_min, i0_max)

        for day in range(size):
            j = day + 1
            I_curr = I0 * (1.029 ** (j - 7))

            loss_melassa = (
                0.1541 * (K + Na)
                + 0.2159 * N
                + 0.9989 * I_curr
                + 0.1967
            )

            total_loss = 1.1 + loss_melassa
            losses[row_idx][day] = total_loss

    return losses
