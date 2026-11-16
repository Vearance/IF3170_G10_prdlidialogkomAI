import numpy as np

def linear_kernel(x_i: np.ndarray, x_j: np.ndarray) -> float:
    return x_i @ x_j

def polynomial_kernel(x_i: np.ndarray, x_j: np.ndarray, degree: int, gamma: float, coef0: float) -> float:
    return (gamma * (x_i @ x_j) + coef0) ** degree

def gaussian_kernel(x_i: np.ndarray, x_j: np.ndarray, gamma: float) -> float:
    return np.exp(-gamma * np.sum((x_i - x_j) ** 2))