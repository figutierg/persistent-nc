
import gudhi
import torch
from tda.denoize import *
from tda.tda_utils import *
import numpy as np
def calculate_ttp(complexity_values):
    """
    Calculates the Topological Transferability measure (TTP) score (theta)
    from a list of complexity values (Omega) across layers/blocks,
    replicating the methodology described in the paper.

    Args:
      complexity_values: A list or NumPy array of average topological
                         complexity values (Omega), ordered by layer/block.

    Returns:
      The calculated TTP score (theta).
    """
    if not isinstance(complexity_values, (list, np.ndarray)):
        raise TypeError("Input 'complexity_values' must be a list or NumPy array.")

    complexity_values = np.asarray(complexity_values)

    if complexity_values.ndim != 1 or len(complexity_values) < 2:
        raise ValueError("'complexity_values' must be a 1D array with at least 2 values.")

    # --- Parameters based on the paper's implementation ---
    # Degree of the polynomial to fit
    degree = 3
    # Weight vector for calculating TTP score from coefficients (for degree 3)
    # Corresponds to weights for [constant, linear, quadratic, cubic] terms
    weight_vector = np.array([0, 1, 0.5, 0.5])
    # ----------------------------------------------------

    if len(weight_vector) != degree + 1:
        raise ValueError(f"Length of weight_vector ({len(weight_vector)}) must match degree + 1 ({degree + 1}).")

    # 1. Prepare data
    num_layers = len(complexity_values)
    x_layers = np.arange(1, num_layers + 1) # Layers indexed starting from 1

    # Normalize the complexity values (as done in the paper's original code)
    # Avoid division by zero if norm is zero
    norm_y = np.linalg.norm(complexity_values)
    if norm_y == 0:
        # Handle the case of all zero complexity values (e.g., return 0 or raise error)
        # Returning 0 as TTP implies no change or complexity
        return 0.0
    y_normalized = complexity_values / norm_y

    # 2. Fit polynomial using numpy.polyfit
    # polyfit returns coefficients from highest power to lowest (e.g., c3, c2, c1, c0 for degree 3)
    coeffs_high_to_low = np.polyfit(x_layers, y_normalized, degree)

    # 3. Reverse coefficients to match weight_vector order (c0, c1, c2, c3)
    coeffs_low_to_high = coeffs_high_to_low[::-1]

    # Ensure the number of coefficients matches the weight vector
    if len(coeffs_low_to_high) != len(weight_vector):
        # This might happen if polyfit returns fewer coefficients due to data
        # Pad with zeros for lower-order terms if necessary, although polyfit
        # should return degree+1 coefficients. Raising error might be safer.
        raise RuntimeError(f"Mismatch between number of coefficients ({len(coeffs_low_to_high)}) and weight vector size ({len(weight_vector)}).")


    # 4. Calculate TTP score using dot product
    ttp_score = np.dot(coeffs_low_to_high, weight_vector)

    return ttp_score



def compute_omega(x):
    print(f'computing omega for batch of size {x.shape[0]}')

    batch_size = x.shape[0]
    complexity = []

    for sample in x.detach().cpu().numpy():

        sample_chw = sample.T
        cubical_complex = gudhi.CubicalComplex(vertices=sample_chw)
        persistent_diagram = cubical_complex.persistence(homology_coeff_field=2, min_persistence=0)

        complexity = len(denoized_diagram(np.array(persistent_diagram, dtype='object')))

    print('complexity computed')

    return np.array(complexity).sum(axis=0)/batch_size
