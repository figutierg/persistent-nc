import gudhi as gd
import numpy as np
from scipy import stats
import warnings


def _calculate_l0(points: np.ndarray) -> float:
    """
    Internal helper to calculate L^0 for a given point cloud.
    L^0 is the sum of finite 0-dim persistence interval lengths.
    """
    # Create a Rips complex and compute persistence
    rips_complex = gd.RipsComplex(points=points)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)
    persistence_pairs = simplex_tree.persistence()

    # Manually filter for 0-dim intervals and remove the infinite one
    h0_intervals = [p[1] for p in persistence_pairs if p[0] == 0]
    total_length = sum(death - birth for birth, death in h0_intervals if death != float('inf'))

    return total_length

def compute_ph_dimension(
    embeddings: np.ndarray,
    n_start: int = 200,
    n_end: int = 5000,
    step: int = 200,
    return_slope: bool = False
):
    """
    Computes the 0-dimensional Persistent Homology (PH) dimension of a point cloud.

    This function subsamples the provided point cloud at various sizes (n)
    and analyzes how the total length of the 0-dimensional persistence intervals (L^0)
    scales with n. This scaling relationship reveals the intrinsic dimension.

    Args:
        embeddings (np.ndarray): The input point cloud, e.g., neural network
                                 embeddings, as a NumPy array of shape (num_points, dim).
        n_start (int): The starting number of points to sample.
        n_end (int): The maximum number of points to sample.
        step (int): The step size for increasing the number of samples.
        return_slope (bool): If True, returns a tuple (dimension, slope).

    Returns:
        float or tuple: The estimated PH dimension, or (dimension, slope) if
                        return_slope is True. Returns np.nan on failure.
    """
    num_points = embeddings.shape[0]

    # --- Input Validation ---
    if num_points < n_end:
        print(f"Warning: n_end ({n_end}) is larger than the number of available points ({num_points}).")
        n_end = num_points
    if n_start >= n_end:
        print("Error: n_start must be smaller than n_end.")
        return np.nan

    # --- Main Simulation Loop ---
    n_values = np.arange(n_start, n_end + 1, step)
    l0_values = []

    print(f"Analyzing PH dimension for sample sizes from {n_start} to {n_end}...")

    for n in n_values:
        # Randomly subsample n points from the embeddings
        indices = np.random.choice(num_points, n, replace=False)
        sample = embeddings[indices]

        # Calculate L^0 for the sample
        l0 = _calculate_l0(sample)
        l0_values.append(l0)

    # --- Analysis ---
    # Filter out non-positive L0 values to prevent log errors
    n_filtered = [n for n, l0 in zip(n_values, l0_values) if l0 > 0]
    l0_filtered = [l0 for l0 in l0_values if l0 > 0]

    if len(n_filtered) < 2:
        print("Error: Not enough valid data points for regression analysis.")
        return np.nan

    # Perform log-log linear regression
    n_log = np.log10(n_filtered)
    l0_log = np.log10(l0_filtered)
    slope, intercept, _, _, _ = stats.linregress(n_log, l0_log)

    # Calculate dimension from slope: slope = (d-1)/d => d = 1/(1-slope)
    if not np.isnan(slope) and slope < 1:
        dimension_d = 1 / (1 - slope)
    else:
        dimension_d = float('inf')

    if return_slope:
        return dimension_d, slope
    return dimension_d
