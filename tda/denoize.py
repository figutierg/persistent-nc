"""
This file contains functions to filter persistence diagrams, specifically,
functions that takes a persistence diagram as input and returns a filtered persistence diagram
"""


import numpy as np
from tda.tda_utils import get_persistence
from tda.features import persistent_entropy

def persistent_entropy(L: np.ndarray) -> np.ndarray:
    """
    Calculates persistence entropy of a persistence module (Atienza et al 2017) (expressed as persistence of each point)
    L: np.ndarray of persistences
    """

    Sl = np.sum(L)
    Hl = -np.sum(L / Sl * np.log(L / Sl))

    return Hl


def AGR_algorithm(diag: np.ndarray, dim: int = 0) -> int:
    """
    Implementation of Atienza, Gonzalez, Rucco (2017) procedure to  separate topological features from noise
    diag: np.ndarray of a persistence diagram (list)
    returns: The threshold number of most relevant features
    """

    # Step 1

    L = get_persistence(diag, dim=dim)
    # Step 2
    HL = persistent_entropy(L)
    # instantiate L_prime_i
    # Step 3
    L_prime = L
    previous_Hl_prime = HL
    n = L.shape[0]

    for i in range(len(L)):
        #####################3a###########################

        L_i = L[i + 1:]
        S_i = np.sum(L_i)
        k_prime = S_i / np.exp(persistent_entropy(L_i))

        L_prime[i] = k_prime
        H_l_prime = persistent_entropy(L_prime)
        ######################3b##########################
        Hrel = (H_l_prime - previous_Hl_prime) / (np.log(n) - HL)

        previous_Hl_prime = H_l_prime


        if Hrel < (i + 1) / n:

            return i

    print('git gut')


def denoized_diagram(diagram: np.ndarray) -> np.ndarray:
    """
    Separate topological features based on persistent entropy
    input: np.ndarray of a persistence diagram
    returns: filtrated diagram
    """

    diag_inf = np.array([row for row in diagram if np.isinf(row[1][1])], dtype='object')
    diag_np = np.array([row for row in diagram if np.isfinite(row[1][1])], dtype='object')

    dimensions = {feature[0] for feature in diag_np}
    filtered_diags = []

    filtered_diags.extend(diag_inf)

    for dim in dimensions:
        diag = filter_diagram(diag_np, AGR_algorithm(diag_np, dim), dim=dim)
        filtered_diags.extend(diag)

    return filtered_diags


###NAIVE FILTERS###


def filter_diagram(diag: np.ndarray, i: int, dim: int = 0) -> np.ndarray:
    """
    Filter a persistence diagram up to i most persistent features
    """
    dim_filtered_pd = [feature for feature in diag if feature[0] == dim]

    return dim_filtered_pd[:i]
