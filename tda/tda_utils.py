from gudhi.wasserstein import wasserstein_distance
import numpy as np
import pickle
import os
########################################################################################
#####################Persistence Diagrams processing tools##############################
########################################################################################

def get_persistence(diag: np.ndarray, dim: int = 0) -> np.array:
    """
    Calculates persistence for a persistence module from a diagram
    diag: np.ndarray of a persistence diagram (list)
    dim: dimension of a diagram
    """

    diag = diag[diag[:, 0] == dim][:, 1]



    return (np.array([point[1] for point in diag if point[1] != np.inf]) -
            np.array([point[0] for point in diag if point[1] != np.inf]))


def extract_persistences(persistence_diagram, dimension):
    """
    Extracts persistences (lifespans) for a specified dimension from a GUDHI persistence diagram, ignoring infinite death times.

    Parameters:
    - persistence_diagram (list of tuples): The persistence diagram to analyze, where each tuple is (dim, (birth, death))
    - dimension (int): The dimension to filter the persistence modules

    Returns:
    - np.array: Array of finite persistences for the specified dimension
    """
    # Filter the persistence pairs by the specified dimension and calculate their persistences
    # Only include intervals where the death time is not infinite
    persistences = np.array([death - birth for dim, (birth, death) in persistence_diagram
                             if dim == dimension and np.isfinite(death)])

    return persistences

def transform_to_intervals(persistence_diagram, dimension):
    """
    Transforms a GUDHI persistence diagram to the format used by persistence_intervals_in_dimension(),
    filtering for a specific dimension and returning a 2D numpy array of intervals.

    Parameters:
    - persistence_diagram (list of tuples): The persistence diagram to analyze, where each tuple is (dim, (birth, death))
    - dimension (int): The dimension to filter the persistence modules

    Returns:
    - numpy array: 2D numpy array of (birth, death) intervals for the specified dimension
    """
    # Filter the persistence pairs by the specified dimension
    intervals = np.array([(birth, death) for dim, (birth, death) in persistence_diagram if dim == dimension and not np.isinf(death)])

    return intervals


def calc_distance(diag1: list, diag2: list, dim: int = 0,order: int = 1.0, internal_p = 2) -> float:
    """
    Returns Wasserstein distance between two diagrams (as given by gudhi)
    group_dim: persistence module dim
    order: order for distance calculation
    """

    dim_filtered_pd_1 = [feature for feature in diag1 if feature[0] == dim]
    dim_filtered_pd_2 = [feature for feature in diag2 if feature[0] == dim]
    diag1 = np.array([pt[1] for pt in dim_filtered_pd_1])
    diag2 = np.array([pt[1] for pt in dim_filtered_pd_2])

    return wasserstein_distance(diag1, diag2, order=order, internal_p = internal_p)


########################################################################################
################################Diagramsloaders#########################################
########################################################################################


def load_diagrams(diagrams_directory: str) -> list:
    diagrams_route = [f for f in os.listdir(diagrams_directory) if f.endswith('.pkl')]

    # Sort files based on the float value extracted from the filename
    sorted_diagrams_files = sorted(diagrams_route, key=lambda x: float(x.split('_')[1].replace('.pkl', '')))

    print(sorted_diagrams_files)  # Print sorted filenames to verify order
    diagrams = []
    for diagram in sorted_diagrams_files:
        with open(os.path.join(diagrams_directory, diagram), 'rb') as f:
            diagrams.append(pickle.load(f))

    return diagrams
########################################################################################
###########################Diagrams generation tools####################################
########################################################################################

def n_cubical(x: np.ndarray) -> int:
    pass

