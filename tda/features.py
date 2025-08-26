import numpy as np
import gudhi as gd
from tda.tda_utils import extract_persistences

def total_persistence_by_dim(persistences: dict, dim=0) -> float:
    return sum(persistence for persistence in persistences[dim] if not np.isinf(persistence))

def persistent_entropy(L: np.ndarray) -> np.ndarray:
    """
    Calculates persistence entropy of a persistence module (Atienza et al 2017) (expressed as persistence of each point)
    L: np.ndarray of persistences
    """

    Sl = np.sum(L)
    Hl = -np.sum(L / Sl * np.log(L / Sl))

    return Hl

def count_persistence_modules(persistence_diagram, dimension):
    """
    Count the number of persistence modules in the given dimension from a GUDHI persistence diagram.

    Parameters:
    - persistence_diagram (list of tuples): The persistence diagram to analyze, where each tuple is (dim, (birth, death))
    - dimension (int): The dimension to filter the persistence modules

    Returns:
    - int: Count of persistence modules in the specified dimension
    """
    # Filter the persistence pairs by the specified dimension
    persistence_pairs = [pair for pair in persistence_diagram if pair[0] == dimension]

    # Return the count of these pairs
    return len(persistence_pairs)

def compute_persistence_features(persistence_diagram, dimension):
    """
    Extracts statistical features from the birth and death times of a persistence diagram for the specified dimension,
    handling infinite death times appropriately.

    Parameters:
    - persistence_diagram (list of tuples): The persistence diagram to analyze, where each tuple is (dim, (birth, death))
    - dimension (int): The dimension to filter the persistence modules

    Returns:
    - dict: Dictionary of statistical features for births, deaths, midpoints, and lifespans
    """
    # Filter the persistence pairs by the specified dimension
    filtered_pairs = [(birth, death) for dim, (birth, death) in persistence_diagram if dim == dimension and death != float('inf')]


    # Separate births and deaths
    births = np.array([pair[0] for pair in filtered_pairs])
    deaths = np.array([pair[1] for pair in filtered_pairs])

    # Replace inf in deaths for calculations
    finite_deaths = np.where(np.isinf(deaths), np.nan, deaths)

    # Calculate midpoints and lifespans
    midpoints = (births + finite_deaths) / 2
    lifespans = finite_deaths - births

    # Helper function to calculate required statistics
    def calc_stats(data):
        finite_data = data[~np.isnan(data)]  # Remove NaN values for statistics
        if finite_data.size == 0:
            return None  # Return None if data is empty after removing NaN
        return {
            'mean': np.nanmean(data),
            'std_dev': np.nanstd(data),
            'median': np.nanmedian(data),
            'iqr': np.percentile(finite_data, 75) - np.percentile(finite_data, 25),
            'full_range': np.ptp(finite_data),
            '10th_percentile': np.percentile(finite_data, 10),
            '25th_percentile': np.percentile(finite_data, 25),
            '75th_percentile': np.percentile(finite_data, 75),
            '90th_percentile': np.percentile(finite_data, 90)
        }

    # Compute statistics for each feature

    stats_births = calc_stats(births)
    stats_deaths = calc_stats(finite_deaths)
    stats_midpoints = calc_stats(midpoints)
    stats_lifespans = calc_stats(lifespans)

    # Return the statistics in a structured dictionary

    return {
        'births': stats_births,
        'deaths': stats_deaths,
        'midpoints': stats_midpoints,
        'lifespans': stats_lifespans
    }

def features_to_list(features_dict):
    """
    Converts the dictionary of statistical features into a list format.

    Parameters:
    - features_dict (dict): Dictionary containing statistical features for each category

    Returns:
    - list: List of all statistical values in a flat structure
    """
    if not features_dict or any(value is None for value in features_dict.values()):
        return []  # Return an empty list if the input dictionary is None or any category is None

    # Initialize an empty list to collect features
    features_list = []

    # Order categories and feature names to ensure consistent list structure
    categories = ['births', 'deaths', 'midpoints', 'lifespans']
    feature_names = ['mean', 'std_dev', 'median', 'iqr', 'full_range', '10th_percentile', '25th_percentile', '75th_percentile', '90th_percentile']

    # Iterate through each category and feature name to extract values
    for category in categories:
        for feature_name in feature_names:
            value = features_dict[category].get(feature_name)
            if value is not None:
                features_list.append(value)
            else:
                features_list.append(np.nan)  # Append NaN if the feature value is missing

    return features_list

def get_statistical_features(diagram, dimension):
    statistical_features = features_to_list(compute_persistence_features(diagram, dimension=dimension))
    statistical_features.append(count_persistence_modules(diagram, dimension=dimension))
    statistical_features.append(persistent_entropy(extract_persistences(diagram, dimension=dimension)))
    return np.array(statistical_features)