from itertools import combinations
import numpy as np
from tda.tda_utils import calc_distance


def distance_statistics(diagrams: list, dim: int = 0, statistic=np.mean) -> float or list:
    """
    Return given statistics for the distance between a list of diagrams
    """
    print('getting pairs')
    pairs = combinations(diagrams, 2)
    print('calculating distance pairs')
    distances = [calc_distance(*pair, dim) for pair in pairs]

    if isinstance(statistic, list):
        return [i(distances) for i in statistic]
    else:
        return statistic(distances)
