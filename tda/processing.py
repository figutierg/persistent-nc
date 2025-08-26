"""
This file contains functions to proccess persistence diagrams, specifically,
functions that takes a persistence diagram as input and returns a persistence diagram or
a vectorial repressentation as an output
"""

import numpy as np


def scaled_persistence_diagram(diagram: np.ndarray) -> list:
    # Extract all death times and find the maximum (avoiding infinity)
    if len(diagram)>1:
        max_death_time = max(death for _, (_, death) in diagram if death != float('inf'))

        # Normalize the death times by the maximum death time
        scaled_diagram = [(dim, (birth / max_death_time, death / max_death_time if death != float('inf') else death))
                          for dim, (birth, death) in diagram]

        return scaled_diagram
    else:
        return diagram


def diagram_to_persistence(diagram) -> dict:
    """
    Takes a Gudhi persistence diagram and returns the persistence. as a dictionary with key dimension
    """
    persistence = {}
    for point in diagram:
        dim = point[0]  # Dimension of the homology group
        birth = point[1][0]  # Birth coordinate
        death = point[1][1]  # Death coordinate

        if dim not in persistence:
            persistence[dim] = []

        persistence[dim].append(death - birth)

    return persistence


