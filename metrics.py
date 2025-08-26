 
import torch
import numpy as np
import random
import gudhi
from nc_computing import calculate_nc_and_face_variance_collapse


def compute_epoch_persistence_diagram(model, data_loader, device, sample_size, max_edge_length):
    """
    Computes a single persistence diagram from a sample of the total embedding space.
    This is the rigorous method for analyzing inter-cluster separability.
    """
    model.eval()
    all_features = []

    print("Extracting total embeddings for unified persistence diagram...")
    with torch.no_grad():
        for images, _ in data_loader:  # Labels are not needed for the unified diagram
            images = images.to(device)
            features = model.embed(images)
            all_features.append(features.cpu().numpy())

    all_features = np.concatenate(all_features)

    # Subsample from the entire set of features
    if len(all_features) > sample_size:
        # Use a fixed seed for reproducibility of the sample within an epoch
        np.random.seed(42)
        sample_indices = np.random.choice(len(all_features), sample_size, replace=False)
        feature_sample = all_features[sample_indices]
    else:
        feature_sample = all_features

    print(f"Computing unified diagram on a sample of {feature_sample.shape[0]} points...")
    # Compute one diagram on the total sample
    rips_complex = gudhi.RipsComplex(points=feature_sample, max_edge_length=max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)  # H0, H1
    diag = simplex_tree.persistence()
    h0_diag = [(pair[1][0], pair[1][1]) for pair in diag if pair[0] == 0 and pair[1][1] != float('inf')]

    return h0_diag

def compute_epoch_persistence_diagram_per_class(model, data_loader, device, sample_size, max_edge_length):
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            features = model.embed(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_features = np.concatenate(all_features)
    all_labels = np.concatenate(all_labels)

    unique_labels = np.unique(all_labels)
    persistence_diagrams = {}

    for label in unique_labels:
        label_features = all_features[all_labels == label]
        if len(label_features) > sample_size:
            label_features = label_features[random.sample(range(len(label_features)), sample_size)]

        rips_complex = gudhi.RipsComplex(points=label_features, max_edge_length=max_edge_length)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
        diag = simplex_tree.persistence()
        persistence_diagrams[label] = diag

    return persistence_diagrams

def calculate_nc_metrics(model, data_loader, device, num_classes):
    model.eval()
    return calculate_nc_and_face_variance_collapse(model, data_loader, device_to_use=device, num_classes=num_classes)
