import gudhi
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


def get_embedding_diag(model: nn.Module, data: torch.Tensor, max_dimension: int = 0) -> list:
    """
    Embeds data from a model and returns the persistence diagram of the embedding space
    Model: A torch model with an embed method implemented (a forward up to the embedding layer)
    """

    projection = model.embed(data.to(model.device))
    if model.device == 'cuda':
        projection = projection.cpu()

    return gudhi.RipsComplex(points=projection).\
        create_simplex_tree(max_dimension=max_dimension).persistence()


def get_all_emb_diags(models: list, data: torch.Tensor, device='cpu') -> list:
    """
    Calculate persistence diagrams from a list of models
    models: list of models with the embed method
    data: data to project
    Return: list of persistence diagrams
    """

    return [get_embedding_diag(model.to(device), data) for model in models]



