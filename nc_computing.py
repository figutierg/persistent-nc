import torch
import torch.nn as nn
from neural_collapse.accumulate import (MeanAccumulator, VarNormAccumulator, CovarAccumulator)
from neural_collapse.kernels import kernel_stats, log_kernel
from neural_collapse.measure import (covariance_ratio, simplex_etf_error, variability_cdnv)
import math  # For math.exp


def calculate_nc_and_face_variance_collapse(
    model_instance,
    train_loader_for_nc,
    device_to_use,
    num_classes,
    epsilon: float = 1e-7
):
    """
    Calculates NC1, NC2 metrics and the unnormalized FaCe Variance Collapse term (Cm).
    Avoids GPU memory build-up by ensuring features are stored/processed on CPU
    and only moved to GPU temporarily for operations that require it.
    """
    if not hasattr(model_instance, '_target_layer') or model_instance._target_layer is None:
        if hasattr(model_instance, '_register_penultimate_hook'):
            print("Attempting to register penultimate hook...")
            model_instance._register_penultimate_hook()
        if not hasattr(model_instance, '_target_layer') or model_instance._target_layer is None:
            raise RuntimeError("Target layer for hook not found.")

    if not isinstance(model_instance._target_layer, nn.Linear):
        raise TypeError(
            f"Expected _target_layer to be nn.Linear, got {type(model_instance._target_layer)}."
        )

    feature_dim = model_instance._target_layer.in_features
    print(f"Using inferred feature_dim: {feature_dim} for NC calculations.")

    results = {
        "nc1_pinv": float('nan'), "nc1_svd": float('nan'),
        "nc1_quot": float('nan'), "nc1_cdnv": float('nan'),
        "nc2_etf_err": float('nan'), "nc2g_dist": float('nan'),
        "nc2g_log": float('nan'),
        "face_variance_collapse_Cm": float('nan')
    }

    # Accumulators stay on CPU
    mean_accum = MeanAccumulator(num_classes, feature_dim, device="cpu")
    var_norms_accum = VarNormAccumulator(num_classes, feature_dim, device="cpu")
    covar_accum_sigma_w = CovarAccumulator(num_classes, feature_dim, device="cpu")

    # --- Pass 1: Means ---
    print("Pass 1: Calculating means...")
    for images, labels in train_loader_for_nc:
        labels = labels.cpu()
        with torch.no_grad():
            features_batch = model_instance.embed(images)  # already on CPU
        if features_batch is None:
            continue
        mean_accum.accumulate(features_batch, labels)

    means, mG = mean_accum.compute()
    if means is None or mG is None or any(t.numel() == 0 for t in [means, mG]):
        print("Warning: Mean computation failed or empty.")
        return results

    # --- NC2 (requires GPU ops, but only temporarily) ---
    means_dev = means.to(device_to_use)
    mG_dev = mG.to(device_to_use)
    try:
        results["nc2_etf_err"] = simplex_etf_error(means_dev, mG_dev)
        _, dist_var = kernel_stats(means_dev, mG_dev, tile_size=64)
        results["nc2g_dist"] = dist_var
        _, log_var = kernel_stats(means_dev, mG_dev, kernel=log_kernel, tile_size=64)
        results["nc2g_log"] = log_var
    except Exception as e:
        print(f"Warning: NC2 calculations failed: {e}")
    del means_dev, mG_dev  # free GPU mem

    # --- Pass 2: Var norms & Sigma_W ---
    print("Pass 2: Calculating Sigma_W and var_norms...")
    for images, labels in train_loader_for_nc:
        labels = labels.cpu()
        with torch.no_grad():
            features_batch = model_instance.embed(images)  # already on CPU
        if features_batch is None:
            continue
        var_norms_accum.accumulate(features_batch, labels, means)
        covar_accum_sigma_w.accumulate(features_batch, labels, means)

    var_norms, _ = var_norms_accum.compute()
    covar_within_sigma_w = covar_accum_sigma_w.compute()

    # --- NC1 & Cm ---
    if covar_within_sigma_w is not None and var_norms is not None:
        try:
            covar_dev = covar_within_sigma_w.to(device_to_use)
            var_norms_dev = var_norms.to(device_to_use)
            means_dev = means.to(device_to_use)
            mG_dev = mG.to(device_to_use)

            results["nc1_pinv"] = covariance_ratio(covar_dev, means_dev, mG_dev, metric="pinv")
            results["nc1_svd"] = covariance_ratio(covar_dev, means_dev, mG_dev, metric="svd")
            results["nc1_quot"] = covariance_ratio(covar_dev, means_dev, mG_dev, metric="quotient")
            results["nc1_cdnv"] = variability_cdnv(var_norms_dev, means_dev, tile_size=64)

            nc1_pinv_val = results["nc1_pinv"]
            if isinstance(nc1_pinv_val, (int, float)) and not math.isnan(nc1_pinv_val):
                results["face_variance_collapse_Cm"] = -nc1_pinv_val
        except Exception as e:
            print(f"Warning: NC1/Cm calculations failed: {e}")
        finally:
            del covar_dev, var_norms_dev, means_dev, mG_dev  # free GPU mem

    return results
