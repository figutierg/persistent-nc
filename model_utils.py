import torch
import torch.nn as nn
import torchvision.models as models
from models.pretrained2 import pretrained_model

import torch
import torch.nn as nn
import torchvision.models as models
from models.pretrained2 import pretrained_model


def _replace_classifier(model, num_classes):
    """
    Finds the final linear layer of a model and replaces it with a new one.
    This works for most common architectures (ResNet, VGG, DenseNet, EfficientNet, etc.).

    Returns:
        The number of input features to the new classifier.
    """
    # Common attribute names for classifiers
    classifier_names = ['fc', 'classifier', 'heads']

    for name in classifier_names:
        if hasattr(model, name):
            classifier = getattr(model, name)

            # Case 1: The classifier is a single Linear layer (like ResNet's fc)
            if isinstance(classifier, nn.Linear):
                in_features = classifier.in_features
                setattr(model, name, nn.Linear(in_features, num_classes))
                return in_features

            # Case 2: The classifier is a Sequential block (like VGG, EfficientNet)
            if isinstance(classifier, nn.Sequential):
                # Iterate backwards to find the last Linear layer to replace
                for i in range(len(classifier) - 1, -1, -1):
                    if isinstance(classifier[i], nn.Linear):
                        in_features = classifier[i].in_features
                        classifier[i] = nn.Linear(in_features, num_classes)
                        return in_features

            # Case 3: Special case for Vision Transformer (heads.head)
            if name == 'heads' and hasattr(classifier, 'head') and isinstance(classifier.head, nn.Linear):
                in_features = classifier.head.in_features
                classifier.head = nn.Linear(in_features, num_classes)
                return in_features

    raise TypeError(
        f"Could not automatically find and replace the classifier for model {model.__class__.__name__}. Please add a special case in model_utils.py")


def create_model(device, num_classes, architecture_name, model_adapter):
    """
    Model factory that creates a model and adapts it for the specified architecture and dataset.
    """
    if not hasattr(models, architecture_name):
        raise ValueError(f"Architecture '{architecture_name}' not found in torchvision.models")

    model_constructor = getattr(models, architecture_name)

    # --- Instantiate the base model ---
    # SqueezeNet and some others require num_classes at construction time.
    # Our generic method handles others by replacing the last layer.
    if 'squeezenet' in architecture_name:
        base_model = model_constructor(num_classes=num_classes)
    else:
        base_model = model_constructor(weights=None)
        _replace_classifier(base_model, num_classes)

    # --- Handle special cases ---
    # Inception v3 requires a special modification for its auxiliary head.
    if architecture_name == 'inception_v3':
        aux_in_features = base_model.AuxLogits.fc.in_features
        base_model.AuxLogits.fc = nn.Linear(aux_in_features, num_classes)

    # --- Apply input adaptations based on the dataset ---
    if model_adapter == "small_3_channel":
        # Modify the first layer for small 32x32 3-channel images
        if hasattr(base_model, 'conv1'):  # ResNet-style
            base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if hasattr(base_model, 'features') and isinstance(base_model.features, nn.Sequential):  # DenseNet-style
            base_model.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if hasattr(base_model, 'maxpool'):
            base_model.maxpool = nn.Identity()

    elif model_adapter == "small_1_channel":
        # Modify the first layer for small 1-channel images (e.g., MNIST)
        if hasattr(base_model, 'conv1'):
            base_model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if hasattr(base_model, 'features') and isinstance(base_model.features, nn.Sequential):
            base_model.features.conv0 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if hasattr(base_model, 'maxpool'):
            base_model.maxpool = nn.Identity()

    # --- Wrap the adapted model in our custom wrapper ---
    model_wrapper = pretrained_model(
        device=device,
        num_classes=num_classes,
        model=base_model  # Pass the fully configured base_model
    )
    model_wrapper.to(device)

    return model_wrapper


def prepare_for_transfer(model_wrapper, target_classes, device, freeze_layers=True):
    num_ftrs = model_wrapper.model.fc.in_features
    model_wrapper.model.fc = nn.Linear(num_ftrs, target_classes)
    model_wrapper.update_hook()
    model_wrapper.to(device)

    if freeze_layers:
        for name, param in model_wrapper.named_parameters():
            if 'model.fc' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    return model_wrapper


def load_checkpoint(model, filepath):
    model.load_state_dict(torch.load(filepath))
    return model

def prepare_for_transfer(model_wrapper, target_classes, device, freeze_layers=True):
    """
    Prepares a pretrained_model wrapper for transfer learning.

    1. Replaces the final fully-connected layer.
    2. Updates the penultimate layer hook to point to the new layer.
    3. Moves the model to the specified device.
    4. Optionally freezes all layers except the new classifier.
    """
    # Find the last linear layer to determine the number of input features
    # This logic assumes the final classifier is named 'fc' as is common
    if not hasattr(model_wrapper.model, 'fc') or not isinstance(model_wrapper.model.fc, nn.Linear):
        raise TypeError("Could not find a final 'fc' Linear layer to replace.")

    num_ftrs = model_wrapper.model.fc.in_features
    model_wrapper.model.fc = nn.Linear(num_ftrs, target_classes)

    # CRITICAL: The old hook is now invalid. We must update it.
    model_wrapper.update_hook()
    model_wrapper.to(device)

    # Freeze layers if requested
    if freeze_layers:
        for name, param in model_wrapper.named_parameters():
            if 'model.fc' not in name: # Only train the new final layer
                param.requires_grad = False
            else:
                param.requires_grad = True # Ensure the new layer is trainable
    else:
        # Ensure all parameters are trainable if not freezing
        for param in model_wrapper.parameters():
            param.requires_grad = True

    return model_wrapper
