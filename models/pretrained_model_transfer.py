import torch
import torch.nn as nn
import torch.nn.functional as F
# Assuming 'utilities', 'pruning_utilities', and 'n_cubical_for_etas' are available
# from utilities import set_seed
# from pruning_utilities.pruning_utils import get_prunable_layers
# from your_complexity_module import n_cubical_for_etas # Import your complexity function
from torchvision import models
from torch.utils.hooks import RemovableHandle
from typing import Optional, List, Dict, Tuple, Union # Updated imports
from tda.complexity import compute_omega
from typing import Optional, List, Dict, Tuple, Union, Any # Added Any



class pretrained_model(nn.Module):
    def __init__(self, model, last_layer: nn.Module, device='cuda', seed=42):
        """
        Initializes the model wrapper.

        Args:
            model (nn.Module): The pretrained model.
            last_layer (nn.Module): The final linear layer object whose input will be
                                    captured as the embedding.
            device (str): The device to run the model on.
        """
        super(pretrained_model, self).__init__()
        self.device = device
        self.seed = seed
        self.model = model
        # The layer object is stored directly.
        self.last_layer = last_layer

        # --- Initializations for the embedding hook ---
        self._captured_feature: Optional[torch.Tensor] = None
        self._embedding_hook_handle: Optional[RemovableHandle] = None
        self._embedding_target_layer: Optional[nn.Module] = None
        self._register_embedding_hook()

        # --- Training and evaluation trackers (unchanged) ---
        self.pruning_method = 'Dense'
        self.train_accs, self.train_losses = [], []
        self.val_accs, self.val_losses = [], []
        self.test_accs, self.test_losses = [], []
        self.test_accs_top3, self.test_accs_top5 = [], []
        self.best_test, self.best_acc, self.best_loss = 0, 0, 999999
        self.epochs_best_acc, self.trained_epochs = 0, 0
        self.expected_epochs, self.expected_val_acc = None, None

        self.to(self.device)

    def _capture_input_hook(self, module, input_args, output):
        """The hook function that captures the input to the target layer."""
        if isinstance(input_args, tuple) and len(input_args) > 0:
            if isinstance(input_args[0], torch.Tensor):
                self._captured_feature = input_args[0].clone().detach()
            else:
                print(f"Warning: Unexpected input type to hook: {type(input_args[0])}")
                self._captured_feature = None
        elif isinstance(input_args, torch.Tensor):
            self._captured_feature = input_args.clone().detach()
        else:
            print(f"Warning: Unexpected input type to hook: {type(input_args)}")
            self._captured_feature = None

    def _register_embedding_hook(self):
        """
        Registers the forward hook on the provided layer object.
        """
        target_layer = self.last_layer

        if target_layer is None:
            # CHANGED: Updated the error message to not use last_layer_name
            raise ValueError("Error: The provided 'last_layer' is None.")

        # CHANGED: Updated the print statement to not use last_layer_name
        print(f"Registering embedding hook on layer: {target_layer.__class__.__name__}")
        self._embedding_target_layer = target_layer

        if self._embedding_hook_handle is not None:
            self._embedding_hook_handle.remove()

        self._embedding_hook_handle = self._embedding_target_layer.register_forward_hook(self._capture_input_hook)

    def remove_embedding_hook(self):
        """Removes the registered embedding hook."""
        if self._embedding_hook_handle is not None:
            print("Removing embedding hook...")
            self._embedding_hook_handle.remove()
            self._embedding_hook_handle = None
            self._embedding_target_layer = None
            self._captured_feature = None

    # CHANGED: This method is updated to accept a layer object instead of a name.
    def update_embedding_hook(self, new_last_layer: Optional[nn.Module] = None):
        """
        Removes any existing hook and registers a new one.
        Optionally updates the target layer object.
        """
        if new_last_layer is not None:
            self.last_layer = new_last_layer

        print("Updating embedding hook registration...")
        self.remove_embedding_hook()
        self._register_embedding_hook()

    def forward(self, x):
        x = self.model(x.to(self.device))
        return x

    def embed(self, x):
        """
        Extracts the feature embedding for the input tensor 'x'.
        """
        if self._embedding_hook_handle is None or self._embedding_target_layer is None:
            raise RuntimeError("Embedding hook not registered. Cannot capture features.")

        self._captured_feature = None
        initial_mode = self.model.training
        self.model.eval()

        with torch.no_grad():
            try:
                _ = self.forward(x)
            except Exception as e:
                self.model.train(initial_mode)
                raise RuntimeError(f"Error during forward pass within embed: {e}")

        self.model.train(initial_mode)

        if self._captured_feature is None:
            raise RuntimeError("Feature capture failed. The hook did not run or failed to capture the feature.")

        return self._captured_feature