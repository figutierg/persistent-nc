import torch
import torch.nn as nn
from torchvision import models
from torch.utils.hooks import RemovableHandle
from typing import Optional, List, Dict, Tuple, Union, Any
from tda.complexity import compute_omega
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.hooks import RemovableHandle
from typing import Optional, List, Dict, Tuple, Union, Any
from tda.complexity import compute_omega

class pretrained_model(nn.Module):
    """
    Model wrapper that avoids storing GPU tensors on the object or in persistent lists.
    - Hook capture stores only CPU copies.
    - forward_features returns CPU tensors (and clears GPU intermediates).
    - Complexity hooks convert outputs to CPU.
    - forward() still returns GPU logits (to allow training/backward on GPU).
    """

    def __init__(self, device='cuda', num_classes=10, dropout=0, seed=42,
                 model_arch=models.resnet18, weights=None, model=False, model_kwargs={}):
        super(pretrained_model, self).__init__()
        self.device = torch.device(device if isinstance(device, str) else device)
        self.seed = seed
        self.num_classes = num_classes

        # underlying backbone model (ResNet by default)
        if model:
            self.model = model
        else:
            self.model = model_arch(num_classes=num_classes, weights=weights, **model_kwargs)

        # Hook for embed() features — store only CPU copies
        self._captured_feature: Optional[torch.Tensor] = None  # will always be a CPU tensor or None
        self._hook_handle: Optional[RemovableHandle] = None
        self._target_layer: Optional[nn.Module] = None
        self._register_penultimate_hook()

        # Complexity hooks bookkeeping (do not hold GPU tensors)
        self._complexity_hook_handles: List[RemovableHandle] = []
        self._complexity_target_layers: List[nn.Module] = []
        self._complexity_target_layer_names: List[str] = []
        self._complexity_results: List[Dict[str, Any]] = []

        # Training stats (lists are safe — should only store scalars or CPU data)
        self.pruning_method = 'Dense'
        self.train_accs, self.train_losses = [], []
        self.val_accs, self.val_losses = [], []
        self.test_accs, self.test_losses = [], []
        self.test_accs_top3, self.test_accs_top5 = [], []
        self.best_test, self.best_acc, self.best_loss = 0, 0, 999999
        self.epochs_best_acc, self.trained_epochs = 0, 0
        self.expected_epochs, self.expected_val_acc = None, None

        # Put model parameters on device
        self.to(self.device)

    # ----------------------
    # Utility helpers
    # ----------------------
    @staticmethod
    def _safe_to_cpu(tensor):
        """Return CPU detached copy (or return unchanged if not a tensor)."""
        if torch.is_tensor(tensor):
            # detach then move to CPU, preserving dtype
            return tensor.detach().cpu()
        return tensor

    @staticmethod
    def _clear_refs(*names):
        """Utility to attempt to delete local references to free memory faster."""
        for n in names:
            try:
                del n
            except Exception:
                pass

    # ----------------------
    # Penultimate hook
    # ----------------------
    def _capture_input_hook(self, module, input_args, output):
        """
        Called as a forward hook on the final linear layer.
        We immediately take a detached CPU copy of the input (the penultimate features)
        and store only the CPU copy in self._captured_feature.
        """
        try:
            if isinstance(input_args, tuple) and len(input_args) > 0 and torch.is_tensor(input_args[0]):
                # Move to CPU immediately and detach (no grad, no GPU ref)
                self._captured_feature = input_args[0].detach().cpu()
            elif torch.is_tensor(input_args):
                self._captured_feature = input_args.detach().cpu()
            else:
                self._captured_feature = None
        except Exception as e:
            # Safe fallback: avoid leaving GPU refs
            self._captured_feature = None
            # don't raise; hook should be robust
            print(f"Warning: capture hook error: {e}")

    def _register_penultimate_hook(self):
        """Find last nn.Linear and register forward hook (robust against nested sequential)."""
        target_layer = None
        for module in reversed(list(self.model.children())):
            if isinstance(module, nn.Linear):
                target_layer = module
                break
            if isinstance(module, nn.Sequential):
                for sub_module in reversed(list(module.children())):
                    if isinstance(sub_module, nn.Linear):
                        target_layer = sub_module
                        break
                if target_layer:
                    break

        if target_layer is None:
            # no linear found; do not register hook
            self._target_layer = None
            self._hook_handle = None
            # print warning once
            # print("Warning: final Linear layer not found; penultimate hook not registered.")
            return

        # If a previous handle exists on another layer, remove it
        if self._hook_handle is not None and self._target_layer is not target_layer:
            try:
                self._hook_handle.remove()
            except Exception:
                pass
            self._hook_handle = None

        # register if not already on that layer
        if self._hook_handle is None:
            self._target_layer = target_layer
            self._hook_handle = self._target_layer.register_forward_hook(self._capture_input_hook)

    def remove_hook(self):
        if self._hook_handle is not None:
            try:
                self._hook_handle.remove()
            except Exception:
                pass
            self._hook_handle = None
            self._target_layer = None
            # ensure no CPU feature is retained accidentally
            self._captured_feature = None

    def update_hook(self):
        self.remove_hook()
        self._register_penultimate_hook()

    # ----------------------
    # Forward & embed
    # ----------------------
    def forward(self, x: torch.Tensor):
        """
        Standard forward. We move input to device, run the model, and return the model's output ON THE DEVICE.
        Important: we DO NOT keep any model internals on the object. The hook above will store only CPU copy.
        """
        # ensure input on device (non_blocking if pinned memory)
        x_dev = x.to(self.device, non_blocking=True)
        out = self.model(x_dev)
        # don't assign 'out' to attributes; return it directly (local only)
        return out

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass and return penultimate features as a CPU tensor.
        The hook captures CPU copy; we delete GPU locals ASAP.
        """
        if self._hook_handle is None or self._target_layer is None:
            # attempt to register again (robustness)
            self._register_penultimate_hook()
            if self._hook_handle is None:
                raise RuntimeError("Penultimate hook not registered; cannot embed()")

        # clear any previous CPU capture
        self._captured_feature = None

        initial_mode = self.model.training
        self.model.eval()

        # move input to device and run forward; hook will capture CPU copy
        with torch.no_grad():
            x_dev = x.to(self.device, non_blocking=True)
            try:
                _ = self.model(x_dev)
            finally:
                # explicitly delete any GPU locals referencing large tensors
                try:
                    del _
                except Exception:
                    pass
                try:
                    del x_dev
                except Exception:
                    pass
                # release cached blocks back to allocator (best-effort)
                torch.cuda.empty_cache()

        # restore mode
        self.model.train(initial_mode)

        # now _captured_feature should be a CPU tensor (or None if hook failed)
        if self._captured_feature is None:
            raise RuntimeError("Feature capture failed. Hook did not set captured feature.")
        captured = self._captured_feature
        # clear stored CPU copy on the object to avoid accumulating CPU memory unless user wants it
        self._captured_feature = None
        return captured

    # ----------------------
    # forward_features: return CPU copies and free GPU intermediates
    # ----------------------
    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Compute per-layer summary features for PD/persistence or analysis.
        This function runs the forward on GPU but returns CPU tensors and clears GPU intermediates ASAP.
        """
        if not isinstance(self.model, models.ResNet):
            raise NotImplementedError("forward_features is implemented for torchvision ResNet only.")

        # Move input to device
        x_dev = x.to(self.device, non_blocking=True)

        # Run the standard ResNet forward only up to what we need here,
        # keep only local GPU variables and delete them immediately after converting to CPU.
        with torch.no_grad():
            # conv1..relu..maxpool..layer1..layer2 on device
            conv1_out = self.model.conv1(x_dev)
            bn1_out = self.model.bn1(conv1_out)
            relu_out = self.model.relu(bn1_out)
            maxpool_out = self.model.maxpool(relu_out)

            layer1_out = self.model.layer1(maxpool_out)
            # delete earlier intermediates we won't use
            del conv1_out, bn1_out, relu_out, maxpool_out

            layer2_out = self.model.layer2(layer1_out)
            del layer1_out

            # layer3 -> mean over spatial dims -> move to CPU
            layer3_out = self.model.layer3(layer2_out)  # (B, C3, H3, W3)
            layer3_means = torch.mean(layer3_out, dim=[2, 3])  # still on GPU
            layer3_means_cpu = layer3_means.detach().cpu()
            # clear GPU intermediates
            del layer3_out, layer3_means

            # layer4 -> mean -> CPU
            layer4_out = self.model.layer4(layer2_out)  # NOTE: some ResNet variants use layer3_out as input; we keep layer2_out to be safe
            layer4_means = torch.mean(layer4_out, dim=[2, 3])
            layer4_means_cpu = layer4_means.detach().cpu()
            del layer4_out, layer4_means

            # avgpool and FC path -> flatten to CPU
            avgpool_out = self.model.avgpool(layer2_out)  # (B, C4, 1, 1)
            flattened = torch.flatten(avgpool_out, 1)  # still on GPU
            flattened_cpu = flattened.detach().cpu()
            del avgpool_out, flattened, layer2_out

            # final linear output if needed (but keep on CPU)
            linear_out = self.model.fc(flattened_cpu.to(self.model.fc.weight.device)).detach().cpu() \
                if isinstance(self.model.fc, nn.Linear) else torch.tensor([], dtype=torch.float32)

            # ensure no large GPU locals remain
            torch.cuda.empty_cache()

        # Return CPU tensors only (no GPU references stored anywhere)
        return [layer3_means_cpu, layer4_means_cpu, flattened_cpu, linear_out]

    # ----------------------
    # Complexity hooks (already return CPU results)
    # ----------------------
    def _get_complexity_target_layers(self) -> List[Dict[str, nn.Module]]:
        targets = []
        if not isinstance(self.model, models.ResNet):
            return []

        if hasattr(self.model, 'conv1'):
            targets.append({'name': 'conv1', 'module': self.model.conv1})
        if hasattr(self.model, 'bn1'):
            targets.append({'name': 'bn1', 'module': self.model.bn1})
        if hasattr(self.model, 'relu'):
            targets.append({'name': 'relu', 'module': self.model.relu})

        for layer_num in range(1, 5):
            layer_attr_name = f'layer{layer_num}'
            if hasattr(self.model, layer_attr_name):
                layer_module = getattr(self.model, layer_attr_name)
                if isinstance(layer_module, nn.Sequential):
                    for block_idx, block in enumerate(layer_module.children()):
                        targets.append({'name': f'{layer_attr_name}_block{block_idx}', 'module': block})
        return targets

    def _complexity_capture_hook(self, module: nn.Module, input_args: tuple, output: torch.Tensor):
        """
        Forward hook that immediately computes complexity on CPU and appends a small dict.
        The complexity computation moves the output to CPU inside this hook to avoid storing GPU tensors.
        """
        try:
            cpu_out = output.detach().cpu()
            complexity_value = compute_omega(cpu_out)
            # store only small CPU results (not the big tensor)
            self._complexity_results.append({'layer_name': getattr(module, '__name__', module.__class__.__name__),
                                             'complexity': complexity_value})
            # explicitly delete local
            del cpu_out
        except Exception as e:
            # don't let a complexity error bring down the forward
            print("Complexity hook error:", e)

    def remove_complexity_hooks(self):
        for handle in self._complexity_hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._complexity_hook_handles.clear()
        self._complexity_target_layers.clear()
        self._complexity_target_layer_names.clear()

    def compute_complexity_with_hooks(self, x: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Register hooks that compute complexity and return small CPU results.
        Hooks themselves will move large activation tensors to CPU before computing.
        """
        # remove any existing
        self.remove_complexity_hooks()
        self._complexity_results = []

        target_layers_info = self._get_complexity_target_layers()
        if not target_layers_info:
            return []

        self._complexity_target_layers = [info['module'] for info in target_layers_info]
        self._complexity_target_layer_names = [info['name'] for info in target_layers_info]

        # register forward hooks
        for module in self._complexity_target_layers:
            handle = module.register_forward_hook(self._complexity_capture_hook)
            self._complexity_hook_handles.append(handle)

        # run model; hooks will run and collect CPU-only results
        initial_mode = self.model.training
        self.model.eval()
        with torch.no_grad():
            try:
                # move input to device for forward, hooks will do CPU conversion
                _ = self.model(x.to(self.device, non_blocking=True))
            finally:
                # cleanup
                try:
                    del _
                except Exception:
                    pass
                # remove hooks and restore mode
                self.remove_complexity_hooks()
                self.model.train(initial_mode)
                torch.cuda.empty_cache()

        return self._complexity_results