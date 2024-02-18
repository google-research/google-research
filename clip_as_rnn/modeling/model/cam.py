import cv2
import numpy as np
import torch
from typing import Callable, List, Tuple, Union


def scale_cam_image(cam, target_size=None) -> np.ndarray:
    result = []
    for img in cam:
        img = img - np.min(img)
        img = img / (1e-15 + np.max(img))
        if target_size is not None:
            img = cv2.resize(img, target_size)
        result.append(img)
    result = np.float32(result)

    return result


class ActivationsAndGradients:
    """Class for extracting activations and
    registering gradients from targetted intermediate layers"""

    def __init__(self, model, target_layers, reshape_transform, stride=16):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        self.stride = stride
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation)
            )
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient)
            )

    def save_activation(self, module, input, output):
        """Saves activations from targetted layer"""
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(
                activation, self.height, self.width
            )
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad, self.height, self.width)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x, H, W):
        self.height = H // self.stride
        self.width = W // self.stride
        self.gradients = []
        self.activations = []
        if isinstance(x, tuple) or isinstance(x, list):
            return self.model.forward_last_layer(x[0], x[1])
        else:
            return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()


class CAM:
    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: List[torch.nn.Module],
        use_cuda: bool = False,
        reshape_transform: Callable = None,
        compute_input_gradient: bool = False,
        stride: int = 16,
    ) -> None:
        self.model = model.eval()
        self.target_layers = target_layers
        self.cuda = use_cuda
        self.model = model.cuda() if self.cuda else self.model
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform, stride=stride
        )

    def get_cam(
        self, activations: torch.Tensor, grads: torch.Tensor
    ) -> np.ndarray:
        weights = np.mean(grads, axis=(2, 3))
        weighted_activations = weights[:, :, None, None] * activations
        cam = weighted_activations.sum(axis=1)
        return cam

    def forward(
        self,
        input_tensor: Union[torch.Tensor, List[torch.Tensor]],
        targets: List[torch.nn.Module],
        target_size,
    ) -> np.ndarray:
        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(
                input_tensor, requires_grad=True
            )

        W, H = self.get_target_width_height(input_tensor)
        outputs = self.activations_and_grads(input_tensor, H, W)

        self.model.zero_grad()
        if isinstance(input_tensor, (tuple, list)):
            loss = sum(
                [target(output[0]) for target, output in zip(targets, outputs)]
            )
        else:
            loss = sum(
                [target(output) for target, output in zip(targets, outputs)]
            )
        loss.backward(retain_graph=True)
        cam_per_layer = self.compute_cam_per_layer(target_size)
        if isinstance(input_tensor, (tuple, list)):
            return (
                self.aggregate_multi_layers(cam_per_layer),
                outputs[0],
                outputs[1],
            )
        else:
            return self.aggregate_multi_layers(cam_per_layer), outputs

    def get_target_width_height(
        self, input_tensor: torch.Tensor
    ) -> Tuple[int, int]:
        if isinstance(input_tensor, (tuple, list)):
            width, height = input_tensor[-1], input_tensor[-2]
        return width, height

    def compute_cam_per_layer(self, target_size) -> np.ndarray:
        activations_list = [
            a.cpu().data.numpy()
            for a in self.activations_and_grads.activations
        ]
        grads_list = [
            g.cpu().data.numpy() for g in self.activations_and_grads.gradients
        ]

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam(layer_activations, layer_grads)
            cam = np.maximum(cam, 0).astype(np.float32)  # float16->32
            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(
        self, cam_per_target_layer: np.ndarray
    ) -> np.ndarray:
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return scale_cam_image(result)

    def __call__(
        self,
        input_tensor: torch.Tensor,
        targets: List[torch.nn.Module] = None,
        target_size=None,
    ) -> np.ndarray:
        return self.forward(input_tensor, targets, target_size)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. "
                f"Message: {exc_value}"
            )
            return True
