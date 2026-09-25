"""
Dermascope AI — Utility Functions
Metadata encoding, Test-Time Augmentation, and Filtered Grad-CAM.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import torchvision.transforms as T

# Categories fitted on training set (deterministic order)
SEX_CATS = ['Female', 'Male', 'unknown']
LOC_CATS = [
    'abdomen', 'back', 'chest', 'ear', 'face', 'foot',
    'genital', 'hand', 'lower extremity', 'neck', 'scalp',
    'trunk', 'unknown', 'upper extremity'
]

MAX_AGE = 85.0
NUM_TABULAR_FEATURES = 1 + len(SEX_CATS) + len(LOC_CATS)  # 18

# Standard ImageNet normalization
VAL_TRANSFORM_512 = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

OPTIMAL_THRESHOLD = 0.4607


def encode_metadata(age: float, sex: str, loc: str) -> np.ndarray:
    """
    Encode a single patient's clinical metadata into a feature vector.

    Args:
        age: Patient age in years (0–100).
        sex: One of 'Female', 'Male', 'unknown'.
        loc: Body localization string.

    Returns:
        np.ndarray of shape (1, NUM_TABULAR_FEATURES), dtype float32.
    """
    age_norm = min(age / MAX_AGE, 1.0)
    sex_arr = [1.0 if sex.lower() == c.lower() else 0.0 for c in SEX_CATS]
    loc_arr = [1.0 if loc.lower() == c.lower() else 0.0 for c in LOC_CATS]
    return np.array([[age_norm] + sex_arr + loc_arr], dtype=np.float32)


def predict_tta(model, image_tensor, meta_tensor):
    """
    Test-Time Augmentation: averages predictions over 5 geometric views.
    """
    augmentations = [
        lambda x: x,
        lambda x: torch.flip(x, dims=[3]),
        lambda x: torch.flip(x, dims=[2]),
        lambda x: torch.rot90(x, k=1, dims=[2, 3]),
        lambda x: torch.rot90(x, k=2, dims=[2, 3]),
    ]
    preds = []
    for aug_fn in augmentations:
        with torch.amp.autocast('cuda'):
            preds.append(torch.sigmoid(model(aug_fn(image_tensor), meta_tensor)))
    return torch.stack(preds).mean(dim=0)


class GradCAM_Sniper:
    """
    Filtered Grad-CAM with center-weighted masking and morphological cleanup.
    Produces SAM-like heatmaps that focus precisely on the lesion.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_act)
        target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out):
        self.activations = out.detach()

    def _save_grad(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, image_tensor, meta_tensor, img_size=512):
        self.model.eval()
        output = self.model(image_tensor, meta_tensor)
        self.model.zero_grad()
        output.backward(torch.ones_like(output))

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(img_size, img_size),
                            mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Filter 1: Normalize to [0, 1]
        if cam.max() - cam.min() > 1e-8:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        # Filter 2: Center-weighted Gaussian mask (kills corner artifacts)
        h, w = cam.shape
        y = np.linspace(-1, 1, h)
        x = np.linspace(-1, 1, w)
        X, Y = np.meshgrid(x, y)
        center_mask = np.exp(-(X ** 2 + Y ** 2) / (2 * 0.6 ** 2))
        cam = cam * center_mask

        if cam.max() > 1e-8:
            cam = cam / cam.max()

        # Filter 3: Aggressive threshold (keep only strong activations)
        cam = np.where(cam < 0.45, 0, cam)

        # Filter 4: Morphological cleanup (remove small dots, fill holes)
        mask = (cam > 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cam = cam * mask

        return cam, torch.sigmoid(output).item()
