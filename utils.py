"""
Dermascope AI — Utility Functions
Metadata encoding and Test-Time Augmentation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import torchvision.transforms as T

# Categories fitted on training set (deterministic order)
SEX_CATS = ['Female', 'Male', 'unknown']
LOC_CATS = [
    'abdomen', 'acral', 'back', 'chest', 'ear', 'face', 'foot',
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

