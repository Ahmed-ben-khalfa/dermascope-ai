"""
============================================================
DermaScope AI — Prétraitement d'Images Médicales
============================================================
Auteur  : Bilel Kahma
Projet  : Détection de Pathologies Cutanées par Deep Learning
============================================================
Ce module contient la pipeline de prétraitement des images
(DullRazor pour les poils, segmentation SAM et Medical Bokeh).
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Any

from src import config

__all__ = ['remove_hair_dullrazor', 'segment_lesion_hybrid', 'apply_medical_bokeh', 'process_single_image']

def remove_hair_dullrazor(image_path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Algorithme DullRazor Universel pour la suppression des poils et reflets.
    
    Rationnel médical: Les poils sombres et reflets blancs (flash) introduisent
    du bruit empêchant l'analyse correcte des structures dermoscopiques.
    
    Étapes:
    1. Redimensionnement à 512x512 pour invariance d'échelle.
    2. Filtre Black-Hat (croix 17x17) pour extraire les poils sombres.
    3. Filtre Top-Hat pour extraire les poils blancs/flash.
    4. Masque combiné via un OU logique (np.where).
    5. Dilatation rectangulaire (5x5) pour combler les intersections des poils.
    6. Inpainting de Telea (rayon=5) pour restaurer les textures sous-jacentes.
    
    Args:
        image_path (str | Path): Chemin vers l'image.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (Image originale redimensionnée, masque des poils, image inpainted).
    """
    path_str = str(image_path)
    img = cv2.imread(path_str)
    if img is None:
        raise ValueError(f"Impossible de lire l'image : {path_str}")
        
    # 1. Redimensionnement
    img_resized = cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # 2 & 3. Création des noyaux
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (config.DULLRAZOR_KERNEL_SIZE, config.DULLRAZOR_KERNEL_SIZE))
    
    # Filtre Black-Hat
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_cross)
    _, mask_black = cv2.threshold(blackhat, config.DULLRAZOR_BLACKHAT_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    # Filtre Top-Hat
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_cross)
    _, mask_white = cv2.threshold(tophat, config.DULLRAZOR_TOPHAT_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    # 4. Masque combiné
    mask_combined = np.where((mask_black == 255) | (mask_white == 255), 255, 0).astype(np.uint8)
    
    # 5. Dilatation rectangulaire
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (config.DULLRAZOR_DILATION_SIZE, config.DULLRAZOR_DILATION_SIZE))
    mask_dilated = cv2.dilate(mask_combined, kernel_rect, iterations=1)
    
    # 6. Inpainting de Telea
    inpainted = cv2.inpaint(img_resized, mask_dilated, config.DULLRAZOR_INPAINT_RADIUS, cv2.INPAINT_TELEA)
    
    return img_resized, mask_dilated, inpainted


def segment_lesion_hybrid(cleaned_image: np.ndarray, predictor: Any) -> Tuple[np.ndarray, int, int]:
    """
    Segmentation de la lésion via SAM (Segment Anything Model) avec visée hybride.
    
    Rationnel: Trouve le point central d'attention (la lésion) en utilisant les différences
    de couleur avec la peau saine (bords) et en appliquant un biais de gravité centrale.
    
    Étapes:
    1. Conversion en espace couleur LAB.
    2. Échantillonnage de la couleur de la peau sur 4 bords (10px).
    3. Carte d'anomalie de distance Euclidienne.
    4. Biais de gravité centrale gaussienne 2D (sigma=0.5).
    5. Fusion (anomalie x gravité) pour trouver le point de prompt maximal.
    6. Prédiction du masque via SAM.
    
    Args:
        cleaned_image (np.ndarray): Image sans poils (inpainted).
        predictor (Any): Prédicteur SAM initialisé.
        
    Returns:
        Tuple[np.ndarray, int, int]: (Masque binaire de la lésion, Coordonnée X, Coordonnée Y).
    """
    h, w = cleaned_image.shape[:2]
    
    # 1. Conversion LAB
    lab = cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # 2. Échantillonnage de la peau (10px)
    border_pixels = np.vstack([
        lab[0:10, :].reshape(-1, 3),
        lab[-10:, :].reshape(-1, 3),
        lab[:, 0:10].reshape(-1, 3),
        lab[:, -10:].reshape(-1, 3)
    ])
    skin_mean = np.mean(border_pixels, axis=0)
    
    # 3. Carte d'anomalie
    anomaly_map = np.sqrt(np.sum((lab - skin_mean)**2, axis=2))
    # Normalisation
    anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    
    # 4. Biais de gravité centrale
    y_indices, x_indices = np.indices((h, w))
    cy, cx = h / 2, w / 2
    # sigma = 0.5 de la taille de l'image => std = h/2, w/2
    sigma_y, sigma_x = h / 2, w / 2
    gravity_map = np.exp(-(((y_indices - cy)**2) / (2 * sigma_y**2) + ((x_indices - cx)**2) / (2 * sigma_x**2)))
    
    # 5. Fusion
    fused_map = anomaly_map * gravity_map
    target_y, target_x = np.unravel_index(np.argmax(fused_map), fused_map.shape)
    target_y, target_x = int(target_y), int(target_x)
    
    # 6. SAM Predictor
    predictor.set_image(cleaned_image)
    input_point = np.array([[target_x, target_y]])
    input_label = np.array([1]) # Foreground
    
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False
    )
    
    binary_mask = masks[0].astype(np.uint8) * 255
    return binary_mask, target_x, target_y


def apply_medical_bokeh(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Flou d'arrière-plan (Bokeh médical) avec transition douce.
    
    Rationnel: Estompe la peau saine sans créer d'artefacts de bords nets qui 
    pourraient biaiser le réseau de neurones convolutif.
    
    Étapes:
    1. Flou gaussien intense sur le fond (85x85).
    2. Feathering: Adoucissement du masque lui-même (25x25) pour une transition douce.
    3. Alpha blending entre l'image d'origine et le fond flouté selon le masque adouci.
    
    Args:
        image (np.ndarray): L'image contenant la lésion.
        mask (np.ndarray): Le masque binaire (0 ou 255) de la lésion.
        
    Returns:
        np.ndarray: L'image finale avec le bokeh appliqué.
    """
    # 1. Flou d'arrière-plan
    blurred_bg = cv2.GaussianBlur(image, (85, 85), 0)
    
    # 2. Feathering (masque adouci)
    soft_mask = cv2.GaussianBlur(mask.astype(np.float32), (25, 25), 0) / 255.0
    soft_mask = np.expand_dims(soft_mask, axis=-1)
    
    # 3. Alpha blending
    final_image = (image * soft_mask) + (blurred_bg * (1 - soft_mask))
    return final_image.astype(np.uint8)


def process_single_image(image_path: str | Path, predictor: Any) -> np.ndarray:
    """
    Exécute l'intégralité de la pipeline de prétraitement sur une image.
    
    Pipeline: DullRazor -> SAM Hybrid Auto-aim -> Medical Bokeh.
    
    Args:
        image_path (str | Path): Chemin de l'image.
        predictor (Any): Instance du modèle SAM initialisée.
        
    Returns:
        np.ndarray: L'image finale prétraitée.
    """
    # 1. Nettoyage des poils
    _, _, cleaned = remove_hair_dullrazor(image_path)
    
    # 2. Segmentation SAM
    mask, _, _ = segment_lesion_hybrid(cleaned, predictor)
    
    # 3. Application du Bokeh médical
    final = apply_medical_bokeh(cleaned, mask)
    
    return final
