#######################################################################################################################################
###                                                                                                                                 ###
### This code is NOT meant to be executed directly. It is a module that provides functions to crop images to the desired dimensions ###
###                                                                                                                                 ###
#######################################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import astroalign as aa
import os
import rawpy
from pathlib import Path
from PIL import Image
import gc
import imageio.v3 as iio
from skimage.restoration import richardson_lucy
from typing import Any, cast

### Ritaglio di un'immagine BGR basato su percentuali per ogni lato
def crop_by_percentage(img, top_pc, bottom_pc, left_pc, right_pc):
    """
    Ritaglia un'immagine BGR basandosi sulle percentuali fornite per ogni lato.
    
    Inputs:
    - img: Array NumPy (H, W, 3).
    - top_pc: % da rimuovere dall'alto (es. 10).
    - bottom_pc: % da rimuovere dal basso (es. 20).
    - left_pc: % da rimuovere da sinistra (es. 15).
    - right_pc: % da rimuovere da destra (es. 35).
    
    Output:
    - cropped_img: Sottomatrice dell'immagine originale.
    """
    h, w = img.shape[:2]

    # Calcolo degli indici di taglio (pixel)
    # Usiamo la divisione intera // per ottenere indici validi per la matrice
    start_row = int(h * (top_pc / 100.0))
    end_row   = int(h * (1 - (bottom_pc / 100.0)))
    
    start_col = int(w * (left_pc / 100.0))
    end_col   = int(w * (1 - (right_pc / 100.0)))

    # Lo slicing di NumPy: img[altezza_inizio : altezza_fine, larghezza_inizio : larghezza_fine]
    # Manteniamo tutti i canali (:)
    cropped_img = img[start_row:end_row, start_col:end_col, :]

    return cropped_img

def normalizer(image: np.ndarray, reference_image: np.ndarray) -> np.ndarray:
    """
    Normalizza la luminosità di un'immagine rispetto a un'immagine di riferimento.
    Normalizziamo sul valore medio (o mediano) per ignorare i pixel estremi (hot/dead pixels).
    """

    # 1. Scegli l'immagine di riferimento (usiamo la prima come baseline)
    mask = np.any(reference_image > 0.7, axis=2)
    celestial_pixels = reference_image[mask]
    # 2. Calcola il fattore di riferimento (es. la mediana di tutti i pixel)
    # np.median è più robusto rispetto a np.mean o np.max per ignorare gli estremi.
    reference_median = np.median(celestial_pixels, axis=(0, 1))  # Mediana per canale (R, G, B)

    # Calcola la mediana dell'immagine corrente
    mask = np.any(image > 0.7, axis=2)
    current_median = np.median(image[mask], axis=(0, 1))
    # Calcola il fattore di scaling necessario
    scale_factor = reference_median / current_median
    # Applica il fattore di scaling
    normalized_img = image * scale_factor
        
    print(f"✅ Normalizzazione completata")
    return normalized_img

def alignment_prep(image: np.ndarray):
    # Prepara l'immagine per l'allineamento (median denoising, normalizzazione, thresholding, gaussian blur, ecc.)
    # Implementazione specifica dipende dal contesto

    # Median Denoising
    image = cv2.medianBlur(image, 3)  # Esempio: filtro mediano per ridurre il rumore
    
    # Normalizzazione manuale per canale (per evitare problemi di tipo con cv2.normalize)
    # Trova min/max per canale e scala a [0, 1]
    image = (image - image.min(axis=(0, 1), keepdims=True)) / (image.max(axis=(0, 1), keepdims=True) - image.min(axis=(0, 1), keepdims=True))

    # Thresholding per evidenziare le stelle
    _, image = cv2.threshold(image, 0.15, 1.0, cv2.THRESH_TOZERO)

    # Gaussian Blur per ridurre il rumore
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    return image

def remove_green_cast(img):
    # Per ogni pixel, se il verde è maggiore del massimo tra rosso e blu,
    # lo limitiamo al valore massimo degli altri due.
    out = img.copy()
    avg_rb = (out[:,:,0] + out[:,:,2]) / 2
    mask = out[:,:,1] > avg_rb
    out[:,:,1][mask] = avg_rb[mask]
    return out