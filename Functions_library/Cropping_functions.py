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