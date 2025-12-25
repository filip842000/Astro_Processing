############################################################################################################################################################
###                                                                                                                                                      ###
###   This code is NOT meant to be executed directly. It is a module that provides functions for converting images from and into the desired data type   ###
###                                                                                                                                                      ###
############################################################################################################################################################

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

# Cross references to other function libraries
# import Import_functions as Imports

# Specifica and detailed conversions
def float_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    ## Funzione ausiliaria

    Converte un'immagine in virgola mobile (float32 o float64) in uint8 (0-255)
    scalando i valori in base al minimo e massimo dell'immagine.
    """
    image_clipped = np.clip(image, 0.0, 1.0)
    image_8bit = (image_clipped * 255).astype(np.uint8)
    return image_8bit

def float_to_uint16(image: np.ndarray) -> np.ndarray:
    """
    ## Funzione ausiliaria

    Converte un'immagine in virgola mobile (float32 o float64) in uint16 (0-65535)
    scalando i valori in base al minimo e massimo dell'immagine.
    """
    image_clipped = np.clip(image, 0.0, 1.0)
    image_16bit = (image_clipped * 65535).astype(np.uint16)
    return image_16bit

def uint8_to_float32(image: np.ndarray) -> np.ndarray:
    """
    ## Funzione ausiliaria

    Converte un'immagine uint8 (0-255) in float32 (0.0-1.0).
    """
    return (image.astype(np.float32)) / 255.0

def uint16_to_float32(image: np.ndarray) -> np.ndarray:
    """
    ## Funzione ausiliaria

    Converte un'immagine uint16 (0-65535) in float32 (0.0-1.0).
    """
    return (image.astype(np.float32)) / 65535.0

def uint8_to_float64(image: np.ndarray) -> np.ndarray:
    """
    ## Funzione ausiliaria

    Converte un'immagine uint8 (0-255) in float64 (0.0-1.0).
    """
    return (image.astype(np.float64)) / 255.0

def uint16_to_float64(image: np.ndarray) -> np.ndarray:
    """
    ## Funzione ausiliaria

    Converte un'immagine uint16 (0-65535) in float64 (0.0-1.0).
    """
    return (image.astype(np.float64)) / 65535.0

def uint16_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    ## Funzione ausiliaria

    Converte un'immagine uint16 (0-65535) in uint8 (0-255) scalando i valori.
    """
    image_8bit = cv2.convertScaleAbs(image, alpha=(255.0/65535.0))
    return image_8bit

def uint8_to_uint16(image: np.ndarray) -> np.ndarray:
    """
    ## Funzione ausiliaria

    Converte un'immagine uint8 (0-255) in uint16 (0-65535) scalando i valori.
    """
    image_16bit = cv2.convertScaleAbs(image, alpha=(65535.0/255.0))
    return image_16bit

# Generic conversions -- Less accurate but easier to use
def to_8bit(image: np.ndarray) -> np.ndarray:
    """
    ## Funzione ausiliaria

    Scala l'array a uint8 (0-255) per l'analisi.
    """
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) # type: ignore

def to_16bit(image: np.ndarray) -> np.ndarray:
    """
    ## Funzione ausiliaria

    Scala l'array generico a uint16 (0-65535) per l'analisi.
    """
    return cv2.normalize(image, None, 0, 65535, cv2.NORM_MINMAX).astype(np.uint16) #type: ignore