######################################################################################################################################
###                                                                                                                                ###
### This code is NOT meant to be executed directly. It is a module that provides functions to stack images using Drizzle algorithm ###
###                                                                                                                                ###
######################################################################################################################################

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

class SingleChannelStacker:
    def __init__(self, ref_h, ref_w, upscale_factor=3):
        """
        Inizializzatore per un singolo canale (R, G o B).
        """
        self.new_h = int(ref_h * upscale_factor)
        self.new_w = int(ref_w * upscale_factor)
        
        # Accumulatore per il segnale (2D)
        self.accum_data = np.zeros((self.new_h, self.new_w), dtype=np.float32)
        
        # Accumulatore specifico per i pesi di QUESTO canale (2D)
        self.accum_weights = np.zeros((self.new_h, self.new_w), dtype=np.float32)

    def add_channel_frame(self, drizzled_channel, weight_map):
        """
        Aggiunge il singolo piano drizzlato e la sua mappa pesi dedicata.
        """
        self.accum_data += drizzled_channel
        self.accum_weights += weight_map

    def finalize(self):
        """
        Normalizza il segnale dividendolo per i suoi pesi specifici.
        """
        mask = self.accum_weights > 0
        final_channel = np.zeros_like(self.accum_data)
        
        # Divisione pixelwise: solo segnale coerente con il proprio peso
        final_channel[mask] = self.accum_data[mask] / self.accum_weights[mask]
        
        return final_channel