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
    def __init__(self, ref_h, ref_w, upscale_factor):
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
    
class DrizzleStacker:
    def __init__(self, ref_h, ref_w, upscale_factor=3):
        """
        Inizializza gli accumulatori per lo stacking.
        """
        self.new_h = int(ref_h * upscale_factor)
        self.new_w = int(ref_w * upscale_factor)
        
        # Accumulatore Immagine (3 canali BGR)
        # Un'immagine 6000x4000 float32 occupa circa 288MB
        self.accum_image = np.zeros((self.new_h, self.new_w, 3), dtype=np.float32)
        
        # Accumulatore Pesi (1 canale)
        # Occupa circa 96MB
        self.accum_weights = np.zeros((self.new_h, self.new_w), dtype=np.float32)

    def add_frame(self, drizzled_img, weight_map):
        """
        Aggiunge un frame drizzle all'accumulatore.
        """
        # Somma pixelwise dei valori colore
        self.accum_image += drizzled_img
        
        # Somma pixelwise dei pesi
        self.accum_weights += weight_map

    def get_final_image(self):
        """
        Esegue la divisione finale e restituisce lo stack normalizzato.
        """
        # Creiamo una maschera per evitare la divisione per zero
        # (zone dove nessun frame è mai 'caduto')
        mask = self.accum_weights > 0
        
        final_img = np.zeros_like(self.accum_image)
        
        # Dividiamo ogni canale per la mappa dei pesi
        # Usiamo [:, :, None] per trasmettere (broadcast) il peso 2D sui 3 canali RGB
        for i in range(3):
            final_img[mask, i] = self.accum_image[mask, i] / self.accum_weights[mask]
            
        return final_img

def drizzle_core(img, M, upscale_factor=3, pixfrac=1.0):
    """
    Esegue il Drizzle raffinato con gestione del PixFrac tramite mappatura inversa.
    
    Inputs:
    - img: Immagine BGR (float32).
    - M: Matrice di allineamento 3x3 (float64).
    - upscale_factor: Fattore di ingrandimento (default 3).
    - pixfrac: Dimensione relativa della 'goccia' (0.1 - 1.0).
    
    Outputs:
    - drizzled_img: Frame proiettato sulla griglia densa.
    - weight_map: Mappa dei pesi basata sull'area del drop.
    """
    h, w = img.shape[:2]
    new_h, new_w = int(h * upscale_factor), int(w * upscale_factor)
    
    # 1. Creazione della griglia di destinazione (High-Res)
    # Generiamo tutte le coordinate (x, y) della griglia 3x
    y_dst, x_dst = np.indices((new_h, new_w), dtype=np.float32)
    
    # 2. Inversione della trasformazione
    # Vogliamo sapere per ogni pixel della griglia finale, dove "cade" nell'originale.
    # Scaliamo M per tenere conto dell'upscale
    S = np.array([[upscale_factor, 0, 0], [0, upscale_factor, 0], [0, 0, 1]])
    M_drizzle = S @ M
    M_inv = np.linalg.inv(M_drizzle)
    
    # Trasformazione delle coordinate tramite la matrice inversa
    # Calcolo coordinate omogenee: x_src = (M_inv * x_dst) / w_src
    denominator = M_inv[2,0] * x_dst + M_inv[2,1] * y_dst + M_inv[2,2]
    x_src = (M_inv[0,0] * x_dst + M_inv[0,1] * y_dst + M_inv[0,2]) / denominator
    y_src = (M_inv[1,0] * x_dst + M_inv[1,1] * y_dst + M_inv[1,2]) / denominator

    # 3. Logica Raffinata del PixFrac
    # Troviamo la distanza tra la coordinata calcolata e il centro del pixel sorgente
    # dx e dy rappresentano quanto siamo lontani dal centro perfetto del pixel originale
    dx = np.abs(x_src - np.round(x_src))
    dy = np.abs(y_src - np.round(y_src))
    
    # Un punto della griglia finale 'riceve' luce solo se cade dentro il perimetro del PixFrac.
    # Se pixfrac=1.0, la soglia è 0.5 (copre tutto il pixel).
    # Se pixfrac=0.5, la soglia è 0.25 (il drop è un quadratino centrale).
    limit = pixfrac / 2.0
    inside_drop = (dx < limit) & (dy < limit)
    
    # 4. Ricostruzione dell'Immagine
    # Preleviamo il valore del pixel originale usando l'interpolazione Nearest
    # perché il Drizzle non deve creare nuovi valori, deve solo 'spostarli'.
    drizzled_img = cv2.remap(img, x_src, y_src, cv2.INTER_NEAREST)
    
    # Applichiamo il ritaglio del PixFrac: azzeriamo tutto ciò che è fuori dal drop
    drizzled_img[~inside_drop] = 0
    
    # 5. Generazione Mappa Pesi
    # Ogni pixel che ha ricevuto dati pesa pixfrac^2 (area del drop)
    weight_map = inside_drop.astype(np.float32) * (pixfrac**2)
    
    return drizzled_img, weight_map

def apply_drizzle_step(img, M, upscale_factor=3, pixfrac=1.0):
    """
    Esegue un passo di Drizzle su un singolo frame.
    
    Inputs:
    - img: Immagine BGR float32.
    - M: Matrice 3x3 (allineamento).
    - upscale_factor: Fattore di ingrandimento (es. 3).
    - pixfrac: Dimensione del 'drop' (0.1 a 1.0). Default 1.0.
    
    Outputs:
    - drizzled_frame: Frame ingrandito e trasformato.
    - weight_map: Mappa dei pesi corrispondente.
    """
    h, w = img.shape[:2]
    new_h, new_w = int(h * upscale_factor), int(w * upscale_factor)
    
    # 1. Matrice di Scalatura per l'upscale
    S = np.array([
        [upscale_factor, 0, 0],
        [0, upscale_factor, 0],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Matrice finale: sposta e poi ingrandisce
    M_drizzle = np.matmul(S, M)
    
    # 2. Logica del PixFrac
    # Se pixfrac < 1.0, il drop è più piccolo del pixel di output. 
    # Per simulare questo con le funzioni di OpenCV:
    # Applichiamo un leggero restringimento (shrink) locale se pixfrac < 1.0
    if pixfrac < 1.0:
        # Questo è un approccio avanzato: ridimensioniamo l'immagine sorgente 
        # mantenendo i centroidi, per simulare drop più piccoli.
        # Per ora, con pixfrac=1.0, saltiamo questo calcolo pesante.
        pass

    # 3. Trasformazione Geometrica
    # Usiamo INTER_NEAREST per non 'sfumare' il dato originale durante lo spostamento.
    drizzled_frame = cv2.warpPerspective(
        img, 
        M_drizzle, 
        (new_w, new_h), 
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue = [0, 0, 0]
    ) #type: ignore
    
    # 4. Generazione Mappa dei Pesi
    # La mappa dei pesi deve avere lo stesso valore del pixfrac 
    # (o 1.0 se il pixel è presente) per pesare correttamente lo stacking finale.
    # Creiamo una maschera booleana (dove c'è segnale) e la trasformiamo in pesi.
    mask = (np.any(drizzled_frame > 0, axis=2)).astype(np.float32)
    weight_map = mask * pixfrac
    
    return drizzled_frame, weight_map