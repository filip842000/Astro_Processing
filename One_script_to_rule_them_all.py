import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import astroalign as aa
import os
import rawpy
import tifffile
from pathlib import Path
from PIL import Image
import gc
import imageio.v3 as iio
from skimage.restoration import richardson_lucy
from rawpy import DemosaicAlgorithm # type: ignore
from scipy.linalg import logm, expm

import Functions_library.Import_functions as Import
import Functions_library.Conversion_functions as Conversion
import Functions_library.Alignment_functions as Alignment
import Functions_library.Cropping_functions as Cropping
import Functions_library.Stacking_functions as Stacking

####################################################################################################################################
###                                                                                                                              ###
###   This script is meant to execute the entire image processing pipeline using functions from the Functions_library modules.   ###
###                                                                                                                              ###
####################################################################################################################################

### Variables and Parameters
camera_acquisitions_folder = "C:/Users/filip/Desktop/Sessione_25-12-28/Orion 01/Foto all'ombra"
midsave_folder = "C:/Users/filip/Desktop/Sessione_25-12-28/Orion 01/Foto all'ombra/MidSaves"
input_format = ".dng" # Da usare se si vuole importare un formato nello specifico
output_format = ".tif" # Formato di output desiderato
processing_format = "float32" # Formato di elaborazione desiderato: "uint8", "uint16", "float32", "float64"
output_bit_depth = "uint16" # Profondità di bit di output desiderata: "uint8", "uint16", "float32"
reference_identifier = 0 # Identificatore dell'immagine di riferimento per l'allineamento (indice nell'array)
# Cropping paramteres
crop_top_pc = 10.0    # Percentuale da ritagliare dall'alto
crop_bottom_pc = 10.0 # Percentuale da ritagliare dal basso
crop_left_pc = 10.0   # Percentuale da ritagliare da sinistra
crop_right_pc = 10.0  # Percentuale da ritagliare da destra
# Alignment parameters
max_alignment_iterations = 5000
alignment_precision = 1e-7
# Drizzle parameters
upscale_factor = 3
drizzle_pixel_fraction = 0.8
# Stacking parameters
use_richardson_lucy = True
rl_iterations = 30
# Optional: Imposta 'True' per non salvare le immagini intermedie e ottenre solo il risultato finale
life_in_the_fast_lane = True


### Inizializzazione cartella
images_paths = list(Path(camera_acquisitions_folder).glob(f'*{input_format}'))
print(f"✅ Trovate {len(images_paths)} immagini con estensione {input_format} nella cartella {camera_acquisitions_folder}.")
if len(images_paths) == 0:
    raise FileNotFoundError(f"Nessun file con estensione {input_format} trovato nella cartella {camera_acquisitions_folder}.")

### Immagine di riferimento per l'allineamento
try:
    reference_bgr = Import.general2bgr(str(images_paths[reference_identifier]), processing_format)
    print(f"✅ Immagine di riferimento caricata: {images_paths[reference_identifier]}.")
except Exception as e:
    print(f"❌ Errore nel caricamento dell'immagine di riferimento {images_paths[reference_identifier]}: {e}")
    raise e

reference = reference_bgr[:, :, 1]  # Canale verde come riferimento
reference = Cropping.crop_by_percentage(reference, crop_top_pc, crop_bottom_pc, crop_left_pc, crop_right_pc) #Cropping opzionale
# Deallocazione 
del reference_bgr
gc.collect()

### Inizializzazione pre-ciclo
channel = {'B': 0, 'G': 1, 'R': 2}
M = [np.eye(3), np.eye(3), np.eye(3)]  # Matrice di trasformazione identità come default
M_earth = np.eye(3)  # Matrice di trasformazione identità per la correzione della rotazione terrestre
M_prev = np.eye(3)
earth_accumulator = np.zeros((3, 3), dtype=np.float32)
earth_counter = 0
stack =  Stacking.DrizzleStacker(reference.shape[0], reference.shape[1], upscale_factor) # Immagine di base per il drizzle

### Ciclo di elaborazione delle immagini
for idx, image in enumerate(images_paths):
    print(f"\n🔄 Elaborazione immagine {idx + 1} di {len(images_paths)}: {image}")
    print(f"Avanzamento: |{'█' * (idx + 1)}{' ' * (len(images_paths) - idx - 1)}| - {((idx + 1) / len(images_paths)) * 100:.2f}%")
    try:
        # Import dell'immagine
        bgr = Import.general2bgr(str(image), processing_format)
        print(f"✅ Immagine caricata: {image}.")
        # Cropping (opzionale)
        bgr = Cropping.crop_by_percentage(bgr, crop_top_pc, crop_bottom_pc, crop_left_pc, crop_right_pc)
    except Exception as e:
        print(f"❌ Errore nel caricamento dell'immagine {image}: {e}")
        continue
    M_prev_inv = np.linalg.inv(M_prev)
    M_delta = np.matmul(M_prev_inv, M[channel['G']]) #Calcolo il delta prima di aggiornare M_prev
    if idx > 1:
        earth_accumulator += logm(M_delta)
        earth_counter += 1
    M_earth = expm(earth_accumulator / earth_counter) # type: ignore
    M_prev = M[channel['G']] #Aggiorno M_prev
    M[channel['G']] = np.matmul(M_earth, M[channel['G']])  # Composizione delle trasformazioni
    # Allineamento dell'immagine
    try:
        M[channel['G']], ecc_success = Alignment.ecc(reference, bgr[channel['G']], M[channel['G']], max_alignment_iterations, alignment_precision)
        if not ecc_success:
            print(f"⚠️ Allineamento ECC fallito per l'immagine {image}: Tentativo con SIFT.")
            M[channel['G']], sift_success = Alignment.sift(reference, bgr[channel['G']])
            M[channel['G']], ecc_success = Alignment.ecc(reference, bgr[channel['G']], M[channel['G']], max_alignment_iterations, alignment_precision)
        M[channel['R']], _ = Alignment.ecc(reference, bgr[channel['R']], M[channel['G']], max_alignment_iterations, alignment_precision)
        M[channel['B']], _ = Alignment.ecc(reference, bgr[channel['B']], M[channel['G']], max_alignment_iterations, alignment_precision)
    except Exception as e:
        print(f"❌ Errore nell'allineamento {image}: {e}")
        print("Suggerimento: controlla il formato di input, forse è da cambiare")
    # Drizzle