#####################################################################################################################################################################
###                                                                                                                                                               ###
###   This code is NOT meant to be executed directly. It is a module that provides functions for importing the images contained inside the acquisitions folder.   ###
###                                                                                                                                                               ###
#####################################################################################################################################################################

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
from rawpy import DemosaicAlgorithm # type: ignore

### Import delle immagini raw e standard con normalizzazione della profondità di bit
def general2bgr(file_path: str, target_dtype: str) -> np.ndarray:
    """
    # Summary
    Carica un singolo file immagine, ne rileva la profondità di bit originale
    e lo converte nel formato target normalizzando rispetto al fondo scala.
    ## Inputs
    - file_path: Percorso completo del file immagine da caricare.
    - target_dtype: Tipo di dato di destinazione. Opzioni valide: 'uint8', 'uint16', 'float32', 'float64'.
    ## Outputs
    - img_converted: Immagine convertita nel formato target con normalizzazione del fondo scala.
    ## Note
    - Supporta formati RAW (.dng, .raw, .arw, .nef, .cr2) e formati standard (.jpg, .jpeg, .tiff, .tif, .png, .bmp).
    - Rileva automaticamente la profondità di bit originale dell'immagine.
    - Normalizza l'immagine rispetto al fondo scala originale.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Il file {file_path} non esiste.")

    valid_dtypes = { 'uint8'   : (np.uint8, 255)    ,
                     'uint16'  : (np.uint16, 65535) ,
                     'float32' : (np.float32, 1.0)  ,
                     'float64' : (np.float64, 1.0)  }
    
    raw_extensions = ['.dng', '.raw', '.arw', '.nef', '.cr2']
    standard_extensions = ['.jpg', '.jpeg', '.tiff', '.tif', '.png', '.bmp']

    if target_dtype not in valid_dtypes:
        raise ValueError(f"Target dtype non supportato: {list(valid_dtypes.keys())}")

    ext = path.suffix.lower()
    
    # --- 1. LETTURA E DETERMINAZIONE FONDO SCALA ORIGINALE ---
    if ext in raw_extensions:
        with rawpy.imread(str(path)) as raw:
            # Usiamo RCD o LMMSE per la massima qualità astronomica
            # demosaic_algorithm=rawpy.DemosaicAlgorithm.RCD  # Prova RCD
            # demosaic_algorithm=rawpy.DemosaicAlgorithm.LMMSE # Alternativa ottima
            # Post-elaborazione: forziamo 16-bit per non perdere precisione durante lo sviluppo
            # no_auto_bright=True evita che il software "stiri" l'istogramma
            img = raw.postprocess( demosaic_algorithm = DemosaicAlgorithm.LMMSE,
                                   use_camera_wb      = False   ,
                                   no_auto_bright     = True    ,
                                   output_bps         = 16      ,
                                   no_auto_scale      = True    ,
                                   user_flip          = 0       )
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # Il valore massimo dei dati RAW dipende dal sensore (es. 12, 14 o 16 bit)
            current_max = raw.white_level

    elif ext in standard_extensions:
        # Formati standard (JPG, TIFF, PNG, BMP)
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Impossibile leggere il file: {path}")

        # Rilevamento dinamico del fondo scala basato sul dtype del file letto
        if img.dtype == np.uint8:
            current_max = 255
        elif img.dtype == np.uint16:
            current_max = 65535
        else:
            current_max = np.max(img)
    else:
        raise ValueError(f"Estensione file non supportata: {ext}")

    # --- 2. NORMALIZZAZIONE E CONVERSIONE ---
    # Convertiamo in float64 per la massima precisione durante il calcolo del rapporto
    img_normalized = img.astype(np.float64) / current_max

    # --- 3. MAPPATURA SUL FORMATO RICHIESTO ---
    target_np_type, target_max = valid_dtypes[target_dtype]
    
    if 'float' in target_dtype:
        return img_normalized.astype(target_np_type)
    else:
        # Per uint8 e uint16, riscaliamo al nuovo fondo scala e arrotondiamo
        return np.clip(img_normalized * target_max, 0, target_max).astype(target_np_type)


































###########################################################################################
# FUNZIONI DA RIVEDERE E ADATTARE SUCCESSIVAMENTE AL NUOVO METODO GENERAL2BGR #
###########################################################################################

def jpg(folder, extension):
   """Conversione da JPG in array BGR a 16 bit"""
   extension = extension.strip('.')
   try:
      imported = []
      for file in Path(folder).glob(f'*.{extension}'):
            img = cv2.imread(str(file), cv2.IMREAD_COLOR).astype(np.uint16)*257
            imported.append(img)
            print(f"✅ Caricata immagine {file}.")
      print(f"✅ Caricate {len(imported)} immagini.")
      return imported
   except FileNotFoundError:
      print(f"❌Problemi con l'import delle immagini")
      return []

def raw(folder, extension):
   """Conversione da RAW (o DNG) in array RGB"""
   extension = extension.strip('.')
   try:
      imported = [rawpy.imread(str(file)).postprocess(gamma=(2.222, 4.5), no_auto_bright=True, output_bps=16, use_camera_wb=True) for file in Path(folder).glob(f'*.{extension}')]
      print(f"✅ Caricate {len(imported)} immagini.")
      return imported
   except FileNotFoundError:
      print(f"❌Problemi con l'import delle immagini")
      return []