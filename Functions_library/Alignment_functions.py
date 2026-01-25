#############################################################################################################################################
###                                                                                                                                       ###
### This code is NOT meant to be executed directly. It is a module that provides functions to align images to the desired reference image ###
###                                                                                                                                       ###
#############################################################################################################################################

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

### Allineamento di un'immagine di input rispetto ad una di riferimento usando SIFT e RANSAC
def sift(target_g: np.ndarray, source_g: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Calcola la matrice di trasformazione Omografica 3x3 tra due immagini.
    
    Inputs:
    - target_g: Immagine di riferimento (canale singolo, float32/64 o uint8).
    - source_g: Immagine da allineare (canale singolo).
    
    Outputs:
    - M_homography: Matrice numpy 3x3 (float64).
    - success: Boolean, True se l'allineamento è riuscito con abbastanza punti.
    """
    # 1. Inizializzazione SIFT (usiamo i parametri standard ottimizzati)
    sift = cv2.SIFT_create() #type: ignore

    # Trova punti chiave e descrittori
    kp_ref, des_ref = sift.detectAndCompute(target_g, None)
    kp_src, des_src = sift.detectAndCompute(source_g, None)

    # Gestione caso base: nessun punto trovato
    if des_ref is None or des_src is None:
        return np.eye(3, dtype=np.float64), False

    # 2. Matching con FLANN (più veloce di Brute Force per SIFT)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(cast(Any, index_params), cast(Any, search_params))
    
    matches = flann.knnMatch(des_ref, des_src, k=2)

    # 3. Lowe's Ratio Test per scartare i match ambigui
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 4. Calcolo della Matrice Omografica
    # Servono almeno 4 punti per una omografia, ma ne chiediamo 10 per stabilità
    if len(good_matches) > 10:
        src_pts = np.array([kp_src[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.array([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # RANSAC isola i movimenti coerenti delle stelle dai pixel caldi/rumore
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is not None:
            return M.astype(np.float64), True
        else:
            return np.eye(3, dtype=np.float64), False
    else:
        # Fallimento: non abbastanza match validi
        return np.eye(3, dtype=np.float64), False
    
### Allineamento di un'immagine di input rispetto ad una di riferimento usando ECC
def ecc(target_g: np.ndarray, source_g: np.ndarray, M_init: np.ndarray, max_iterations: int, precision: float) -> tuple[np.ndarray, bool]:
    """
    Rifinisce l'allineamento usando l'algoritmo ECC (Enhanced Correlation Coefficient).
    
    Inputs:
    - target_g: Immagine di riferimento (canale verde, float32).
    - source_g: Immagine da allineare (canale verde, float32).
    - M_init: Matrice 3x3 derivata da SIFT (float64).
    - max_iterations: Numero massimo di iterazioni per ECC.
    - precision: Precisione di convergenza (es. 1e-8).
    
    Outputs:
    - M_final: Matrice 3x3 rifinata (float64).
    - success: Boolean, True se l'algoritmo è convertito.
    """
    # 1. Definiamo i criteri di terminazione
    # L'algoritmo si ferma dopo 50 iterazioni o se lo spostamento è < 1e-8 (precisione estrema)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, precision)
    
    # 2. Prepariamo la matrice iniziale per OpenCV
    # OpenCV ECC richiede una matrice 3x3 per MOTION_HOMOGRAPHY
    M_ecc = M_init.astype(np.float32)

#    target_g = cv2.GaussianBlur(target_g, (3, 3), 0)
#    source_g = cv2.GaussianBlur(source_g, (3, 3), 0)
#
#    target_back = cv2.blur(target_g, (25, 25))
#    source_back = cv2.blur(source_g, (25, 25))
#
#    # Sottraiamo il fondo: restano solo le stelle su un nero quasi perfetto
#    target_g = cv2.subtract(target_g, target_back)
#    source_g = cv2.subtract(source_g, source_back)

    try:
        # 3. Esecuzione dell'algoritmo ECC
        # MOTION_HOMOGRAPHY permette la massima libertà (distorsioni prospettiche)
        (cc, M_ecc) = cv2.findTransformECC(target_g              ,
                                           source_g              ,
                                           M_ecc                 ,
                                           cv2.MOTION_HOMOGRAPHY ,
                                           criteria              ,
                                           cast(Any, None)       , # Maschera (non necessaria se le immagini sono pulite)
                                           5                     ) # Numero di livelli della piramide gaussiana (aiuta la convergenza)
        
        return M_ecc.astype(np.float64), True

    except cv2.error as e:
        # Se l'algoritmo non converge (es. troppe nuvole o mosso eccessivo)
        print(f"ECC fallito: {e}")
        return M_init.astype(np.float64), False

### Predizione della rotazione terrestre
def stepwise_earth_rotation(M_last: np.ndarray, M_penultimate: np.ndarray) -> np.ndarray:
    """
    Predice la matrice di allineamento per il prossimo frame 
    basandosi sul drift rilevato tra i due precedenti.
    
    Inputs:
    - M_last: Matrice 3x3 del frame n-1 (float64)
    - M_penultimate: Matrice 3x3 del frame n-2 (float64)
    
    Output:
    - M_predicted: Matrice 3x3 predetta (float64)
    """
    try:
        # 1. Calcoliamo la variazione (Delta) tra gli ultimi due frame
        # Delta = M_last * inv(M_penultimate)
        delta_m = np.matmul(M_last, np.linalg.inv(M_penultimate))

        return np.matmul(delta_m, M_last)
    except np.linalg.LinAlgError:
        # In caso di errore (es. matrice singolare), ritorniamo l'ultima matrice nota
        return M_last
    
### Applicazione della matrice di omografia per allineare l'immagine
def apply_transformation(img: np.ndarray, M: np.ndarray, interpolation=cv2.INTER_LANCZOS4) -> np.ndarray:
    """
    Applica una matrice di trasformazione omografica 3x3 a un'immagine.
    
    Inputs:
    - img: Immagine sorgente (H, W, 3) o (H, W).
    - M: Matrice di trasformazione 3x3 (float64).
    - interpolation: Algoritmo di interpolazione. 
                     Default: LANCZOS4 (il più preciso per astrofotografia).
    
    Output:
    - transformed_img: Immagine trasformata con le stesse dimensioni dell'originale.
    """
    h, w = img.shape[:2]
    border_value = (0.0, 0.0, 0.0) if len(img.shape) == 3 else (0.0,)  # Nero per immagini a colori o in scala di grigi
    
    # Applichiamo la trasformazione
    # dsize è (larghezza, altezza)
    transformed_img = cv2.warpPerspective(img                               ,
                                          M                                 ,
                                          (w, h)                            ,
                                          flags       = interpolation       ,
                                          borderMode  = cv2.BORDER_CONSTANT ,
                                          borderValue = border_value        )  # Riempie con il nero
    return transformed_img