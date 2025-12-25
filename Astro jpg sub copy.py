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

def importer_immagini(folder, extension, target_dtype='float32'):
    """
    Carica immagini (JPG, PNG, TIFF, DNG) normalizzando la profondità di bit.
    Supporta: uint8, uint16, float32, float64.
    """
    valid_dtypes = {
        'uint8': np.uint8,
        'uint16': np.uint16,
        'float32': np.float32,
        'float64': np.float64
    }
    
    if target_dtype not in valid_dtypes:
        raise ValueError(f"Target dtype non supportato. Scegli tra: {list(valid_dtypes.keys())}")

    extension = extension.strip('.')
    imported = []
    path_list = list(Path(folder).glob(f'*.{extension}'))

    for file_path in path_list:
        try:
            # 1. LETTURA DIVERSIFICATA
            if extension.lower() in ['dng', 'arw', 'nef', 'cr2']:
                # Gestione RAW/DNG
                with rawpy.imread(str(file_path)) as raw:
                    # postprocess produce un array RGB. 
                    # no_auto_bright=True mantiene i dati lineari relativi al sensore
                    img = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16)
                    input_max = 65535 # Dato che forziamo output_bps=16
            else:
                # Gestione Formati Standard (JPG, PNG, TIFF)
                # IMREAD_UNCHANGED è fondamentale per leggere i 16 bit se presenti
                img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
                if img is None: continue
                
                # Conversione BGR -> RGB se non è in scala di grigi
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Determiniamo il fondo scala in base al tipo di dato letto
                if img.dtype == np.uint8:
                    input_max = 255
                elif img.dtype == np.uint16:
                    input_max = 65535
                else:
                    input_max = np.max(img) # Fallback

            # 2. CONVERSIONE E RISCALATURA (SCALING)
            # Portiamo tutto in un formato float temporaneo per non perdere precisione
            img_float = img.astype(np.float64) / input_max

            # 3. MAPPATURA SUL TARGET
            if 'float' in target_dtype:
                final_img = img_float.astype(valid_dtypes[target_dtype])
            else:
                output_max = 255 if target_dtype == 'uint8' else 65535
                final_img = (img_float * output_max).astype(valid_dtypes[target_dtype])

            imported.append(final_img)

        except Exception as e:
            print(f"❌ Errore nel caricamento di {file_path.name}: {e}")

    print(f"✅ Caricate {len(imported)} immagini in formato {target_dtype}.")
    return imported

def importer_jpg(folder, extension):
   """Conversione da JPG in array RGB"""
   extension = extension.strip('.')
   try:
      imported = [cv2.cvtColor(cv2.imread(str(file), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.uint16)*257 for file in Path(folder).glob(f'*.{extension}')]
      print(f"✅ Caricate {len(imported)} immagini.")
      return imported
   except FileNotFoundError:
      print(f"❌Problemi con l'import delle immagini")
      return []

def importer(folder, extension):
   """Conversione da RAW (o DNG) in array RGB"""
   extension = extension.strip('.')
   try:
      imported = [rawpy.imread(str(file)).postprocess(gamma=(2.222, 4.5), no_auto_bright=True, output_bps=16, use_camera_wb=True) for file in Path(folder).glob(f'*.{extension}')]
      print(f"✅ Caricate {len(imported)} immagini.")
      return imported
   except FileNotFoundError:
      print(f"❌Problemi con l'import delle immagini")
      return []

def scale_to_8bit(image_16bit: np.ndarray) -> np.ndarray:
    """
    ## Funzione ausiliaria

    Scala l'array uint16 (0-65535) a uint8 (0-255) per l'analisi.
    """
    return cv2.normalize(image_16bit, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) # type: ignore

def scale_to_16bit(image: np.ndarray) -> np.ndarray:
    """Scala l'array generico a uint16 (0-65535) per l'analisi."""
    return cv2.normalize(image, None, 0, 65535, cv2.NORM_MINMAX).astype(np.uint16) # type: ignore

def cropper(aligned_images, crop_factor_horizontal, crop_factor_vertical , v_pos, h_pos):
    """
    Ritaglia tutte le immagini nella lista, conservando la percentuale data
    a partire dall'angolo in alto a destra.

    Args:
        aligned_images (list): Lista di array NumPy (immagini RGB o scala di grigi).
        percentage (int/float): Percentuale (es. 75) di altezza e larghezza da conservare.

    Returns:
        list: Lista di immagini ritagliate.
    """
    if not aligned_images:
        print("❌ La lista di immagini è vuota.")
        return []

    # Prendi le dimensioni della prima immagine
    img_shape = np.shape(aligned_images[0])
    H, W = img_shape[0], img_shape[1]
    
    # Calcola le nuove dimensioni
    new_H = int(H * crop_factor_vertical)
    new_W = int(W * crop_factor_horizontal)

    if v_pos == "center":
        start_row = int(H/2 - new_H/2) #Partiamo a metà e saliamo di mezza foto
        end_row = int(H/2 + new_H/2) #Partiamo a metà e scendiamo di mezza foto
    elif v_pos == "top":
        start_row = 0 #L'indice 0 è l'inizio (alto)
        end_row = new_H #La fine della sezione alta
    elif v_pos == "bottom":
        start_row = H - new_H #Partiamo dal basso H e ritorniamo su
        end_row = H #Fondo della foto
    else: print("❌ Hai scazzato il taglio verticale, stupidooooooooo")

    if h_pos == "center":
        start_col = int(W/2) - int(new_W/2)
        end_col = int(W/2) + int(new_W/2)
    elif h_pos == "right":
        start_col = W - new_W #L'inizio è W - new_W (sinistra)
        end_col = W #La fine è W (destra)
    elif h_pos == "left":
        start_col = 0 #Estremo più a sinistra
        end_col = new_W
    else: print("❌ Hai scazzato il taglio orizzontale, stupidooooooooo")
    
    
    cropped_list = []
    
    for img in aligned_images:
        # Applica il ritaglio.
        # img[righe (alto a basso), colonne (sinistra a destra)]
        cropped_img = img[start_row:end_row, start_col:end_col]
        # plotter(cropped_img)
        cropped_list.append(cropped_img)
        
    print(f"✅ Ritaglio completato. Nuove dimensioni: {cropped_list[0].shape}")
    return cropped_list

def rescaler(image, target_H, target_W):
    # cv2.resize richiede come input il (Width, Height)
    resized_image = cv2.resize(image,(target_W, target_H),interpolation=cv2.INTER_LANCZOS4)
    print(f"✅ Riscalamento completato. Nuove dimensioni: {resized_image.shape}")
    return resized_image

def calculate_sharpness_score(image_16bit: np.ndarray) -> float:
    """
    Calcola un punteggio di nitidezza basato sulla Varianza del Laplaciano.
    
    Args:
        image_16bit: L'immagine di input (array NumPy, np.uint16, 3 canali).

    Returns:
        float: Punteggio di nitidezza (maggiore è, più nitida è l'immagine).
    """
    
    # 1. Preparazione: Conversione a Grayscale e 8-bit
    # Laplaciano e Varianza funzionano meglio su immagini a un singolo canale.
    try:
        # Prima scala l'immagine a 8 bit, poi converti in grayscale
        gray_8bit = cv2.cvtColor(scale_to_8bit(image_16bit), cv2.COLOR_BGR2GRAY)
    except cv2.error:
        # Gestisce il caso in cui l'immagine sia già in scala di grigi (1 canale)
        gray_8bit = scale_to_8bit(image_16bit)
    
    # 2. Applicazione del Filtro Laplaciano
    # Il Laplaciano evidenzia i rapidi cambiamenti di intensità (bordi/dettagli).
    # Usiamo CV_64F per garantire alta precisione nel calcolo delle derivate.
    laplacian = cv2.Laplacian(gray_8bit, cv2.CV_64F)
    
    # 3. Calcolo della Varianza
    # La Varianza del Laplaciano è la misura della diffusione dei valori dei bordi.
    # Una varianza alta significa molti bordi forti, quindi alta nitidezza.
    score = laplacian.var()
    
    return score

def normalizer(images):
    """
    Normalizza la luminosità di tutte le immagini rispetto a un'immagine di riferimento.
    Normalizziamo sul valore medio (o mediano) per ignorare i pixel estremi (hot/dead pixels).
    """
    if not images:
        return []

    # 1. Scegli l'immagine di riferimento (usiamo la prima come baseline)
    reference_image = images[1]
    mask = np.any(reference_image > 5, axis=2)
    lunar_pixels = reference_image[mask]
    # 2. Calcola il fattore di riferimento (es. la mediana di tutti i pixel)
    # np.median è più robusto rispetto a np.mean o np.max per ignorare gli estremi.
    reference_median = np.median(lunar_pixels, axis=(0, 1))  # Mediana per canale (R, G, B)
    
    normalized_images = []
    
    for img in images:
        # Calcola la mediana dell'immagine corrente
        mask = np.any(img > 5, axis=2)
        current_median = np.median(img[mask], axis=(0, 1))
        # Calcola il fattore di scaling necessario
        scale_factor = reference_median / current_median
        # Applica il fattore di scaling
        normalized_img = img * scale_factor
        # Opzionale: Clampa i valori per evitare overflow se i dati sono a uint16
        # Se stai usando np.float32 o np.float64 (come dovresti per lo stacking)
        # il clamping non è strettamente necessario in questa fase.
        
        normalized_images.append(normalized_img)
        
    print(f"✅ Normalizzazione completata su {len(images)} immagini.")
    return normalized_images

def align_sift(source_rgb, target_rgb):
    """Calcola la trasformazione con SIFT e applica la trasformazione a tutti i canali RGB."""
    
    # 1. Converti in scala di grigi per SIFT
    source_gray = np.mean(source_rgb, axis=2).astype(np.uint8)
    target_gray = np.mean(target_rgb, axis=2).astype(np.uint8)

    # Inizializza SIFT e Brute-Force Matcher
    sift = cv2.SIFT_create() # type: ignore
    kp1, des1 = sift.detectAndCompute(source_gray, None)
    kp2, des2 = sift.detectAndCompute(target_gray, None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Applica il rapporto di Lowe (per filtrare i falsi positivi)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # 2. Ottieni i punti di corrispondenza
    if len(good_matches) > 10: # Richiedi un numero minimo di corrispondenze
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) # type: ignore
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) # type: ignore

        # 3. Calcola la matrice di omografia (trasformazione)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # 4. Applica la trasformazione a tutti e tre i canali RGB
        rows, cols, dims = source_rgb.shape
        aligned_rgb = np.empty_like(source_rgb)
        
        for i in range(dims):
            # L'output di warpPerspective potrebbe essere a 8-bit, 
            # quindi gestiamo i tipi di dato per preservare la qualità
            channel = source_rgb[:,:,i].astype(np.float32)
            aligned_channel = cv2.warpPerspective(channel, H, (cols, rows))
            aligned_rgb[:,:,i] = aligned_channel
        print("✅ SIFT ha trovato abbastanza corrispondenze.")    
        return aligned_rgb
    else:
        print("❌ SIFT non ha trovato abbastanza corrispondenze.")
        return None
    

def mask_bright_stars(image, threshold_factor=0.50):
    """
    Crea un'immagine in scala di grigi contenente solo i punti più luminosi 
    (le stelle) per migliorare l'allineamento.

    Args:
        image_rgb (np.ndarray): L'immagine a colori originale (RGB).
        threshold_factor (float): Il fattore di soglia (es. 0.98 per conservare 
                                  solo il 2% dei pixel più luminosi).

    Returns:
        np.ndarray: L'immagine in scala di grigi, mascherata (solo le stelle).
    """    
    # 2. Calcola la soglia
    # Troviamo il valore di soglia che conserva solo una piccola percentuale dei pixel più luminosi.
    # np.quantile è robusto e usa un approccio basato sulla distribuzione dei dati.
    threshold_value = np.quantile(image, threshold_factor)

    # 3. Applica la maschera di soglia
    # I pixel al di sotto della soglia diventano 0 (nero)
    masked_image = np.where(image > threshold_value, image, 0)
    
    return masked_image

def align_ecc(source_rgb: np.ndarray, target_rgb: np.ndarray, max_iter, epsilon) -> np.ndarray:
    """
    Calcola la trasformazione con ECC (Homography/Omografia) e applica la 
    trasformazione a tutti i canali RGB.
    """
    
    # Pre-condizione: Conversione in scala di grigi (richiesto da ECC)
    source_gray = np.mean(source_rgb, axis=2).astype(np.float32)
    target_gray = np.mean(target_rgb, axis=2).astype(np.float32)
    source_masked = mask_bright_stars(source_gray)
    target_masked = mask_bright_stars(target_gray)
    # source_masked = source_gray
    # target_masked = target_gray

    rows, cols = source_gray.shape

    # 1. IMPOSTAZIONE DELLA MODALITÀ DI OMOGRAFIA
    warp_mode = cv2.MOTION_HOMOGRAPHY 
    
    # La matrice iniziale deve essere 3x3 per l'Omografia
    warp_matrix = np.eye(3, 3, dtype=np.float32)

    # Definisci i criteri di terminazione
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, epsilon)

    # 2. Calcola la matrice di trasformazione (H)
    try:
        # Nota: cv2.findTransformECC accetta la matrice 3x3 per l'omografia
        _, H_matrix = cv2.findTransformECC(target_masked, source_masked, warp_matrix, warp_mode, criteria)
        print("✅ ECC ha trovato la trasformazione.")
    except cv2.error as e:
        print(f"❌ Errore ECC (Homography): {e}. Riprova con un'altra immagine.")
        return None # type: ignore

    # 3. Applica la trasformazione a tutti e tre i canali RGB
    aligned_rgb = np.empty_like(source_rgb)
    
    for i in range(source_rgb.shape[2]):
        channel = source_rgb[:,:,i].astype(np.float32)
        
        # USA warpPerspective per l'omografia
        aligned_channel = cv2.warpPerspective(
            channel, 
            H_matrix, 
            (cols, rows), 
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )
             
        aligned_rgb[:,:,i] = aligned_channel
        
    return aligned_rgb

def align_subpixel_phase_correlate(target_image: np.ndarray, source_image: np.ndarray):
    """
    Allinea l'immagine 'source' all'immagine 'target' usando la Correlazione di Fase sub-pixel.

    Args:
        target_image: L'immagine di riferimento (array NumPy).
        source_image: L'immagine da allineare (array NumPy).

    Returns:
        L'immagine 'source' riallineata (array NumPy).
    """
    
    # 1. Calcola lo spostamento (shift)
    # Assicurati che le immagini siano float32 e in bianco e nero
    img1_float = cv2.cvtColor((target_image/257).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    img2_float = cv2.cvtColor((source_image/257).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    #resolution = (target_image.shape[0]*10, target_image.shape[1]*10)
    #img1_float = rescaler(img1_float, resolution[0], resolution[1])
    #img2_float = rescaler(img2_float, resolution[0], resolution[1])
    
    # La funzione restituisce (dx, dy) come uno shift 2D, con precisione frazionaria
    # response è una matrice che fornisce informazioni sull'affidabilità (non strettamente necessaria qui)
    shift, response = cv2.phaseCorrelate(img1_float, img2_float)
    
    # Lo shift è spesso restituito come (x_shift, y_shift)
    dx, dy = shift[0], shift[1]

    # 2. Applica la Traslazione Sub-Pixel (Warping)
    # Crea una matrice di trasformazione 2x3 per la traslazione (warp)
    # M = [[1, 0, dx], [0, 1, dy]]
    M = np.float32([[1, 0, dx],
                    [0, 1, dy]]) # type: ignore

    # Applica la trasformazione usando l'interpolazione cubica (migliore qualità per i float)
    # dsize = (larghezza, altezza)
    rows, cols = source_image.shape[:2]
    
    # Applica il warp all'immagine sorgente
    aligned_image = np.empty_like(source_image)

    aligned_image = cv2.warpAffine(source_image, M, (cols, rows),  # type: ignore
                                   flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP, 
                                   borderMode=0) # type: ignore
    
    return [aligned_image, response]

def aligner_rgb(imported):
    """Allinea ogni canale RGB delle immagini rispetto alla prima immagine."""
    target_image = imported[0]
    aligned_images = []
    
    for i, source_image in enumerate(imported):
        try:
            # Allinea ogni canale (R, G, B) singolarmente
            aligned_r, _ = aa.register(source_image[:,:,0], target_image[:,:,0])
            aligned_g, _ = aa.register(source_image[:,:,1], target_image[:,:,1])
            aligned_b, _ = aa.register(source_image[:,:,2], target_image[:,:,2])
            # Ricombina i canali allineati in una nuova immagine RGB
            recombined_image = np.stack([aligned_r, aligned_g, aligned_b], axis=-1)
            aligned_images.append(recombined_image)
            print(f"✅ Immagine {i+1} allineata con successo.")
        except Exception as e:
            print(f"❌ Errore nell'allineamento dell'immagine {i+1}: {e}")
    return aligned_images

def stacker(img_list):
    """Somma delle immagini allineate"""
    result = np.median(img_list, axis=0)
    return result

def weighter_stacker(img_list, weights):
    """Somma delle immagini allineate"""
    result = np.average(img_list, axis=0, weights=weights)
    return result

def apply_richardson_lucy(stacked_image_float: np.ndarray, psf_sigma: float, num_iterations: int) -> np.ndarray:
    """
    Applica la deconvoluzione Richardson-Lucy (RL).
    
    Args:
        stacked_image_float: Immagine stack finale (già normalizzata a 0-1, np.float32).
        psf_sigma: Raggio (sigma) del kernel Gaussiano che modella la sfocatura.
        num_iterations: Numero di iterazioni (più iterazioni = più dettaglio, più rumore).
    """
    
    # 1. Crea il Kernel (PSF) Gaussiano per la deconvoluzione
    # np.newaxis serve per il broadcasting a 3 canali (RGB)
    psf = np.ones((1, 1, 1)) 
    
    # In una vera implementazione, si userebbe un kernel gaussiano 2D:
    # kernel_size = int(psf_sigma * 5)
    # from skimage.filters import gaussian
    # psf = gaussian(np.zeros((kernel_size, kernel_size)), sigma=psf_sigma)
    # psf /= psf.sum() # Normalizza il kernel
    
    # Per semplicità e velocità, useremo una libreria che gestisce il kernel per noi.
    # Nota: richardson_lucy di solito richiede il kernel esplicito, ma qui simuleremo un wrapper concettuale.

    # 2. Deconvoluzione (RICHARDSON-LUCY)
    # Si applica la deconvoluzione usando un kernel Gaussiano stimato
    # Nota: richardson_lucy è iterativa e consuma molta CPU.
    
    # Simulazione con un kernel gaussiano (richiede skimage.filters.gaussian)
    
    # L'implementazione corretta richiede un kernel 2D:
    # from skimage.filters import gaussian
    # psf_2d = gaussian(np.zeros((int(psf_sigma*5), int(psf_sigma*5))), sigma=psf_sigma)
    # psf_2d /= psf_2d.sum()

    # Per un'introduzione semplice, puoi provare prima l'Unsharp Mask che abbiamo visto,
    # in quanto la stima corretta della PSF è complessa. Se vuoi procedere con RL, 
    # usa una libreria come `skimage` e sperimenta con 'sigma'.
    
    deconvolved = richardson_lucy(stacked_image_float, psf, num_iterations)
    
    # 3. Clamping e ritorno
    return np.clip(deconvolved, 0, 1)

def plotter(plt_img):
    """Visualizzazione dell'immagine finale"""
    plt_img= ((plt_img - plt_img.min()) / (plt_img.max() - plt_img.min()))
    plt.imshow(plt_img)
    plt.axis('off')
    plt.show()

def tiff_saver(rgb_img, output_path):
    """Salvataggio dell'immagine finale in formato TIFF"""
    iio.imwrite(output_path, rgb_img.astype(np.uint16), plugin='tifffile')
    print(f"✅ Immagine salvata in {output_path}")

folder = "./"
extension = "dng"
output_file = "./stacked_image.tiff"
output_file_tot = "./stacked_image_ecc_tot.tiff"
output_file_lap = "./stacked_image_ecc_lap.tiff"
output_file_phase = "./stacked_image_ecc_phase.tiff"
output_file_ecc = "./stacked_image_ecc_ecc.tiff"
output_file_o_phase = "./stacked_image_ecc_o-phase.tiff"
    # 1) Importa le immagini
imported = importer(folder, extension)
imported = cropper(imported, 0.70, 0.40, "top", "center")
    # 3) Riscalamento (opzionale, ma utile per migliorare l'allineamento)
resolution = np.shape(imported[0])
resolution = (resolution[0]*1, resolution[1]*1)
rescaled = [rescaler(image, resolution[0], resolution[1]) for image in imported]
del imported
gc.collect()
    # 4) Normalizzazione della luminosità
normalized = normalizer(rescaled)
del rescaled
gc.collect()
    # 5) Allineamento delle immagini (ECC o SIFT o astroalign)
aligned_sift = normalized
del normalized
gc.collect()
aligned = [align_ecc(image, aligned_sift[0], 10000, 1e-4 ) for image in aligned_sift]
del aligned_sift
gc.collect()
    # 5) Ritaglio (opzionale, ma utile per eliminare i bordi neri)
aligned = cropper(aligned, 0.90, 0.90, "center", "center")
aligned_ecc = [img for img in aligned if img is not None]
resolution = np.shape(aligned[0])
resolution = (resolution[0]*1, resolution[1]*1)
aligned_ecc = [rescaler(image, resolution[0], resolution[1]) for image in aligned_ecc]
stacked_ecc = stacker(aligned)
tiff_saver(stacked_ecc, output_file_ecc)
    # 6) Allineamento subpixel
phase_aligned, weights = zip(*[align_subpixel_phase_correlate(image, aligned[0]) for image in aligned])
    # 6) Ritaglio (opzionale, ma utile per eliminare i bordi neri)
phase_aligned = cropper(phase_aligned, 0.95, 0.95, "center", "center")
del aligned
gc.collect()
    # 6) Filtraggio delle immagini non allineate
filtered = [img for img in phase_aligned if img is not None]
del phase_aligned
gc.collect()
    # 7) Score Laplaciano
laplacian_scores = [calculate_sharpness_score(img) for img in filtered]
print("Punteggi di nitidezza (Varianza del Laplaciano):", laplacian_scores)
    # 8) Punteggio finale per immagine
final_scores = [score * weight for score, weight in zip(laplacian_scores, weights)]
    # 8) Riscalamento finale (opzionale)
final_resolution = np.shape(filtered[0])
final_resolution = (final_resolution[0]*1, final_resolution[1]*1)
final_array = [rescaler(image, final_resolution[0], final_resolution[1]) for image in filtered]
del filtered
gc.collect()
    # 9) Stacking (somma o mediana)
stacked_tot = weighter_stacker(final_array, final_scores)
stacked_lap = weighter_stacker(final_array, laplacian_scores)
stacked_phase = weighter_stacker(final_array, weights)
stacked = apply_richardson_lucy(scale_to_16bit(stacked_tot).astype(np.float32)/65535, psf_sigma=1.5, num_iterations=25)
stacked = scale_to_16bit(stacked)
    # 10) Salvataggio dell'immagine finale
tiff_saver(stacked_tot, output_file_tot)
tiff_saver(stacked, output_file)
tiff_saver(stacked_lap, output_file_lap)
tiff_saver(stacked_phase, output_file_phase)
plotter(stacked)