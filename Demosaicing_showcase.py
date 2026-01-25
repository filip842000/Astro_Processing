import rawpy
import numpy as np
import colour_demosaicing
import imageio
import os

def benchmark_demosaicing(raw_path):
    with rawpy.imread(raw_path) as raw:
        # 1. Estrazione dati RAW lineari (non processati)
        # Sottraiamo il black level e normalizziamo per lavorare in float
        raw_data = raw.raw_image_visible.astype(np.float32)
        black = raw.black_level_per_channel[0] # Semplificato
        white = raw.white_level
        
        # Normalizzazione tra 0.0 e 1.0
        cfa_data = (raw_data - black) / (white - black)
        cfa_data = np.clip(cfa_data, 0, 1)
        
        # Identificazione del pattern (es. 'RGGB')
        # rawpy fornisce i codici colore 0=R, 1=G, 2=B
        pattern_map = { (0,1,1,2): 'RGGB', (2,1,1,0): 'BGGR', (1,0,2,1): 'GRBG', (1,2,0,1): 'GBRG' }
        # Otteniamo il pattern specifico della fotocamera
        raw_pattern = tuple(raw.raw_color(i, j) for i in range(2) for j in range(2))
        cfa_pattern = pattern_map.get(raw_pattern, 'RGGB')
        
        print(f"Processando {raw_path} con pattern: {cfa_pattern}...")

        # 2. Dizionario degli algoritmi da testare
        algos = {
            "Bilinear": colour_demosaicing.demosaicing_CFA_Bayer_bilinear,
            "Malvar2004": colour_demosaicing.demosaicing_CFA_Bayer_Malvar2004,
            "Menon2007": colour_demosaicing.demosaicing_CFA_Bayer_Menon2007, # Simile a DDFAPD
        }

        # 3. Esecuzione e salvataggio
        results_dir = "demosaicing_comparison"
        os.makedirs(results_dir, exist_ok=True)

        for name, func in algos.items():
            print(f"Eseguendo {name}...")
            rgb = func(cfa_data, cfa_pattern)
            
            # Applichiamo una curva gamma 2.2 elementare per rendere l'immagine visibile
            # (I dati RAW sono lineari e apparirebbero scurissimi senza questa)
            rgb_gamma = np.power(np.clip(rgb, 0, 1), 1/2.2)
            
            # Conversione in 8-bit per il salvataggio rapido (o 16-bit per qualità)
            img_8bit = (rgb_gamma * 255).astype(np.uint8)
            imageio.imsave(f"{results_dir}/{name}.tiff", img_8bit)

        # 4. Caso speciale: RCD (Ratio Corrected Demosaicing)
        print("Eseguendo RCD...")
        rgb_rcd = colour_demosaicing.demosaicing_CFA_Bayer_RCD(cfa_data, cfa_pattern)
        img_rcd = (np.power(np.clip(rgb_rcd, 0, 1), 1/2.2) * 255).astype(np.uint8)
        imageio.imsave(f"{results_dir}/RCD.tiff", img_rcd)

    print(f"Fatto! Controlla la cartella '{results_dir}'")

# Utilizzo
benchmark_demosaicing('Lights_0002_20251228_234552.dng')