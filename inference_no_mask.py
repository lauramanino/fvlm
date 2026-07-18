import os
import shutil
import tempfile
import torch
import numpy as np
from monai import transforms
from lavis.common.config import Config
from lavis.common.registry import registry
import dicom2nifti

def main():
    # --- CONFIGURAZIONI ---
    cfg_path = 'lavis/projects/blip/train/pretrain_ct.yaml'
    ckpt_path = 'model.pth' # I tuoi pesi (.pth)
    
    # Inserisci il percorso della cartella che contiene la serie di file .dcm associati al DICOMDIR
    dicom_folder_path = "data/exam/" 
    # ----------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Uso il dispositivo: {device}")

    # Creiamo una directory temporanea per gestire la conversione
    temp_dir = tempfile.mkdtemp()
    nii_path = os.path.join(temp_dir, "volume.nii.gz")

    try:
        # 1. Conversione da cartella DICOM a file NIfTI (.nii.gz)
        print("Conversione della serie DICOM in volume NIfTI...")
        dicom2nifti.convert_directory(dicom_folder_path, temp_dir, compression=True, reorient=True)
        
        # Individuiamo il file generato (dicom2nifti assegna un nome basato sulla serie)
        generated_files = [f for f in os.listdir(temp_dir) if f.endswith('.nii.gz')]
        if not generated_files:
            raise FileNotFoundError("Impossibile convertire la cartella DICOM. Verifica i file interni.")
        shutil.move(os.path.join(temp_dir, generated_files[0]), nii_path)

        # 2. Caricamento del Modello FVLM tramite LAVIS
        cfg = Config(argparse_args=type('Args', (), {'cfg_path': cfg_path, 'options': None})())
        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config)

        # 3. Caricamento dei pesi pre-addestrati
        print(f"Caricamento pesi da: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model'], strict=False)
        model.to(device)
        model.eval()

        # 4. Definizione dei testi (Prompts) e delle patologie
        test_items = [
            ('lung', 'Emphysema', 'Not Emphysema.', 'Emphysema.'),
            ('lung', 'Lung nodule', 'Not Nodule.', 'Nodule.'),
            ('heart', 'Cardiomegaly', 'Not Cardiomegaly.', 'Cardiomegaly.'),
            ('aorta', 'Arterial wall calcification', 'Not Arterial wall calcification.', 'Arterial wall calcification.')
        ]
        organs_list = ['lung', 'heart', 'esophagus', 'aorta']

        # 5. Caricamento del volume con MONAI
        loader = transforms.Compose([
            transforms.LoadImaged(keys=["image"], image_only=True, ensure_channel_first=True)
        ])
        
        print("Caricamento del volume medico nel modello...")
        data = loader({'image': nii_path})
        image_tensor = data['image'].as_tensor().unsqueeze(0).to(device) # Shape: [1, C, D, H, W]

        # 6. STRATEGIA SENZA MASCHERE: Generazione di una maschera globale fittizia
        # Visto che FVLM mappa l'organo 'X' sul valore intero (indice + 1) della maschera, 
        # creiamo una maschera fittizia per ogni organo clonando le dimensioni dell'immagine.
        # Attenzione: Il modello analizzerà l'intero volume anziché la regione specifica.
        mask_tensor = torch.zeros_like(image_tensor).to(device)
        
        # Diciamo al modello che l'intero spazio dell'immagine contiene l'organo che vogliamo testare
        # Per testare i polmoni (indice 0 -> valore 1), assegniamo 1 a tutta la maschera
        # Se vuoi testare il cuore (valore 2), dovresti impostare la maschera a 2.
        # Testiamo impostando il valore fisso a 1 (lung) per questa sessione.
        target_organ = 'lung' 
        organ_index = organs_list.index(target_organ)
        mask_tensor.fill_(organ_index + 1) 

        # Filtriamo le informazioni coerentemente con l'organo target selezionato
        active_organs = [target_organ]
        active_items = [item for item in test_items if item[0] == target_organ]
        whole_organ_sizes = {org: (mask_tensor.sum().item() if org == target_organ else 0) for org in organs_list}

        # 7. Estrazione delle feature testuali
        print(f"Estrazione feature testuali per l'organo impostato fittiziamente ({target_organ})...")
        text_feat_dict = model.prepare_text_feat(active_items)
        organ_feat_dict = {}
        organ_logits = {item: [] for item in active_items}

        # 8. Esecuzione dell'inferenza
        print("Esecuzione del modello...")
        with torch.no_grad():
            organ_logits = model.forward_test_win(
                image_tensor, 
                mask_tensor, 
                organ_logits, 
                active_organs, 
                text_feat_dict, 
                organ_feat_dict, 
                whole_organ_sizes, 
                skip_organ=-1
            )

        # 9. Mostra i risultati finali
        print("\n--- RISULTATI INFERENZA (Maschera Globale Fittizia) ---")
        for item, probs in organ_logits.items():
            if len(probs) > 0:
                prob_positive = np.concatenate(probs).mean(0)[1]
                print(f"Organo analizzato (Intero Volume): {item[0]} | Patologia: {item[1]} -> Probabilità: {prob_positive:.4f}")

    finally:
        # Pulizia della cartella temporanea
        shutil.rmtree(temp_dir)

if __name__ == '__main__':
    main()

