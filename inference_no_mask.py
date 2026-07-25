#!python
from monai.data import PydicomReader # Mettilo in cima se manca
import sys
import argparse
import os
import torch
import numpy as np
from monai import transforms
from monai.data import ITKReader
from lavis.common.config import Config
from lavis.common.registry import registry

def main():
    # --- CONFIGURAZIONI ---
    cfg_path = 'lavis/projects/blip/train/pretrain_ct.yaml'
    ckpt_path = 'model.pth' # I tuoi pesi (.pth)
    
    # Inserisci il percorso della cartella che contiene la serie di file DICOM (.dcm)
    if len(sys.argv) < 2:
        print("Errore: Manca il percorso della cartella DICOM.")
        print("Uso: ./inference_no_mask.py <path_alla_cartella_dicom>")
        sys.exit(1)
        
    # Assegna il primo argomento passato dopo il nome dello script
    dicom_folder_path = sys.argv[1]
    # ----------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Uso il dispositivo: {device}")

    # 1. Caricamento del Modello FVLM tramite LAVIS
    args = argparse.Namespace(cfg_path=cfg_path, options=None)
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config)

    # 2. Caricamento dei pesi pre-addestrati
    print(f"Caricamento pesi da: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=False)
    model.to(device)
    model.eval()

    # 3. Definizione dei testi (Prompts) e delle patologie
    test_items = [
        ('lung', 'Emphysema', 'Not Emphysema.', 'Emphysema.'),
        ('lung', 'Lung nodule', 'Not Nodule.', 'Nodule.'),
        ('heart', 'Cardiomegaly', 'Not Cardiomegaly.', 'Cardiomegaly.'),
        ('aorta', 'Arterial wall calcification', 'Not Arterial wall calcification.', 'Arterial wall calcification.')
    ]
    organs_list = ['lung', 'heart', 'esophagus', 'aorta']

    # 4. Lettura NATIVA DICOM tramite MONAI + ITKReader
    # L'ITKReader è in grado di aggregare automaticamente i file .dcm della cartella in un unico volume 3D
    print(f"Caricamento del volume DICOM da: {dicom_folder_path}")
    loader = transforms.Compose([
        transforms.LoadImaged(
            keys=["image"], 
            reader=PydicomReader(), 
            image_only=True, 
            ensure_channel_first=True
        )
    ])
    
    # Passando il percorso della cartella, ITK leggerà l'intera serie medica
    data = loader({'image': dicom_folder_path})
    image_tensor = data['image'].as_tensor().unsqueeze(0).to(device) # Shape: [1, C, D, H, W]
    print(f"Volume caricato con successo. Dimensioni tensore: {list(image_tensor.shape)}")

    # 5. STRATEGIA SENZA MASCHERE: Generazione di una maschera globale fittizia
    mask_tensor = torch.zeros_like(image_tensor).to(device)
    
    target_organ = 'lung' 
    organ_index = organs_list.index(target_organ)
    mask_tensor.fill_(organ_index + 1) 

    active_organs = [target_organ]
    active_items = [item for item in test_items if item[0] == target_organ]
    whole_organ_sizes = {org: (mask_tensor.sum().item() if org == target_organ else 0) for org in organs_list}

    # 6. Estrazione delle feature testuali
    print(f"Estrazione feature testuali per l'organo impostato fittiziamente ({target_organ})...")
    text_feat_dict = model.prepare_text_feat(active_items)
    organ_feat_dict = {}
    organ_logits = {item: [] for item in active_items}

    # 7. Esecuzione dell'inferenza
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

    # 8. Mostra i risultati finali
    print("\n--- RISULTATI INFERENZA (Maschera Globale Fittizia) ---")
    for item, probs in organ_logits.items():
        if len(probs) > 0:
            prob_positive = np.concatenate(probs).mean(0)[1]
            print(f"Organo analizzato (Intero Volume): {item[0]} | Patologia: {item[1]} -> Probabilità: {prob_positive:.4f}")

if __name__ == '__main__':
    main()

