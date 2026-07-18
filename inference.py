import os
import torch
import numpy as np
from monai import transforms
from lavis.common.config import Config
from lavis.common.registry import registry

def main():
    # --- CONFIGURAZIONI ---
    cfg_path = 'lavis/projects/blip/train/pretrain_ct.yaml'
    # Inserisci qui il percorso dei tuoi pesi scaricati (.pth)
    ckpt_path = 'model.pth' 
    
    # Percorsi dell'immagine (.nii.gz) e della maschera degli organi
    image_path = "data/prova.nii.gz"
    mask_path = "percorso/della/tua/maschera.nii.gz" 
    # ----------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Uso il dispositivo: {device}")

    # 1. Caricamento della configurazione e del Modello tramite LAVIS
    cfg = Config(argparse_args=type('Args', (), {'cfg_path': cfg_path, 'options': None})())
    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config)

    # 2. Caricamento dei pesi pre-addestrati
    print(f"Caricamento pesi da: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=False)
    model.to(device)
    model.eval()

    # 3. Definizione dei testi (Prompts) e degli Organi usati da FVLM
    # La struttura del prompt usata dal modello è: (organo, patologia, testo_negativo, testo_positivo)
    test_items = [
        ('lung', 'Emphysema', 'Not Emphysema.', 'Emphysema.'),
        ('lung', 'Lung nodule', 'Not Nodule.', 'Nodule.'),
        ('heart', 'Cardiomegaly', 'Not Cardiomegaly.', 'Cardiomegaly.'),
        ('aorta', 'Arterial wall calcification', 'Not Arterial wall calcification.', 'Arterial wall calcification.')
    ]
    organs_list = ['lung', 'heart', 'esophagus', 'aorta']

    # 4. Pre-processing dell'immagine con MONAI (caricamento volumetrico)
    loader = transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"], image_only=True, ensure_channel_first=True)
    ])
    
    print("Caricamento e pre-processing dell'immagine...")
    data = loader({'image': image_path, 'label': mask_path})
    image_tensor = data['image'].as_tensor().unsqueeze(0).to(device) # Aggiunge dimensione batch [1, C, D, H, W]
    mask_tensor = data['label'].as_tensor().unsqueeze(0).to(device)

    # Calcolo delle dimensioni degli organi nella TAC per filtrare quelli presenti
    whole_organ_sizes = {
        org: torch.eq(mask_tensor, organs_list.index(org) + 1).sum().item() 
        for org in organs_list
    }
    active_organs = [org for org in organs_list if whole_organ_sizes[org] > 0]
    active_items = [item for item in test_items if item[0] in active_organs]

    # 5. Estrazione delle feature testuali
    print("Estrazione delle feature del testo...")
    text_feat_dict = model.prepare_text_feat(active_items)
    organ_feat_dict = {}

    # Inizializzazione dizionario per i logit/risultati
    organ_logits = {item: [] for item in active_items}

    # 6. Esecuzione dell'inferenza
    print("Esecuzione del modello...")
    with torch.no_grad():
        # Adattamento delle funzioni interne del modello FVLM
        organ_logits = model.forward_test_win(
            image_tensor, 
            mask_tensor, 
            organ_logits, 
            active_organs, 
            text_feat_dict, 
            organ_feat_dict, 
            whole_organ_sizes, 
            skip_organ=-1 # Elabora tutti gli organi disponibili
        )

    # 7. Mostra i risultati finali
    print("\n--- RISULTATI DEL MODELLO ---")
    for item, probs in organ_logits.items():
        if len(probs) > 0:
            # Calcola la media delle probabilità lungo le finestre analizzate
            prob_positive = np.concatenate(probs).mean(0)[1]
            print(f"Organo: {item[0]} | Condizione: {item[1]} -> Probabilità: {prob_positive:.4f}")

if __name__ == '__main__':
    main()

