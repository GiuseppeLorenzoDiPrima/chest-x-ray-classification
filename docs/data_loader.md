# data_classes/data_loader.py

Gestione completa del dataset: download, preprocessing e costruzione dei DataLoader.

---

## Funzioni principali

### `load_config(path)`
Carica il file YAML di configurazione.

### `download_dataset(config)`
Scarica il dataset da Kaggle (`paultimothymooney/chest-xray-pneumonia`) nella
cartella `data_dir`. Il download viene saltato se il dataset è già presente.

Richiede le credenziali Kaggle in `~/.kaggle/kaggle.json`.

### `load_and_preprocess(config) → dict`
Pipeline completa:
1. Download (se necessario)
2. Organizzazione struttura binaria/ternaria
3. Caricamento dataset PyTorch con trasformazioni
4. Ridistribuzione train/val secondo `split_percentage`

**Restituisce:**
```python
{
    "train_dataset":       ChestXrayDataset,
    "val_dataset":         ChestXrayDataset,
    "test_dataset":        ChestXrayDataset,
    "classes":             list[str],
    "classification_type": "binary" | "ternary",
}
```

---

## Classi principali

### `ChestXrayDataset`
Wrapper su `torchvision.datasets.ImageFolder`. Ogni `__getitem__` restituisce:
```python
{"image": torch.Tensor, "label": int}
```

### `LabelPreservingConcatDataset` / `LabelPreservingSubset`
Versioni di `ConcatDataset` e `Subset` che espongono gli attributi
`targets`, `classes`, `path` — necessari per il WeightedRandomSampler.

---

## Struttura cartelle (binaria vs ternaria)

| Tipo | Cartelle train/val/test |
|------|------------------------|
| Binaria | `NORMAL/` · `PNEUMONIA/` |
| Ternaria | `BACTERIA/` · `NORMAL/` · `VIRUS/` |

La conversione è automatica e reversibile: `split_to_binary()` fonde
BACTERIA e VIRUS in PNEUMONIA; `split_to_ternary()` separa i file di
PNEUMONIA per prefisso (`bacteria_*` / `virus_*`).

---

## Trasformazioni

| Split | Augmentations |
|-------|---------------|
| Train | Resize 224, CenterCrop, RandomHFlip, RandomRotation ±10°, RandomAffine |
| Val / Test | Resize 224, CenterCrop |

Normalizzazione ImageNet: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.
