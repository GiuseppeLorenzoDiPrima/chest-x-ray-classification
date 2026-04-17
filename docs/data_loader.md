# data_classes/data_loader.py

Gestione completa del dataset: download da Kaggle, pulizia, preprocessing, struttura cartelle e costruzione dei DataLoader PyTorch.

---

## Costanti

```python
BINARY_CLASSES  = ["NORMAL", "PNEUMONIA"]
TERNARY_CLASSES = ["BACTERIA", "NORMAL", "VIRUS"]
```

---

## Funzioni di configurazione

### `load_config`

```python
load_config(path: str = "config/config.yaml") -> dict
```

Carica il file YAML di configurazione.

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `path` | `str` | Percorso al file YAML (default: `"config/config.yaml"`) |

**Restituisce:** `dict` — dizionario Python corrispondente al file YAML.

---

## Download e pulizia dataset

### `download_dataset`

```python
download_dataset(config: dict)
```

Scarica il dataset `paultimothymooney/chest-xray-pneumonia` da Kaggle tramite API ufficiale. Se la `data_dir` esiste già e non è vuota, il download viene saltato. Dopo l'estrazione, rimuove automaticamente tutto ciò che non è `train/`, `test/` o `val/` (es. `__MACOSX__`, `.DS_Store`).

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `config` | `dict` | Configurazione YAML caricata da `load_config()` |

**Richiede:** credenziali Kaggle in `~/.kaggle/kaggle.json`.

**Solleva:** `ImportError` se il pacchetto `kaggle` non è installato.

---

### `_cleanup_data_dir`

```python
_cleanup_data_dir(data_dir: str)
```

Rimuove ricorsivamente da `data_dir` qualsiasi file o cartella il cui nome non sia `train`, `test` o `val`.

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `data_dir` | `str` | Percorso della cartella radice del dataset |

---

## Gestione struttura binaria / ternaria

### `split_to_binary`

```python
split_to_binary(data_dir: str)
```

Converte la struttura **ternaria → binaria**: fonde le sottocartelle `BACTERIA/` e `VIRUS/` in `PNEUMONIA/` per ogni split (train, val, test).

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `data_dir` | `str` | Percorso radice del dataset (contiene train/, val/, test/) |

---

### `split_to_ternary`

```python
split_to_ternary(data_dir: str)
```

Converte la struttura **binaria → ternaria**: smista i file di `PNEUMONIA/` in `BACTERIA/` o `VIRUS/` usando il prefisso nel nome file (`bacteria_*` / `virus_*`).

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `data_dir` | `str` | Percorso radice del dataset |

---

### `_is_binary`

```python
_is_binary(data_dir: str) -> bool
```

Restituisce `True` se la struttura è già binaria (`PNEUMONIA/` presente, `BACTERIA/` assente in `train/`).

---

### `_is_ternary`

```python
_is_ternary(data_dir: str) -> bool
```

Restituisce `True` se la struttura è ternaria (`BACTERIA/` presente in `train/`).

---

### `_ensure_structure`

```python
_ensure_structure(data_dir: str, classification_type: str)
```

Rileva la struttura corrente e la converte se non corrisponde al tipo richiesto.

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `data_dir` | `str` | Percorso radice del dataset |
| `classification_type` | `str` | `"binary"` oppure `"ternary"` |

---

## Trasformazioni

### `_get_transforms`

```python
_get_transforms() -> dict[str, transforms.Compose]
```

Restituisce un dizionario con le pipeline di trasformazione per ogni split.

| Split | Trasformazioni applicate |
|-------|--------------------------|
| `"train"` | Resize 224 → CenterCrop 224 → RandomHFlip(p=0.5) → RandomRotation(±10°) → RandomAffine → ToTensor → Normalize |
| `"val"` | Resize 224 → CenterCrop 224 → ToTensor → Normalize |
| `"test"` | Resize 224 → CenterCrop 224 → ToTensor → Normalize |

Normalizzazione ImageNet: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.

---

## Classi dataset

### `ChestXrayDataset`

Wrapper su `torchvision.datasets.ImageFolder`. Ogni chiamata a `__getitem__` restituisce:

```python
{"image": torch.Tensor,  # shape (3, 224, 224)
 "label": int}           # indice di classe
```

Espone `targets`, `classes`, `path` per compatibilità con `WeightedRandomSampler`.

---

### `LabelPreservingConcatDataset`

Versione di `ConcatDataset` che propaga gli attributi `targets`, `classes`, `path` ai dataset concatenati. Necessaria per mantenere la compatibilità con il sampler ponderato dopo la fusione di train + val.

---

### `LabelPreservingSubset`

Versione di `Subset` che propaga gli attributi `targets`, `classes`, `path`. Necessaria dopo il re-split del dataset concatenato.

---

## Ridistribuzione train / val

### `_resize_splits`

```python
_resize_splits(
    train_ds: Dataset,
    val_ds: Dataset,
    split_pct: float,
) -> tuple[LabelPreservingSubset, LabelPreservingSubset]
```

Concatena train e val, mescola con seed fisso e ridistribuisce secondo `split_pct`.

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `train_ds` | `Dataset` | Dataset di training originale |
| `val_ds` | `Dataset` | Dataset di validazione originale |
| `split_pct` | `float` | Frazione destinata al training (es. `0.9` → 90% train, 10% val) |

**Restituisce:** `(new_train, new_val)` — coppia di `LabelPreservingSubset`.

---

## Pipeline principale

### `load_and_preprocess`

```python
load_and_preprocess(config: dict) -> dict
```

Pipeline completa: download → struttura → trasformazioni → split.

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `config` | `dict` | Configurazione YAML caricata da `load_config()` |

**Restituisce:**

```python
{
    "train_dataset":       ChestXrayDataset,  # split train ridistribuito
    "val_dataset":         ChestXrayDataset,  # split val ridistribuito
    "test_dataset":        ChestXrayDataset,  # split test originale
    "classes":             list[str],          # es. ["NORMAL", "PNEUMONIA"]
    "classification_type": str,                # "binary" o "ternary"
}
```

---

## Struttura cartelle attesa

| Tipo | Struttura generata |
|------|--------------------|
| **Binaria** | `train/NORMAL/` · `train/PNEUMONIA/` (e analoghe per val, test) |
| **Ternaria** | `train/BACTERIA/` · `train/NORMAL/` · `train/VIRUS/` (e analoghe) |
