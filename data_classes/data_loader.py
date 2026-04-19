"""
Modulo di caricamento e preprocessing del dataset Chest X-Ray.

Gestisce:
- Download automatico da Kaggle (paultimothymooney/chest-xray-pneumonia)
- Riorganizzazione della struttura cartelle per classificazione binaria o ternaria
- Costruzione dei dataset PyTorch con trasformazioni opportune
- Ridistribuzione train/val secondo la percentuale configurata
"""

import logging
import os
import shutil

import numpy as np
import torch
import yaml
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

BINARY_CLASSES  = ["NORMAL", "PNEUMONIA"]
TERNARY_CLASSES = ["BACTERIA", "NORMAL", "VIRUS"]


# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Download Kaggle
# ---------------------------------------------------------------------------

SPLITS = ("train", "val", "test")


def download_dataset(config: dict):
    """Scarica il dataset da Kaggle e normalizza la struttura cartelle."""
    data_dir = config["paths"]["data_dir"]

    if _has_final_structure(data_dir):
        logger.info(f"Dataset già presente in '{data_dir}'. Skip download.")
        return

    if not (os.path.exists(data_dir) and os.listdir(data_dir)):
        _download_from_kaggle(config, data_dir)
    else:
        logger.info(
            f"Contenuto trovato in '{data_dir}' ma struttura non normalizzata. "
            "Riorganizzazione in corso..."
        )

    _normalize_structure(data_dir)
    logger.info(f"Struttura dataset pronta in '{data_dir}'.")


def _download_from_kaggle(config: dict, data_dir: str):
    try:
        import kaggle  # type: ignore
    except ImportError:
        raise ImportError(
            "Il pacchetto 'kaggle' non è installato. "
            "Esegui: pip install kaggle"
        )

    os.makedirs(data_dir, exist_ok=True)
    slug = config["dataset"]["kaggle_slug"]
    logger.info(f"Download dataset Kaggle: {slug} → '{data_dir}'")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(slug, path=data_dir, unzip=True, quiet=False)
    logger.info("Download completato.")


def _normalize_structure(data_dir: str):
    """
    Porta la struttura a data_dir/{train,val,test}, indipendentemente
    dal livello di annidamento prodotto dall'archivio Kaggle
    (es. chest_xray/chest_xray/chest_xray) e rimuove residui come
    __MACOSX, .DS_Store, ecc.
    """
    src = _find_splits_dir(data_dir)
    if src is None:
        raise RuntimeError(
            f"Struttura dataset non riconosciuta in '{data_dir}': "
            f"nessuna cartella contiene train/val/test."
        )

    if os.path.abspath(src) != os.path.abspath(data_dir):
        logger.info(f"Split individuati in '{src}'. Promozione a '{data_dir}'.")
        # Sposta prima gli split in una cartella temporanea per evitare
        # conflitti se src è annidato dentro data_dir e data_dir contiene
        # già cartelle omonime (duplicati dell'archivio).
        staging = os.path.join(data_dir, "__staging__")
        if os.path.exists(staging):
            shutil.rmtree(staging)
        os.makedirs(staging)
        for split in SPLITS:
            src_split = os.path.join(src, split)
            if os.path.exists(src_split):
                shutil.move(src_split, os.path.join(staging, split))

        # Rimuove tutto il resto (cartelle annidate, duplicati, __MACOSX, ...)
        for item in os.listdir(data_dir):
            if item == "__staging__":
                continue
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)

        # Promuove gli split dalla staging a data_dir
        for split in SPLITS:
            staged = os.path.join(staging, split)
            if os.path.exists(staged):
                shutil.move(staged, os.path.join(data_dir, split))
        os.rmdir(staging)

    _cleanup_data_dir(data_dir)


def _find_splits_dir(root: str) -> str | None:
    """Prima cartella sotto root che contiene tutti e tre gli split Kaggle."""
    for dirpath, dirnames, _ in os.walk(root):
        if set(SPLITS).issubset(set(dirnames)):
            return dirpath
    return None


def _has_final_structure(data_dir: str) -> bool:
    if not os.path.isdir(data_dir):
        return False
    items = set(os.listdir(data_dir))
    return set(SPLITS).issubset(items) and "chest_xray" not in items


def _cleanup_data_dir(data_dir: str):
    """Rimuove da data_dir qualsiasi file o cartella che non sia train, test o val."""
    for item in os.listdir(data_dir):
        if item in SPLITS:
            continue
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        else:
            os.remove(item_path)
        logger.info(f"Rimosso elemento spurio: '{item}'")


# ---------------------------------------------------------------------------
# Gestione struttura binaria / ternaria
# ---------------------------------------------------------------------------

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def split_to_binary(data_dir: str):
    """
    Converte struttura ternaria (BACTERIA + VIRUS) → binaria (PNEUMONIA).
    Sposta tutti i file di BACTERIA e VIRUS in PNEUMONIA.
    """
    for split in ("train", "val", "test"):
        split_path = os.path.join(data_dir, split)
        pneumonia_path = os.path.join(split_path, "PNEUMONIA")
        _ensure_dir(pneumonia_path)
        for subclass in ("BACTERIA", "VIRUS"):
            subclass_path = os.path.join(split_path, subclass)
            if not os.path.exists(subclass_path):
                continue
            for fname in os.listdir(subclass_path):
                shutil.move(
                    os.path.join(subclass_path, fname),
                    os.path.join(pneumonia_path, fname),
                )
            os.rmdir(subclass_path)
    logger.info("Struttura convertita → binaria (NORMAL / PNEUMONIA).")


def split_to_ternary(data_dir: str):
    """
    Converte struttura binaria (PNEUMONIA) → ternaria (BACTERIA + VIRUS).
    Usa il prefisso 'bacteria'/'virus' nel nome dei file per smistare.
    """
    for split in ("train", "val", "test"):
        split_path  = os.path.join(data_dir, split)
        pneu_path   = os.path.join(split_path, "PNEUMONIA")
        bact_path   = os.path.join(split_path, "BACTERIA")
        virus_path  = os.path.join(split_path, "VIRUS")
        if not os.path.exists(pneu_path):
            continue
        _ensure_dir(bact_path)
        _ensure_dir(virus_path)
        for fname in os.listdir(pneu_path):
            src = os.path.join(pneu_path, fname)
            if "bacteria" in fname.lower():
                shutil.move(src, os.path.join(bact_path, fname))
            else:
                shutil.move(src, os.path.join(virus_path, fname))
        os.rmdir(pneu_path)
    logger.info("Struttura convertita → ternaria (BACTERIA / NORMAL / VIRUS).")


def _is_binary(data_dir: str) -> bool:
    """True se la struttura è già binaria (PNEUMONIA presente, BACTERIA assente)."""
    train = os.path.join(data_dir, "train")
    return (
        os.path.exists(os.path.join(train, "PNEUMONIA"))
        and not os.path.exists(os.path.join(train, "BACTERIA"))
    )


def _is_ternary(data_dir: str) -> bool:
    """True se la struttura è già ternaria (BACTERIA presente)."""
    train = os.path.join(data_dir, "train")
    return os.path.exists(os.path.join(train, "BACTERIA"))


def _ensure_structure(data_dir: str, classification_type: str):
    if classification_type == "binary":
        if not _is_binary(data_dir):
            logger.info("Conversione dataset → struttura binaria...")
            split_to_binary(data_dir)
        else:
            logger.info("Struttura binaria già presente.")
    else:
        if not _is_ternary(data_dir):
            logger.info("Conversione dataset → struttura ternaria...")
            split_to_ternary(data_dir)
        else:
            logger.info("Struttura ternaria già presente.")


# ---------------------------------------------------------------------------
# Trasformazioni
# ---------------------------------------------------------------------------

def _get_transforms() -> dict:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandomAffine(translate=(0.1, 0.05), degrees=10),
            transforms.ToTensor(),
            normalize,
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]),
        "test": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]),
    }


# ---------------------------------------------------------------------------
# Dataset PyTorch
# ---------------------------------------------------------------------------

class ChestXrayDataset(Dataset):
    """
    Dataset per immagini radiografiche del torace.

    Garantisce che tutte le classi attese siano presenti e che gli indici
    delle classi siano coerenti tra split anche quando uno split non
    contiene esempi di una determinata classe (caso tipico: il `val`
    originale di Kaggle, talmente piccolo che dopo la conversione in
    struttura ternaria può non avere file BACTERIA o VIRUS).
    """

    def __init__(self, split: str, root: str, classes: list[str], transform=None):
        path = os.path.join(root, split)

        # Garantisce che tutte le classi attese esistano come sotto-cartelle.
        for cls in classes:
            os.makedirs(os.path.join(path, cls), exist_ok=True)

        # ImageFolder rifiuta cartelle vuote: rimuovile temporaneamente.
        removed: list[str] = []
        for sub in os.scandir(path):
            if sub.is_dir() and not any(os.scandir(sub.path)):
                logger.warning(f"Classe vuota nello split '{split}': {sub.name}")
                removed.append(sub.path)
                os.rmdir(sub.path)

        self.data = ImageFolder(path, transform=transform)

        for p in removed:
            os.makedirs(p, exist_ok=True)

        # Riallinea il mapping delle classi a quello canonico, così che gli
        # indici dei target siano identici in ogni split (necessario per
        # poter concatenare train+val senza falsare le label).
        if list(self.data.classes) != list(classes):
            remap = {old_idx: classes.index(name)
                     for name, old_idx in self.data.class_to_idx.items()}
            self.data.samples = [(p, remap[t]) for p, t in self.data.samples]
            if hasattr(self.data, "imgs"):
                self.data.imgs = self.data.samples
            self.data.targets = [remap[t] for t in self.data.targets]
            self.data.classes = list(classes)
            self.data.class_to_idx = {c: i for i, c in enumerate(classes)}

        self.classes = self.data.classes
        self.targets = self.data.targets
        self.path    = path

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        image, label = self.data[idx]
        return {"image": image, "label": label}


class LabelPreservingConcatDataset(torch.utils.data.ConcatDataset):
    """ConcatDataset che espone gli attributi targets, classes, path."""

    def __init__(self, datasets: list):
        super().__init__(datasets)
        self.targets: list = []
        self.classes = datasets[0].classes
        self.path    = datasets[0].path
        for ds in datasets:
            self.targets.extend(ds.targets)


class LabelPreservingSubset(torch.utils.data.Subset):
    """Subset che espone gli attributi targets, classes, path."""

    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.classes = dataset.classes
        self.targets = [dataset.targets[i] for i in indices]
        self.path    = dataset.path


# ---------------------------------------------------------------------------
# Ridistribuzione train / val
# ---------------------------------------------------------------------------

def _resize_splits(
    train_ds: ChestXrayDataset,
    val_ds: ChestXrayDataset,
    split_pct: float,
):
    """
    Unisce train + val e li ridistribuisce secondo split_pct / (1 - split_pct).
    """
    combined = LabelPreservingConcatDataset([train_ds, val_ds])
    n_train  = int(split_pct * len(combined))
    n_val    = len(combined) - n_train
    train_part, val_part = torch.utils.data.random_split(
        combined, [n_train, n_val]
    )
    return (
        LabelPreservingSubset(combined, train_part.indices),
        LabelPreservingSubset(combined, val_part.indices),
    )


# ---------------------------------------------------------------------------
# Statistiche dataset
# ---------------------------------------------------------------------------

def class_count(dataset) -> np.ndarray:
    counts = np.zeros(len(dataset.classes), dtype=int)
    for label in dataset.targets:
        counts[label] += 1
    return counts


def _log_split_info(name: str, dataset, cls_type: str):
    counts  = class_count(dataset)
    classes = BINARY_CLASSES if cls_type == "binary" else TERNARY_CLASSES
    lines   = [f"  {name}: {len(dataset):,} campioni"]
    for i, c in enumerate(classes):
        lines.append(f"    - {c}: {int(counts[i]):,}")
    logger.info("\n".join(lines))


# ---------------------------------------------------------------------------
# Pipeline principale
# ---------------------------------------------------------------------------

def load_and_preprocess(config: dict) -> dict:
    """
    Esegue l'intera pipeline:
      1. Download Kaggle (se necessario)
      2. Organizzazione struttura binaria / ternaria
      3. Caricamento dataset PyTorch
      4. Ridistribuzione train / val

    Returns
    -------
    dict con chiavi:
        train_dataset, val_dataset, test_dataset,
        classes, classification_type
    """
    data_dir    = config["paths"]["data_dir"]
    cls_type    = config["classification"]["type"].lower()
    split_pct   = config["dataset"]["split_percentage"]

    # 1. Download
    download_dataset(config)

    # 2. Struttura cartelle
    _ensure_structure(data_dir, cls_type)

    # 3. Caricamento
    tfms     = _get_transforms()
    classes  = BINARY_CLASSES if cls_type == "binary" else TERNARY_CLASSES
    train_ds = ChestXrayDataset("train", data_dir, classes, transform=tfms["train"])
    val_ds   = ChestXrayDataset("val",   data_dir, classes, transform=tfms["val"])
    test_ds  = ChestXrayDataset("test",  data_dir, classes, transform=tfms["test"])

    logger.info("\nDataset originali:")
    _log_split_info("Train",      train_ds, cls_type)
    _log_split_info("Validation", val_ds,   cls_type)
    _log_split_info("Test",       test_ds,  cls_type)

    # 4. Ridistribuzione
    train_ds, val_ds = _resize_splits(train_ds, val_ds, split_pct)

    logger.info(
        f"\nDataset dopo la ridistribuzione "
        f"({split_pct:.0%} train / {1 - split_pct:.0%} val):"
    )
    _log_split_info("Train",      train_ds, cls_type)
    _log_split_info("Validation", val_ds,   cls_type)
    _log_split_info("Test",       test_ds,  cls_type)

    return {
        "train_dataset":       train_ds,
        "val_dataset":         val_ds,
        "test_dataset":        test_ds,
        "classes":             train_ds.classes,
        "classification_type": cls_type,
    }
