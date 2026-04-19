"""
Modello SVM per classificazione di immagini radiografiche.

Pipeline:
  1. Estrazione embeddings con Vision Transformer (ViT)
  2. Riduzione dimensionale con PCA
  3. Oversampling SMOTE sul training set
  4. Training e valutazione SVM (kernel RBF)

Gestisce sia la fase di addestramento (fit PCA + SMOTE + SVM)
che quella di test (carica PCA salvata, valuta su test set).
"""

import logging
import os
import time
import warnings

import numpy as np
import torch
from joblib import dump, load
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import hinge_loss, log_loss
from tqdm import tqdm

logger = logging.getLogger(__name__)

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Struttura dati interna
# ---------------------------------------------------------------------------

class _SVMDataset:
    """Contenitore semplice per features, labels e metadati del dataset SVM."""

    def __init__(self, classes, targets, path):
        self.classes        = classes
        self.targets        = targets
        self.path           = path
        self.data: list     = []
        self.features       = None
        self.labels         = None
        self.num_of_features = 0
        self.num_of_samples  = 0


# ---------------------------------------------------------------------------
# Vision Embeddings (ViT)
# ---------------------------------------------------------------------------

class VisionEmbeddings:
    """
    Estrae embedding densi da immagini usando un Vision Transformer pre-addestrato.

    Gli embedding (768-dim dal CLS token medio) vengono poi compressi
    tramite PCA prima di alimentare il classificatore SVM.
    """

    def __init__(self, model_name: str = "google/vit-base-patch16-224",
                 device: str = "cuda"):
        from transformers import ViTImageProcessor, ViTModel  # type: ignore

        self.feature_extractor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"VisionEmbeddings inizializzato: {model_name} su {device}.")

    @staticmethod
    def _denormalize(tensor: torch.Tensor,
                     mean: list, std: list) -> torch.Tensor:
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor

    def _extract(self, dataset, dataset_type: str) -> _SVMDataset:
        new_ds = _SVMDataset(dataset.classes, dataset.targets, dataset.path)
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        for sample in tqdm(dataset, desc=f"ViT embeddings [{dataset_type}]"):
            img = self._denormalize(sample["image"].clone(), mean, std)
            inputs = self.feature_extractor(
                images=img, return_tensors="pt", do_rescale=False
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model(**inputs)
            feat = out.last_hidden_state.mean(dim=1).detach().cpu().numpy()
            new_ds.data.append({"feature": feat.flatten().tolist(),
                                 "label":   sample["label"]})
        return new_ds

    @staticmethod
    def _collect(ds: _SVMDataset):
        features = np.array([item["feature"] for item in ds.data])
        labels   = [item["label"] for item in ds.data]
        n_samples, n_features = features.shape
        ds.features        = features
        ds.labels          = labels
        ds.num_of_samples  = n_samples
        ds.num_of_features = n_features
        return ds

    def extract_all(self, train_ds, val_ds, test_ds,
                    n_components: int, save_path: str,
                    create_scree: bool = False, show_scree: bool = False):
        """
        Estrae embedding, applica PCA (fit su train), SMOTE sul training.
        Salva l'oggetto PCA in save_path/pca.joblib.
        """
        from imblearn.over_sampling import SMOTE  # type: ignore

        train = self._collect(self._extract(train_ds, "train"))
        val   = self._collect(self._extract(val_ds,   "val"))
        test  = self._collect(self._extract(test_ds,  "test"))

        # PCA
        pca = PCA(n_components=n_components)
        pca.fit(train.features)
        train.features = pca.transform(train.features)
        val.features   = pca.transform(val.features)
        test.features  = pca.transform(test.features)

        os.makedirs(save_path, exist_ok=True)
        dump(pca, os.path.join(save_path, "pca.joblib"))
        logger.info(f"PCA salvata in {save_path}/pca.joblib")

        if create_scree:
            logger.info(f"Creazione scree graph PCA (n_components={n_components})...")
            from plot.visualization import plot_scree_graph
            plot_scree_graph(pca, show=show_scree)

        # SMOTE
        smote = SMOTE()
        train.features, train.labels = smote.fit_resample(
            train.features, train.labels
        )
        n_s, n_f = train.features.shape
        train.num_of_samples, train.num_of_features = n_s, n_f
        val.num_of_samples, val.num_of_features   = val.features.shape
        test.num_of_samples, test.num_of_features = test.features.shape

        return train, val, test

    def extract_single(self, dataset, pca_path: str,
                       dataset_type: str = "test",
                       create_scree: bool = False,
                       show_scree: bool = False) -> _SVMDataset:
        """Estrae embedding per un singolo split usando una PCA già salvata."""
        pca = load(pca_path)
        ds  = self._collect(self._extract(dataset, dataset_type))
        ds.features = pca.transform(ds.features)
        ds.num_of_samples, ds.num_of_features = ds.features.shape
        if create_scree and not os.path.exists("outs/imgs/pre-processing/pca_scree.png"):
            print()
            logger.info(f"Creazione scree graph PCA...")
            from plot.visualization import plot_scree_graph
            plot_scree_graph(pca, show=show_scree)
        return ds


# ---------------------------------------------------------------------------
# Training SVM
# ---------------------------------------------------------------------------

def train_svm(data: dict, config: dict) -> dict:
    """
    Addestra il modello SVM con pipeline ViT → PCA → SMOTE.

    Parameters
    ----------
    data   : dizionario restituito da load_and_preprocess()
    config : configurazione YAML

    Returns
    -------
    dict con chiavi:
        model, train_metrics, val_metrics, train_svm, val_svm, test_svm
    """
    cls_type    = config["classification"]["type"].lower()
    models_dir  = config["paths"]["models_dir"]
    svm_cfg     = config["svm"]
    viz_cfg     = config["visualization"]
    device      = config["training"]["device"]

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        
    logger.info(f'=' * 50)
    logger.info("Estrazione embedding ViT + PCA + SMOTE...")
    logger.info(f'=' * 50)

    ve = VisionEmbeddings(device=device)
    emb_start = time.perf_counter()
    train_svm_ds, val_svm_ds, test_svm_ds = ve.extract_all(
        data["train_dataset"],
        data["val_dataset"],
        data["test_dataset"],
        n_components   = svm_cfg["pca_components"],
        save_path      = models_dir,
        create_scree   = viz_cfg["graph"],
        show_scree     = viz_cfg["show"],
    )
    emb_elapsed = time.perf_counter() - emb_start
    emb_min, emb_sec = divmod(emb_elapsed, 60)
    logger.info(
        f"Estrazione embedding ViT + PCA + SMOTE completata in "
        f"{int(emb_min)}m {emb_sec:.1f}s."
    )

    svm_model = svm.SVC(
        C=svm_cfg["C"],
        gamma=svm_cfg["gamma"],
        kernel=svm_cfg["kernel"],
        probability=svm_cfg["probability"],
    )

    print()
    logger.info("Training SVM...")
    svm_train_start = time.perf_counter()
    svm_model.fit(train_svm_ds.features, train_svm_ds.labels)
    svm_train_elapsed = time.perf_counter() - svm_train_start
    svm_min, svm_sec = divmod(svm_train_elapsed, 60)
    logger.info(
        f"Addestramento SVM completato in {int(svm_min)}m {svm_sec:.1f}s."
    )

    # Metriche train / val
    train_pred = svm_model.predict(train_svm_ds.features)
    val_pred   = svm_model.predict(val_svm_ds.features)

    if cls_type == "binary":
        train_loss = hinge_loss(train_svm_ds.labels, train_pred)
        val_loss   = hinge_loss(val_svm_ds.labels,   val_pred)
    else:
        train_loss = log_loss(
            train_svm_ds.labels,
            svm_model.predict_proba(train_svm_ds.features)
        )
        val_loss = log_loss(
            val_svm_ds.labels,
            svm_model.predict_proba(val_svm_ds.features)
        )

    from utils.evaluation import compute_metrics
    train_metrics = compute_metrics(train_svm_ds.labels, train_pred)
    val_metrics   = compute_metrics(val_svm_ds.labels,   val_pred)
    train_metrics["loss"] = train_loss
    val_metrics["loss"]   = val_loss

    logger.info(
        f"  SVM train — loss: {train_loss:.4f}  acc: {train_metrics['accuracy']:.4f}"
    )
    logger.info(
        f"  SVM val   — loss: {val_loss:.4f}  acc: {val_metrics['accuracy']:.4f}"
    )

    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "SVM_best_model.pkl")
    dump(svm_model, model_path)
    
    print()
    logger.info(f"Salvataggio modello SVM...")
    logger.info(f"  Modello SVM salvato in {model_path.replace(os.sep, '/')}")
    
    print()

    return {
        "model":         svm_model,
        "train_metrics": train_metrics,
        "val_metrics":   val_metrics,
        "train_svm":     train_svm_ds,
        "val_svm":       val_svm_ds,
        "test_svm":      test_svm_ds,
    }


# ---------------------------------------------------------------------------
# Valutazione SVM su test set
# ---------------------------------------------------------------------------

def evaluate_svm(data: dict, config: dict,
                 svm_model=None) -> dict:
    """
    Valuta il modello SVM sul test set.
    Se svm_model è None, carica il modello salvato in models_dir.

    Returns
    -------
    dict con chiavi: metrics, confusion_matrix, y_pred, y_true, y_score
    """
    from sklearn.metrics import confusion_matrix

    cls_type   = config["classification"]["type"].lower()
    models_dir = config["paths"]["models_dir"]
    svm_cfg    = config["svm"]
    viz_cfg    = config["visualization"]
    device     = config["training"]["device"]

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    logger.info(f"Estrazione embedding ViT + PCA per test set...")
    # Embeddings test
    ve = VisionEmbeddings(device=device)
    pca_path = os.path.join(models_dir, "pca.joblib")
    test_ds = ve.extract_single(
        data["test_dataset"],
        pca_path=pca_path,
        dataset_type="test",
        create_scree=viz_cfg["graph"],
        show_scree=viz_cfg["show"],
    )

    print()
    logger.info(f"Caricamento modello SVM...")
    
    model_path = os.path.join(models_dir, "SVM_best_model.pkl")
    svm_model  = load(model_path)
        
    logger.info(f"SVM caricata da {model_path}")

    y_pred = svm_model.predict(test_ds.features)
    y_true = np.array(test_ds.labels)
    y_score = (
        svm_model.predict_proba(test_ds.features)
        if svm_cfg["probability"] else None
    )

    if cls_type == "binary":
        loss = hinge_loss(y_true, y_pred)
    else:
        loss = log_loss(y_true, svm_model.predict_proba(test_ds.features))

    from utils.evaluation import compute_metrics
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = loss

    cm = confusion_matrix(y_true, y_pred)
    
    print()
    logger.info(f"Risultati del modello SVM sul test set:")
    logger.info(f"  SVM test  — loss: {loss:.4f}  acc: {metrics['accuracy']:.4f}")

    return {
        "metrics":          metrics,
        "confusion_matrix": cm,
        "y_pred":           y_pred,
        "y_true":           y_true,
        "y_score":          y_score,
        "model":            svm_model,
    }
