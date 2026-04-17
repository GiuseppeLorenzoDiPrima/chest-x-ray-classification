"""
Uncertainty Quantification tramite MC Dropout.

Esegue N forward pass con il dropout attivo sui modelli DL (ResNet, AlexNet)
e calcola:
- Probabilità medie e deviazione standard per campione
- Entropia predittiva come misura di incertezza
- Rejection curve: accuracy vs soglia di incertezza
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attivazione MC Dropout
# ---------------------------------------------------------------------------

def _enable_mc_dropout(model: nn.Module):
    """Imposta il modello in modalità train SOLO per i layer Dropout."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


# ---------------------------------------------------------------------------
# Inferenza MC Dropout
# ---------------------------------------------------------------------------

def mc_dropout_predict(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    n_iterations: int = 50,
) -> dict:
    """
    Esegue n_iterations forward pass con dropout attivo.

    Returns
    -------
    dict con chiavi:
        mean_probs  : (n_samples, n_classes) — probabilità medie
        std_probs   : (n_samples, n_classes) — deviazione standard
        predictions : (n_samples,)           — classe più probabile
        entropy     : (n_samples,)           — entropia predittiva
        y_true      : (n_samples,)           — etichette vere
    """
    _enable_mc_dropout(model)
    softmax = nn.Softmax(dim=1)

    all_probs: list = []
    y_true_list: list = []

    with torch.no_grad():
        for _ in tqdm(range(n_iterations), desc="MC Dropout", leave=False):
            iter_probs: list = []
            iter_labels_set = False
            for batch in dataloader:
                images  = batch["image"].to(device)
                labels  = batch["label"]
                outputs = model(images)
                probs   = softmax(outputs).cpu().numpy()
                iter_probs.append(probs)
                if not iter_labels_set:
                    y_true_list.extend(labels.numpy())
            # Reset label collection after first iteration
            if _ == 0:
                y_true_list = y_true_list[:len(y_true_list)]
            all_probs.append(np.concatenate(iter_probs, axis=0))

    # all_probs: (n_iter, n_samples, n_classes)
    all_probs_arr = np.stack(all_probs, axis=0)
    y_true        = np.array(y_true_list[:all_probs_arr.shape[1]])

    mean_probs = all_probs_arr.mean(axis=0)   # (n_samples, n_classes)
    std_probs  = all_probs_arr.std(axis=0)    # (n_samples, n_classes)
    predictions = mean_probs.argmax(axis=1)   # (n_samples,)

    # Entropia predittiva: H = -Σ p * log(p)
    eps     = 1e-8
    entropy = -np.sum(mean_probs * np.log(mean_probs + eps), axis=1)

    return {
        "mean_probs":  mean_probs,
        "std_probs":   std_probs,
        "predictions": predictions,
        "entropy":     entropy,
        "y_true":      y_true,
    }


# ---------------------------------------------------------------------------
# Pipeline di analisi
# ---------------------------------------------------------------------------

def run_uncertainty_analysis(dl_results: dict, data: dict, config: dict):
    """
    Analisi MC Dropout per tutti i modelli DL presenti in dl_results.

    Parameters
    ----------
    dl_results : {nome_modello: {"model": nn.Module, "test_dl": DataLoader, ...}}
    data       : dizionario del dataset
    config     : configurazione YAML
    """
    if not config["uncertainty"].get("enabled", True):
        logger.info("Uncertainty Quantification disabilitata dalla configurazione.")
        return

    n_iter  = config["uncertainty"]["mc_dropout_iterations"]
    device_str = config["training"]["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device  = torch.device(device_str)

    for model_name, res in dl_results.items():
        model   = res.get("model")
        test_dl = res.get("test_dl")

        if model is None or test_dl is None:
            logger.warning(f"Model o test_dl non disponibile per {model_name}. Skip.")
            continue

        logger.info(f"MC Dropout uncertainty per {model_name} ({n_iter} iterazioni)...")
        model.to(device)

        mc_results = mc_dropout_predict(model, test_dl, device, n_iterations=n_iter)
        y_test     = mc_results["y_true"]

        acc_standard = (mc_results["predictions"] == y_test).mean()
        mean_entropy = mc_results["entropy"].mean()
        logger.info(
            f"  {model_name} — MC accuracy: {acc_standard:.4f}  "
            f"entropy media: {mean_entropy:.4f}"
        )

        if config["visualization"].get("graph", True):
            from plot.visualization import plot_uncertainty_results
            plot_uncertainty_results(mc_results, y_test, model_name, config)

    print()
    logger.info("Uncertainty Quantification completata.")
