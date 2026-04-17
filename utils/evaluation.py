"""
Modulo di valutazione dei modelli.

Fornisce:
- compute_metrics()         → accuracy, precision, recall, f1
- evaluate_dl_model()       → valutazione completa su un DataLoader
- build_comparison_table()  → DataFrame comparativo tra modelli
- generate_full_report()    → pipeline completa: CSV + log + grafici
"""

import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metriche base
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred) -> dict:
    """
    Calcola accuracy, precision, recall, f1 (average='macro').

    Parameters
    ----------
    y_true : array-like di etichette vere
    y_pred : array-like di predizioni

    Returns
    -------
    dict con chiavi accuracy, precision, recall, f1
    """
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0.0),
        "recall":    recall_score(y_true, y_pred, average="macro", zero_division=0.0),
        "f1":        f1_score(y_true, y_pred, average="macro", zero_division=0.0),
    }


# ---------------------------------------------------------------------------
# Valutazione modelli DL
# ---------------------------------------------------------------------------

def evaluate_dl_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Valuta un modello DL su un DataLoader.

    Returns
    -------
    metrics        : dict con accuracy, precision, recall, f1, loss
    confusion_mat  : matrice di confusione (n_classes × n_classes)
    y_true         : etichette vere
    y_pred         : predizioni
    """
    model.eval()
    running_loss = 0.0
    y_pred_list: list = []
    y_true_list: list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Valutazione", leave=False):
            images  = batch["image"].to(device)
            labels  = batch["label"].to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            y_pred_list.extend(preds.cpu().numpy())
            y_true_list.extend(labels.cpu().numpy())

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = running_loss / len(dataloader)

    cm = confusion_matrix(y_true, y_pred)
    return metrics, cm, y_true, y_pred


# ---------------------------------------------------------------------------
# Tabella comparativa
# ---------------------------------------------------------------------------

def build_comparison_table(all_results: dict) -> pd.DataFrame:
    """
    Costruisce un DataFrame con una riga per modello e una colonna per metrica.

    Parameters
    ----------
    all_results : {nome_modello: {"metrics": dict, ...}, ...}
    """
    rows = []
    for name, res in all_results.items():
        m = res.get("metrics", {})
        rows.append({
            "Modello":   name,
            "accuracy":  round(m.get("accuracy",  0.0), 4),
            "precision": round(m.get("precision", 0.0), 4),
            "recall":    round(m.get("recall",    0.0), 4),
            "f1":        round(m.get("f1",        0.0), 4),
            "loss":      round(m.get("loss",      float("nan")), 4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report completo
# ---------------------------------------------------------------------------

def generate_full_report(all_results: dict, data: dict, config: dict) -> pd.DataFrame:
    """
    Genera il report finale:
      - Salva la tabella comparativa in CSV
      - Stampa le classification report
      - Produce confusione matrix e ROC per ogni modello
      - Produce grafici di confronto tra modelli

    Parameters
    ----------
    all_results : {nome_modello: result_dict}
    data        : dizionario del dataset
    config      : configurazione YAML

    Returns
    -------
    comparison : pd.DataFrame tabella comparativa
    """
    from plot.visualization import (
        plot_confusion_matrix,
        plot_roc_curves,
        plot_metrics_comparison,
        plot_metric_groups_comparison,
    )

    results_dir = config["paths"]["results_dir"]
    cls_type    = config["classification"]["type"].lower()
    os.makedirs(results_dir, exist_ok=True)

    classes = data["classes"]
    if cls_type == "binary":
        class_labels = ["Normal", "Pneumonia"]
    else:
        class_labels = ["Bacteria", "Normal", "Virus"]

    comparison = build_comparison_table(all_results)

    # CSV
    csv_path = os.path.join(results_dir, "model_comparison.csv")
    comparison.to_csv(csv_path, index=False)
    logger.info(f"Tabella comparativa salvata in {csv_path}")

    # Report testuale
    report_path = os.path.join(results_dir, "classification_reports.txt")
    with open(report_path, "w", encoding="utf-8") as fp:
        from sklearn.metrics import classification_report
        for name, res in all_results.items():
            y_true = res.get("y_true")
            y_pred = res.get("y_pred")
            if y_true is None or y_pred is None:
                continue
            fp.write(f"{'=' * 60}\n{name}\n{'=' * 60}\n")
            fp.write(classification_report(
                y_true, y_pred, target_names=class_labels, zero_division=0
            ))
            fp.write("\n")
    logger.info(f"Classification reports salvati in {report_path}")

    # Grafici: confusion matrix + ROC per ogni modello
    if config["visualization"].get("graph", True):
        for name, res in all_results.items():
            y_true  = res.get("y_true")
            y_pred  = res.get("y_pred")
            y_score = res.get("y_score")

            if y_true is not None and y_pred is not None:
                safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
                plot_confusion_matrix(
                    y_true, y_pred,
                    labels=class_labels,
                    title=f"Confusion Matrix — {name}",
                    config=config,
                    filename=f"cm_{safe}.png",
                )

            if y_score is not None and y_true is not None:
                safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
                plot_roc_curves(
                    y_true, y_score,
                    labels=class_labels,
                    title=f"ROC Curves — {name}",
                    config=config,
                    filename=f"roc_{safe}.png",
                )

        plot_metrics_comparison(comparison, config)
        plot_metric_groups_comparison(comparison, config)

    return comparison
