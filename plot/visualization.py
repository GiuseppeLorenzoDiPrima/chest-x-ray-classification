"""
Modulo di visualizzazione per il progetto Chest X-Ray Classification.

Include:
- Stile IEEE-ready (palette, font, spines)
- Distribuzione classi per split
- Andamento training (loss / accuracy)
- Matrice di confusione
- Curve ROC (one-vs-rest)
- PCA scree plot
- Confronto metriche tra modelli
- Grafici MC Dropout (uncertainty)
- Grafici SHAP
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Palette e stile IEEE
# ---------------------------------------------------------------------------

IEEE_PALETTE = [
    "#2166AC",  # blu
    "#B2182B",  # rosso scuro
    "#1B7837",  # verde foresta
    "#D6604D",  # arancione mattone
    "#762A83",  # viola
    "#4D4D4D",  # grigio antracite
]

IEEE_LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]
IEEE_MARKERS    = ["o", "s", "D", "^", "v", "P"]

# Classi
BINARY_LABELS  = ["Normal", "Pneumonia"]
TERNARY_LABELS = ["Bacteria", "Normal", "Virus"]


def setup_publication_style(config: dict):
    """Imposta lo stile grafico per paper scientifici (IEEE-ready)."""
    sns.set_palette(IEEE_PALETTE)
    plt.rcParams.update({
        "font.family":        "serif",
        "font.size":          11,
        "axes.titlesize":     13,
        "axes.labelsize":     11,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.fontsize":    10,
        "legend.frameon":     True,
        "legend.edgecolor":   "#d3d3d3",
        "legend.fancybox":    False,
        "figure.dpi":         config["visualization"].get("dpi", 300),
        "savefig.dpi":        config["visualization"].get("dpi", 300),
        "savefig.bbox":       "tight",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     1.2,
        "axes.edgecolor":     "#333333",
        "text.color":         "#333333",
        "axes.labelcolor":    "#333333",
        "xtick.color":        "#333333",
        "ytick.color":        "#333333",
    })


def _class_labels(classification_type: str) -> list[str]:
    return (BINARY_LABELS if classification_type == "binary" else TERNARY_LABELS)


# ---------------------------------------------------------------------------
# Distribuzione classi
# ---------------------------------------------------------------------------

def plot_class_distribution(data: dict, config: dict):
    """
    Grafici a barre della distribuzione delle classi per ogni split
    (train, val, test) e per il dataset completo.
    """
    setup_publication_style(config)
    cls_type = data["classification_type"]
    labels   = _class_labels(cls_type)
    fig_dir  = os.path.join(config["paths"]["figures_dir"], "pre-processing")
    os.makedirs(fig_dir, exist_ok=True)
    dpi     = config["visualization"]["dpi"]
    figsize = config["visualization"]["figsize"]

    splits = {
        "Training":   data["train_dataset"],
        "Validation": data["val_dataset"],
        "Test":       data["test_dataset"],
    }

    all_splits = {**splits, "Overall": None}

    for split_name, ds in all_splits.items():
        if ds is None:
            targets = np.concatenate([
                np.array(splits[s].targets) for s in splits
            ])
        else:
            targets = np.array(ds.targets)
        unique, counts = np.unique(targets, return_counts=True)
        split_labels   = [labels[u] for u in unique]

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        bars = ax.bar(
            split_labels, counts,
            color=[IEEE_PALETTE[i % len(IEEE_PALETTE)] for i in range(len(unique))],
            edgecolor="#333333", linewidth=0.8,
        )
        title_suffix = "intero dataset" if split_name == "Overall" else f"{split_name} set"
        ax.set_ylabel("Numero di campioni", fontweight="bold")
        ax.set_title(f"Distribuzione classi — {title_suffix}", pad=12)
        ax.grid(True, linestyle=":", alpha=0.7, color="#A9A9A9", axis="y", zorder=0)
        for bar in bars:
            bar.set_zorder(2)
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                f"{count:,}", ha="center", va="bottom", fontsize=10,
            )
        fig.tight_layout()
        fname = f"class_distribution_{split_name.lower()}.png"
        fig.savefig(os.path.join(fig_dir, fname))
        plt.close(fig)
        logger.info(f"  Salvato {fname}")


# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------

def plot_training_history(
    training_metrics: list[dict],
    validation_metrics: list[dict],
    model_name: str,
    config: dict,
):
    """
    Curve di loss e accuracy su training e validation per modelli DL.

    Parameters
    ----------
    training_metrics   : lista di dict {"loss": ..., "accuracy": ..., ...}
    validation_metrics : lista di dict
    model_name         : nome del modello (es. "ResNet")
    config             : configurazione YAML
    """
    setup_publication_style(config)
    safe_name = model_name.lower().replace(" ", "_")
    fig_dir = os.path.join(config["paths"]["figures_dir"], "training", safe_name)
    os.makedirs(fig_dir, exist_ok=True)
    dpi     = config["visualization"]["dpi"]
    figsize = [14, 8]

    metrics_to_plot = config["visualization"].get("metrics_to_plot",
                                                   ["loss", "accuracy"])
    epochs = range(1, len(training_metrics) + 1)

    for metric in metrics_to_plot:
        if metric not in training_metrics[0]:
            continue
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.plot(
            epochs, [m[metric] for m in training_metrics],
            label="Train", linewidth=1.8,
            color=IEEE_PALETTE[0],
        )
        ax.plot(
            epochs, [m[metric] for m in validation_metrics],
            label="Validation", linewidth=1.8,
            color=IEEE_PALETTE[1], linestyle="--",
        )
        ax.set_xlabel("Epoch", fontweight="bold")
        ax.set_ylabel(metric.capitalize(), fontweight="bold")
        ax.set_title(f"{model_name} — {metric.capitalize()} per epoch", pad=12)
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.7, color="#A9A9A9")
        fig.tight_layout()
        safe_metric = metric.replace(" ", "_")
        fname = f"{model_name.lower()}_{safe_metric}.png"
        fig.savefig(os.path.join(fig_dir, fname))
        plt.close(fig)
        logger.info(f"  Salvato {fname}")


# ---------------------------------------------------------------------------
# Matrice di confusione
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true, y_pred,
    labels: list[str],
    title: str,
    config: dict,
    filename: str,
    subdir: str = "confusion_matrix",
):
    """Salva la matrice di confusione come immagine IEEE-style."""
    setup_publication_style(config)
    fig_dir = os.path.join(config["paths"]["figures_dir"], subdir)
    os.makedirs(fig_dir, exist_ok=True)
    dpi     = config["visualization"]["dpi"]
    figsize = config["visualization"]["figsize"]

    cm   = confusion_matrix(y_true, y_pred)
    n    = cm.shape[0]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor("#333333")

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="#bbbbbb", linewidth=0.6, linestyle="-")
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(title, pad=12)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, filename))
    plt.close(fig)
    logger.info(f"  Salvato {filename}")


# ---------------------------------------------------------------------------
# Curve ROC
# ---------------------------------------------------------------------------

def plot_roc_curves(
    y_true, y_score,
    labels: list[str],
    title: str,
    config: dict,
    filename: str,
    subdir: str = "roc_curves",
):
    """Curve ROC one-vs-rest per classificazione multi-classe."""
    setup_publication_style(config)
    fig_dir = os.path.join(config["paths"]["figures_dir"], subdir)
    os.makedirs(fig_dir, exist_ok=True)
    dpi     = config["visualization"]["dpi"]
    figsize = [10, 8]

    n_classes = len(labels)
    y_bin     = label_binarize(y_true, classes=list(range(n_classes)))
    # label_binarize restituisce shape (n, 1) per caso binario — espandiamo a (n, 2)
    if n_classes == 2:
        y_bin = np.column_stack([1 - y_bin, y_bin])

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])  # type: ignore
        roc_auc     = auc(fpr, tpr)
        ax.plot(
            fpr, tpr,
            lw=2,
            color=IEEE_PALETTE[i % len(IEEE_PALETTE)],
            linestyle=IEEE_LINESTYLES[i % len(IEEE_LINESTYLES)],
            marker=IEEE_MARKERS[i % len(IEEE_MARKERS)],
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            markevery=max(1, len(fpr) // 10),
            alpha=0.95,
            label=f"{labels[i]} (AUC = {roc_auc:.3f})",
        )
    ax.plot([0, 1], [0, 1], color="#999999", lw=1, linestyle="--", alpha=0.7)
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontweight="bold")
    ax.set_title(title, pad=12)
    ax.legend(fontsize=9, framealpha=0.95)
    ax.grid(True, linestyle=":", alpha=0.7, color="#A9A9A9")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, filename))
    plt.close(fig)
    logger.info(f"  Salvato {filename}")


# ---------------------------------------------------------------------------
# PCA Scree plot
# ---------------------------------------------------------------------------

def plot_scree_graph(pca, config: dict = None, show: bool = False):
    """Visualizza la varianza spiegata per componente PCA."""
    from plot.visualization import setup_publication_style  # noqa: F401

    fig_dir = "outs/imgs/pre-processing"
    dpi     = 300
    figsize = [10, 6]

    if config is not None:
        setup_publication_style(config)
        fig_dir = os.path.join(config["paths"]["figures_dir"], "pre-processing")
        dpi     = config["visualization"]["dpi"]
        figsize = config["visualization"]["figsize"]

    os.makedirs(fig_dir, exist_ok=True)

    evr = pca.explained_variance_ratio_
    cumulative = np.cumsum(evr)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.bar(
        range(1, len(evr) + 1), evr,
        alpha=0.7, color=IEEE_PALETTE[0],
        edgecolor="#333333", linewidth=0.8,
        label="Varianza spiegata per componente",
        zorder=2,
    )
    ax.plot(
        range(1, len(evr) + 1), cumulative,
        color=IEEE_PALETTE[1], linewidth=2,
        marker="o", markersize=5,
        label="Varianza cumulativa",
    )
    ax.set_xlabel("Componenti principali", fontweight="bold")
    ax.set_ylabel("Varianza spiegata", fontweight="bold")
    ax.set_title("PCA — Varianza spiegata per componente", pad=12)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.7, color="#A9A9A9", zorder=0)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "pca_scree.png"))
    if show:
        plt.show()
    plt.close(fig)
    logger.info("  Salvato pca_scree.png")


# ---------------------------------------------------------------------------
# Confronto modelli
# ---------------------------------------------------------------------------

def plot_metrics_comparison(comparison: pd.DataFrame, config: dict):
    """Grafico a barre orizzontali per ogni metrica, un file per metrica."""
    setup_publication_style(config)
    metrics = config["visualization"].get("comparison_metrics", ["accuracy"])
    metrics = [m for m in metrics if m in comparison.columns]
    if not metrics:
        return

    fig_dir = os.path.join(config["paths"]["figures_dir"], "model_comparison")
    os.makedirs(fig_dir, exist_ok=True)
    dpi     = config["visualization"]["dpi"]
    figsize = [10, 8]

    models  = comparison["Modello"].tolist()
    colors  = [IEEE_PALETTE[i % len(IEEE_PALETTE)] for i in range(len(models))]

    for metric in metrics:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        values  = comparison[metric].tolist()
        bars    = ax.barh(
            models[::-1], values[::-1],
            color=colors[::-1],
            edgecolor="#333333", linewidth=0.8,
        )
        for bar, value in zip(bars, values[::-1]):
            ax.text(
                bar.get_width() + 0.004,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.4f}", va="center", fontsize=9,
            )
        ax.set_xlabel(metric.capitalize(), fontweight="bold")
        ax.set_title(f"Confronto {metric.capitalize()} tra modelli", pad=12)
        ax.set_xlim(0, 1.08)
        ax.grid(True, linestyle=":", alpha=0.7, color="#A9A9A9", axis="x", zorder=0)
        for bar in bars:
            bar.set_zorder(2)
        fig.tight_layout()
        fname = f"model_{metric}_comparison.png"
        fig.savefig(os.path.join(fig_dir, fname))
        plt.close(fig)
        logger.info(f"  Salvato {fname}")


def plot_metric_groups_comparison(comparison: pd.DataFrame, config: dict):
    """Grafico a barre raggruppate per modello con multiple metriche."""
    setup_publication_style(config)
    metrics = config["visualization"].get("comparison_group_metrics", ["accuracy"])
    metrics = [m for m in metrics if m in comparison.columns]
    if not metrics:
        return

    fig_dir = os.path.join(config["paths"]["figures_dir"], "model_comparison")
    os.makedirs(fig_dir, exist_ok=True)
    dpi     = config["visualization"]["dpi"]
    figsize = [14, 8]

    models    = comparison["Modello"].tolist()
    n_models  = len(models)
    n_metrics = len(metrics)
    x         = np.arange(n_models)
    width     = 0.8 / n_metrics
    colors    = [IEEE_PALETTE[i % len(IEEE_PALETTE)] for i in range(n_metrics)]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for i, metric in enumerate(metrics):
        values = comparison[metric].tolist()
        ax.bar(
            x + i * width, values, width,
            label=metric.capitalize(),
            color=colors[i], edgecolor="#333333", linewidth=0.8, zorder=2,
        )
    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel("Valore", fontweight="bold")
    ax.set_title("Confronto metriche per modello", pad=12)
    ax.set_ylim(0, 1.05)
    ax.legend(title="Metrica")
    ax.grid(True, linestyle=":", alpha=0.7, color="#A9A9A9", axis="y", zorder=0)
    for i, metric in enumerate(metrics):
        values = comparison[metric].tolist()
        for j, value in enumerate(values):
            ax.text(
                x[j] + i * width, value + 0.01,
                f"{value:.3f}", ha="center", va="bottom", fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "model_comparison_groups.png"))
    plt.close(fig)
    logger.info("  Salvato model_comparison_groups.png")


# ---------------------------------------------------------------------------
# Uncertainty (MC Dropout)
# ---------------------------------------------------------------------------

def plot_uncertainty_results(
    mc_results: dict,
    y_test: np.ndarray,
    model_name: str,
    config: dict,
):
    """
    Grafici di uncertainty quantification da MC Dropout:
    1. Distribuzione entropia (corrette vs errate)
    2. Rejection curve
    3. Boxplot incertezza per classe
    """
    setup_publication_style(config)
    safe    = model_name.lower().replace(" ", "_")
    unc_dir = os.path.join(config["paths"]["figures_dir"], "uncertainty", safe)
    os.makedirs(unc_dir, exist_ok=True)
    dpi     = config["visualization"]["dpi"]
    figsize = [10, 8]
    cls_type = config["classification"]["type"].lower()
    labels   = _class_labels(cls_type)

    entropy = mc_results["entropy"]
    y_pred  = mc_results["predictions"]
    correct = y_pred == y_test

    # 1. Distribuzione entropia
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.hist(entropy[correct],  bins=50, alpha=0.6, density=True,
            color=IEEE_PALETTE[0], edgecolor="white", linewidth=0.4,
            label="Predizioni corrette")
    ax.hist(entropy[~correct], bins=50, alpha=0.6, density=True,
            color=IEEE_PALETTE[1], edgecolor="white", linewidth=0.4,
            label="Predizioni errate")
    ax.set_xlabel("Entropia", fontweight="bold")
    ax.set_ylabel("Densità", fontweight="bold")
    ax.set_title(f"{model_name} — Incertezza: corrette vs errate", pad=12)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.7, color="#A9A9A9")
    fig.tight_layout()
    fname = f"uncertainty_entropy_{safe}.png"
    fig.savefig(os.path.join(unc_dir, fname))
    plt.close(fig)
    logger.info(f"  Salvato {fname}")

    # 2. Rejection curve
    thresholds = np.linspace(0, np.max(entropy), 100)
    accs, fracs = [], []
    for thr in thresholds:
        mask = entropy <= thr
        if mask.sum() == 0:
            continue
        accs.append((y_pred[mask] == y_test[mask]).mean())
        fracs.append(mask.mean())

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot([f * 100 for f in fracs], accs, lw=2, color=IEEE_PALETTE[0])
    baseline_acc = (y_pred == y_test).mean()
    ax.axhline(y=baseline_acc, color=IEEE_PALETTE[1], ls="--",
               label=f"Accuracy senza filtro: {baseline_acc:.4f}")
    ax.set_xlabel("Percentuale eventi accettati (%)", fontweight="bold")
    ax.set_ylabel("Accuracy sugli eventi accettati", fontweight="bold")
    ax.set_title(f"{model_name} — Rejection Curve", pad=12)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.7, color="#A9A9A9")
    fig.tight_layout()
    fname = f"rejection_curve_{safe}.png"
    fig.savefig(os.path.join(unc_dir, fname))
    plt.close(fig)
    logger.info(f"  Salvato {fname}")

    # 3. Boxplot incertezza per classe
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    class_entropies = [entropy[y_test == c] for c in range(len(labels))]
    ax.boxplot(class_entropies, labels=labels,
               patch_artist=True,
               boxprops=dict(facecolor=IEEE_PALETTE[0], alpha=0.6))
    ax.set_ylabel("Entropia", fontweight="bold")
    ax.set_title(f"{model_name} — Incertezza per classe", pad=12)
    ax.grid(True, linestyle=":", alpha=0.7, color="#A9A9A9")
    fig.tight_layout()
    fname = f"uncertainty_per_class_{safe}.png"
    fig.savefig(os.path.join(unc_dir, fname))
    plt.close(fig)
    logger.info(f"  Salvato {fname}")


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def plot_shap_results(
    shap_values,
    X_sample: np.ndarray,
    feature_names: list[str],
    class_labels: list[str],
    model_name: str,
    fig_dir: str,
    dpi: int,
    figsize: tuple,
):
    """
    Genera summary plot, bar plot importanza media e plot per classe (SHAP).

    Parameters
    ----------
    shap_values   : output shap.KernelExplainer (lista di array o array 3D)
    X_sample      : features campionate (n_samples, n_features)
    feature_names : etichette delle features (es. ["PC1", "PC2", ...])
    class_labels  : etichette delle classi
    model_name    : nome del modello
    fig_dir       : cartella di salvataggio
    dpi, figsize  : parametri grafici
    """
    import shap  # type: ignore

    os.makedirs(fig_dir, exist_ok=True)
    safe = model_name.lower().replace(" ", "_")

    # Normalizza in lista di array per multi-classe
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv_list = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
    elif isinstance(shap_values, list):
        sv_list = shap_values
    else:
        sv_list = [shap_values]

    figsize_small = (6, 4)

    # 1. Summary aggregato
    shap.summary_plot(sv_list, X_sample, feature_names=feature_names,
                      class_names=class_labels, show=False, plot_size=None)
    fig_s = plt.gcf()
    ax_s  = plt.gca()
    ax_s.set_title(f"SHAP Summary — {model_name}", fontsize=13, pad=12)
    fig_s.set_size_inches(figsize_small)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"SHAP_summary_{safe}.png"), dpi=dpi)
    plt.close("all")
    logger.info(f"    Salvato SHAP_summary_{safe}.png")

    # 2. Bar plot importanza media
    mean_abs    = np.mean([np.abs(sv).mean(axis=0) for sv in sv_list], axis=0)
    sorted_idx  = np.argsort(mean_abs)
    n_feats     = len(sorted_idx)
    bar_colors  = [IEEE_PALETTE[i % len(IEEE_PALETTE)] for i in range(n_feats)]
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        mean_abs[sorted_idx],
        color=bar_colors, edgecolor="#333333", linewidth=0.8, zorder=2,
    )
    ax.set_xlabel("Mean |SHAP value|", fontweight="bold")
    ax.set_title(f"SHAP Feature Importance — {model_name}", pad=12)
    ax.grid(True, linestyle=":", alpha=0.7, color="#A9A9A9", axis="x", zorder=0)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"SHAP_bar_{safe}.png"), dpi=dpi)
    plt.close(fig)
    logger.info(f"    Salvato SHAP_bar_{safe}.png")

    # 3. Summary per singola classe
    for class_idx, label in enumerate(class_labels):
        shap.summary_plot(sv_list[class_idx], X_sample,
                          feature_names=feature_names,
                          show=False, plot_size=None)
        fig_c = plt.gcf()
        ax_c  = plt.gca()
        ax_c.set_title(f"SHAP {model_name} — {label}", fontsize=13, pad=12)
        fig_c.set_size_inches(figsize_small)
        plt.tight_layout()
        safe_label = label.lower()
        plt.savefig(
            os.path.join(fig_dir, f"SHAP_{safe}_{safe_label}.png"), dpi=dpi
        )
        plt.close("all")
    logger.info(f"    Salvati SHAP per classe ({model_name}).")
