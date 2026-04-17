"""
Analisi di interpretabilità tramite SHAP values.

Applicata al modello SVM addestrato sugli embeddings PCA del ViT:
poiché il SVM opera su un vettore compatto di N componenti PCA,
è possibile calcolare SHAP values (TreeExplainer o KernelExplainer)
per quantificare il contributo di ciascun componente alla predizione.
"""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _pca_feature_names(n: int) -> list[str]:
    return [f"PC{i + 1}" for i in range(n)]


# ---------------------------------------------------------------------------
# Analisi SHAP
# ---------------------------------------------------------------------------

def run_shap_analysis(all_results: dict, data: dict, config: dict):
    """
    Esegue l'analisi SHAP per i modelli che dispongono delle features PCA.

    Per ogni modello SVM nei risultati, calcola:
    - Summary plot aggregato
    - Bar plot dell'importanza media assoluta
    - Plot per singola classe (classificazione ternaria)

    Parameters
    ----------
    all_results : {nome_modello: result_dict}
                  Il result_dict deve contenere 'model' e 'test_svm'
                  (istanza _SVMDataset con .features e .labels).
    data        : dizionario del dataset (non usato direttamente, per coerenza)
    config      : configurazione YAML
    """
    if not config["interpretability"].get("enabled", True):
        logger.info("Analisi SHAP disabilitata dalla configurazione.")
        return

    try:
        import shap  # type: ignore
    except ImportError:
        logger.warning("Pacchetto 'shap' non installato. Analisi SHAP saltata.")
        return

    cls_type    = config["classification"]["type"].lower()
    n_samples   = config["interpretability"]["shap_samples"]
    bg_clusters = config["interpretability"]["background_clusters"]
    fig_dir     = os.path.join(config["paths"]["figures_dir"], "interpretability")
    os.makedirs(fig_dir, exist_ok=True)
    dpi     = config["visualization"]["dpi"]
    figsize = tuple(config["visualization"]["figsize"])

    if cls_type == "binary":
        class_labels = ["Normal", "Pneumonia"]
    else:
        class_labels = ["Bacteria", "Normal", "Virus"]

    for model_name, res in all_results.items():
        if "svm" not in model_name.lower():
            continue

        svm_model = res.get("model")
        test_svm  = res.get("test_svm")

        if svm_model is None or test_svm is None:
            logger.warning(f"Modello o test_svm non disponibile per {model_name}. Skip.")
            continue

        X_test   = test_svm.features
        n_feats  = X_test.shape[1]
        feat_names = _pca_feature_names(n_feats)

        # Subsample per efficienza
        n_plot = min(n_samples, len(X_test))
        idx    = np.random.choice(len(X_test), n_plot, replace=False)
        X_sample = X_test[idx]

        logger.info(f"SHAP analysis per {model_name} ({n_plot} campioni)...")

        try:
            # Usa KernelExplainer con background ridotto tramite kmeans
            background = shap.kmeans(X_test, bg_clusters)
            explainer  = shap.KernelExplainer(svm_model.predict_proba, background)
            shap_values = explainer.shap_values(X_sample)
        except Exception as exc:
            logger.error(f"SHAP fallito per {model_name}: {exc}")
            continue

        from plot.visualization import plot_shap_results
        plot_shap_results(
            shap_values=shap_values,
            X_sample=X_sample,
            feature_names=feat_names,
            class_labels=class_labels,
            model_name=model_name,
            fig_dir=fig_dir,
            dpi=dpi,
            figsize=figsize,
        )

    print()
    logger.info("Analisi SHAP completata.")
    print()
