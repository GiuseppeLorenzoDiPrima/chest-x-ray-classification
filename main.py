"""
=============================================================================
Chest X-Ray Classification — Barbera & Di Prima
=============================================================================

Classificazione di radiografie toraciche con approcci Deep Learning e SVM:
  - Classificazione binaria:  NORMAL vs PNEUMONIA
  - Classificazione ternaria: BACTERIA vs NORMAL vs VIRUS

Pipeline suddivisa in 5 fasi:

  Fase 1: Caricamento dati e visualizzazione esplorativa
  Fase 2: Training modelli Deep Learning (ResNet, AlexNet)
  Fase 3: Training modello SVM (ViT embeddings + PCA + SMOTE)
  Fase 4: Interpretabilità (SHAP) e Uncertainty (MC Dropout)
  Fase 5: Valutazione finale e confronto tra modelli

Uso:
    python main.py                     # Pipeline completa
    python main.py --phase 1           # Solo una fase
    python main.py --phases 1 2 3      # Fasi selezionate
    python main.py --config my.yaml    # Configurazione custom
    python main.py --quick             # Run veloce (5 epoche, batch ridotto)
"""

import argparse
import logging
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")

import torch
import torch.nn as nn
from tabulate import tabulate

from data_classes.data_loader import load_config, load_and_preprocess
from models.resnet_model import build_resnet
from models.alexnet_model import build_alexnet
from models.svm_model import train_svm, evaluate_svm
from plot.visualization import (
    setup_publication_style,
    plot_class_distribution,
    plot_training_history,
    plot_confusion_matrix,
    plot_roc_curves,
)
from utils.evaluation import (
    compute_metrics,
    evaluate_dl_model,
    generate_full_report,
)
from utils.interpretability import run_shap_analysis
from utils.uncertainty import run_uncertainty_analysis


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(config: dict):
    """
    Console : solo messaggi del progetto, formato compatto.
    File    : tutto (librerie incluse), timestamp completo.
    """
    log_dir = config["paths"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # File handler
    fh = logging.FileHandler(
        os.path.join(log_dir, "run.log"), mode="w", encoding="utf-8"
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
    ))
    root.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    _PREFIXES = ("data_classes.", "models.", "utils.", "plot.")

    class _ProjectFilter(logging.Filter):
        def filter(self, record):
            return record.name in ("main", "__main__") or any(
                record.name.startswith(p) for p in _PREFIXES
            )

    ch.addFilter(_ProjectFilter())
    root.addHandler(ch)

    for noisy in ("transformers", "PIL", "matplotlib", "imblearn", "shap"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Chest X-Ray Classification — ML Pipeline"
    )
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Percorso al file di configurazione YAML")
    parser.add_argument("--phase", type=int, default=None,
                        help="Esegui solo una fase specifica (1-5)")
    parser.add_argument("--phases", type=int, nargs="+", default=None,
                        help="Esegui le fasi specificate (es. --phases 1 2 3)")
    parser.add_argument("--quick", action="store_true",
                        help="Run veloce: 5 epoche, batch 64")
    return parser.parse_args()


def _should_run(phase: int, args) -> bool:
    if args.phase is not None:
        return phase == args.phase
    if args.phases is not None:
        return phase in args.phases
    return True


# ---------------------------------------------------------------------------
# Utilità DL
# ---------------------------------------------------------------------------

def _get_device(config: dict) -> torch.device:
    dev_str = config["training"]["device"]
    if dev_str == "cuda" and not torch.cuda.is_available():
        dev_str = "cpu"
    device = torch.device(dev_str)
    return device


def _build_dataloader(dataset, config: dict, shuffle: bool,
                      weighted_sampler: bool = False):
    batch_size = config["deep_learning"]["batch_size"]
    sampler    = None

    if weighted_sampler and shuffle:
        targets       = torch.tensor(dataset.targets)
        class_counts  = torch.bincount(targets)
        class_weights = 1.0 / class_counts.float()
        sample_weights = (
            class_weights[dataset.targets]
            * config["deep_learning"]["oversampling_strength"]
        )
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True
        )
        shuffle = False

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
    )


def _train_one_epoch(model, dataloader, criterion, optimizer,
                     scheduler, device) -> dict:
    """Training di una singola epoca."""
    model.train()
    running_loss = 0.0
    y_pred_list, y_true_list = [], []

    from tqdm import tqdm
    for batch in tqdm(dataloader, desc="  Train", leave=False):
        images  = batch["image"].to(device)
        labels  = batch["label"].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        y_pred_list.extend(preds.cpu().numpy())
        y_true_list.extend(labels.cpu().numpy())

    metrics = compute_metrics(y_true_list, y_pred_list)
    metrics["loss"] = running_loss / len(dataloader)
    return metrics


def _train_dl_model(model, model_name: str, config: dict,
                    train_dl, val_dl, device) -> tuple[dict, list, list]:
    """
    Loop di training con early stopping.
    Restituisce (best_val_metrics, training_history, validation_history).
    """
    eval_metric  = config["training"]["evaluation_metric"]
    lower_better = config["training"]["best_metric_lower_is_better"]
    es_metric    = config["training"]["early_stopping_metric"]
    patience     = config["training"]["early_stopping_patience"]
    n_epochs     = config["deep_learning"]["epochs"]
    lr           = config["training"]["learning_rate"]
    opt_name     = config["deep_learning"]["optimizer"].lower()
    warmup_ratio = config["deep_learning"]["warmup_ratio"]

    if opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    total_steps  = len(train_dl) * n_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler    = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: (
            s / warmup_steps if s < warmup_steps
            else max(0.0, (total_steps - s) / (total_steps - warmup_steps))
        ),
    )
    criterion = nn.CrossEntropyLoss()

    best_val = float("inf") if lower_better else float("-inf")
    best_model_state = None
    no_improve = 0
    train_hist, val_hist = [], []

    for epoch in range(n_epochs):
        logger.info(f"  {model_name} — Epoch {epoch + 1}/{n_epochs}")
        train_m = _train_one_epoch(model, train_dl, criterion, optimizer,
                                    scheduler, device)
        val_m, _, _, _ = evaluate_dl_model(model, val_dl, criterion, device)
        train_hist.append(train_m)
        val_hist.append(val_m)

        logger.info(
            f"    train loss {train_m['loss']:.4f}  acc {train_m['accuracy']:.4f} | "
            f"val loss {val_m['loss']:.4f}  acc {val_m['accuracy']:.4f}"
        )

        # Aggiorna miglior modello
        current = val_m[eval_metric]
        is_best = (current < best_val) if lower_better else (current > best_val)
        if is_best:
            best_val       = current
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
            logger.info(f"    ✓ Nuovo miglior modello (val {eval_metric}: {best_val:.4f})")
        else:
            no_improve += 1
            if val_m[es_metric] < val_hist[-2][es_metric] if len(val_hist) > 1 else True:
                no_improve = 0

        # Early stopping
        if no_improve >= patience:
            logger.info(f"    Early stopping alla epoch {epoch + 1}.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    best_val_metrics = val_hist[
        min(range(len(val_hist)),
            key=lambda i: val_hist[i][eval_metric]
            if lower_better else -val_hist[i][eval_metric])
    ]
    return best_val_metrics, train_hist, val_hist


# ---------------------------------------------------------------------------
# Fase 2: Deep Learning
# ---------------------------------------------------------------------------

def run_phase2_dl(data: dict, config: dict,
                  device: torch.device, all_results: dict):
    """
    Addestra i modelli DL presenti in config.models.to_train.
    Salva pesi in models_dir, aggiunge risultati ad all_results.
    """
    models_dir = config["paths"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    to_train   = config["models"]["to_train"]
    criterion  = nn.CrossEntropyLoss()

    train_dl = _build_dataloader(data["train_dataset"], config,
                                  shuffle=True, weighted_sampler=True)
    val_dl   = _build_dataloader(data["val_dataset"],   config, shuffle=False)
    test_dl  = _build_dataloader(data["test_dataset"],  config, shuffle=False)

    for model_name in to_train:
        if model_name.lower() == "resnet":
            model = build_resnet(config).to(device)
        elif model_name.lower() == "alexnet":
            model = build_alexnet(config).to(device)
        else:
            continue

        print()
        logger.info(f"{'=' * 50}")
        logger.info(f"Training {model_name}...")
        logger.info(f"{'=' * 50}")

        best_val_m, train_hist, val_hist = _train_dl_model(
            model, model_name, config, train_dl, val_dl, device
        )

        # Grafici training
        if config["visualization"].get("graph", True):
            plot_training_history(train_hist, val_hist, model_name, config)

        # Valutazione test
        test_m, cm, y_true, y_pred = evaluate_dl_model(
            model, test_dl, criterion, device
        )

        y_score = None
        softmax = nn.Softmax(dim=1)
        model.eval()
        probs_list = []
        with torch.no_grad():
            for batch in test_dl:
                out  = model(batch["image"].to(device))
                probs_list.append(softmax(out).cpu().numpy())
        import numpy as np
        y_score = np.concatenate(probs_list, axis=0)

        logger.info(
            f"{model_name} test — "
            f"acc {test_m['accuracy']:.4f}  "
            f"f1 {test_m['f1']:.4f}  "
            f"loss {test_m['loss']:.4f}"
        )

        # Salva pesi
        save_path = os.path.join(models_dir, f"{model_name}_best_model.pt")
        torch.save(model.state_dict(), save_path)
        logger.info(f"Modello salvato in {save_path}")

        all_results[model_name] = {
            "metrics":          test_m,
            "confusion_matrix": cm,
            "y_true":           y_true,
            "y_pred":           y_pred,
            "y_score":          y_score,
            "model":            model,
            "test_dl":          test_dl,
            "train_history":    train_hist,
            "val_history":      val_hist,
        }


# ---------------------------------------------------------------------------
# Fase 3: SVM
# ---------------------------------------------------------------------------

def run_phase3_svm(data: dict, config: dict, all_results: dict):
    """Addestra il modello SVM e valuta sul test set."""
    to_train = config["models"]["to_train"]
    if not any("svm" in m.lower() for m in to_train):
        return

    print()
    logger.info("=" * 50)
    logger.info("Training SVM...")
    logger.info("=" * 50)

    svm_train_res = train_svm(data, config)
    svm_eval_res  = evaluate_svm(data, config, svm_model=svm_train_res["model"])

    logger.info(
        f"SVM test — "
        f"acc {svm_eval_res['metrics']['accuracy']:.4f}  "
        f"f1 {svm_eval_res['metrics']['f1']:.4f}  "
        f"loss {svm_eval_res['metrics']['loss']:.4f}"
    )

    all_results["SVM"] = {
        "metrics":          svm_eval_res["metrics"],
        "confusion_matrix": svm_eval_res["confusion_matrix"],
        "y_true":           svm_eval_res["y_true"],
        "y_pred":           svm_eval_res["y_pred"],
        "y_score":          svm_eval_res["y_score"],
        "model":            svm_eval_res["model"],
        "test_svm":         svm_train_res["test_svm"],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

logger = logging.getLogger("main")


def main():
    args   = parse_args()
    config = load_config(args.config)

    if args.quick:
        config["deep_learning"]["epochs"]                  = 5
        config["deep_learning"]["batch_size"]              = 64
        config["uncertainty"]["mc_dropout_iterations"]     = 10
        config["interpretability"]["shap_samples"]         = 50

    setup_logging(config)
    device = _get_device(config)

    logger.info("=" * 60)
    logger.info("  CHEST X-RAY CLASSIFICATION — Barbera & Di Prima")
    cls_type = config["classification"]["type"]
    logger.info(f"  Tipo di classificazione: {cls_type.upper()}")
    if torch.cuda.is_available():
        logger.info(f"  Device: {torch.cuda.get_device_name()}")
    else:
        logger.info("  Device: CPU")
    logger.info("=" * 60)

    t_start     = time.time()
    all_results = {}

    # ================================================================
    # FASE 1: Caricamento dati e visualizzazione
    # ================================================================
    if _should_run(1, args):
        print()
        logger.info("=" * 55)
        logger.info("FASE 1: Caricamento dati e visualizzazione")
        logger.info("=" * 55)

        data = load_and_preprocess(config)
        setup_publication_style(config)

        if config["visualization"].get("graph", True):
            print()
            logger.info("Generazione visualizzazioni esplorative...")
            plot_class_distribution(data, config)

        print()
        logger.info("Fase 1 completata.")
    else:
        data = load_and_preprocess(config)

    # ================================================================
    # FASE 2: Deep Learning
    # ================================================================
    if _should_run(2, args):
        print()
        logger.info("=" * 55)
        logger.info("FASE 2: Training modelli Deep Learning")
        logger.info("=" * 55)

        run_phase2_dl(data, config, device, all_results)

        print()
        logger.info("Fase 2 completata.")

    # ================================================================
    # FASE 3: SVM
    # ================================================================
    if _should_run(3, args):
        print()
        logger.info("=" * 55)
        logger.info("FASE 3: Training modello SVM")
        logger.info("=" * 55)

        run_phase3_svm(data, config, all_results)

        print()
        logger.info("Fase 3 completata.")

    # ================================================================
    # FASE 4: Interpretabilità e Uncertainty
    # ================================================================
    if _should_run(4, args):
        print()
        logger.info("=" * 55)
        logger.info("FASE 4: Interpretabilità e Uncertainty")
        logger.info("=" * 55)

        # SHAP (SVM)
        run_shap_analysis(all_results, data, config)

        # MC Dropout (DL)
        dl_results = {
            k: v for k, v in all_results.items()
            if "svm" not in k.lower() and "model" in v and "test_dl" in v
        }
        if dl_results:
            run_uncertainty_analysis(dl_results, data, config)

        print()
        logger.info("Fase 4 completata.")

    # ================================================================
    # FASE 5: Report finale
    # ================================================================
    if _should_run(5, args) and all_results:
        print()
        logger.info("=" * 55)
        logger.info("FASE 5: Valutazione finale e confronto")
        logger.info("=" * 55)

        comparison = generate_full_report(all_results, data, config)

        print()
        logger.info("=" * 55)
        logger.info("TABELLA RIEPILOGATIVA DEI RISULTATI")
        logger.info("=" * 55)
        print()
        print(tabulate(
            comparison, headers="keys", tablefmt="grid",
            floatfmt=".4f", showindex=False, missingval="-"
        ))
        print()
        logger.info("Fase 5 completata.")

    # ================================================================
    elapsed = time.time() - t_start
    print()
    logger.info(f"Pipeline completata in {elapsed:.1f} s.")
    logger.info(f"Output salvati in: {config['paths']['output_dir']}/")


if __name__ == "__main__":
    main()
