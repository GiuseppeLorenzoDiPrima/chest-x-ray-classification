# utils/

Moduli di valutazione, interpretabilità e uncertainty quantification.

---

## evaluation.py

### `compute_metrics`

```python
compute_metrics(y_true, y_pred) -> dict
```

Calcola le metriche di classificazione con `average="macro"`.

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `y_true` | `array-like` | Etichette vere |
| `y_pred` | `array-like` | Predizioni del modello |

**Restituisce:**

```python
{
    "accuracy":  float,  # accuracy_score
    "precision": float,  # precision_score(average="macro", zero_division=0)
    "recall":    float,  # recall_score(average="macro", zero_division=0)
    "f1":        float,  # f1_score(average="macro", zero_division=0)
}
```

---

### `evaluate_dl_model`

```python
evaluate_dl_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]
```

Valuta un modello Deep Learning su un DataLoader in modalità `eval()` (no gradient).

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `model` | `nn.Module` | Modello da valutare (ResNet o AlexNet) |
| `dataloader` | `DataLoader` | DataLoader dello split da valutare |
| `criterion` | `nn.Module` | Funzione di loss (es. `CrossEntropyLoss`) |
| `device` | `torch.device` | Device su cui spostare tensori e modello |

**Restituisce:** `(metrics, confusion_matrix, y_true, y_pred)`

| Elemento | Tipo | Descrizione |
|----------|------|-------------|
| `metrics` | `dict` | Dizionario con `accuracy`, `precision`, `recall`, `f1`, `loss` |
| `confusion_matrix` | `np.ndarray` | Matrice `(n_classes, n_classes)` |
| `y_true` | `np.ndarray` | Etichette vere, shape `(n_samples,)` |
| `y_pred` | `np.ndarray` | Predizioni, shape `(n_samples,)` |

---

### `build_comparison_table`

```python
build_comparison_table(all_results: dict) -> pd.DataFrame
```

Costruisce la tabella comparativa con una riga per modello.

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `all_results` | `dict` | Dizionario `{nome_modello: result_dict}` dove ogni `result_dict` ha una chiave `"metrics"` |

**Restituisce:** `pd.DataFrame` con colonne `Modello · accuracy · precision · recall · f1 · loss`.

---

### `generate_full_report`

```python
generate_full_report(
    all_results: dict,
    data: dict,
    config: dict,
) -> pd.DataFrame
```

Pipeline completa di reportistica finale: salva CSV, scrive classification reports, genera grafici di valutazione e confronto.

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `all_results` | `dict` | Dizionario `{nome_modello: result_dict}` con chiavi `y_true`, `y_pred`, `y_score`, `metrics` |
| `data` | `dict` | Output di `load_and_preprocess()` (usato per il tipo di classificazione) |
| `config` | `dict` | Configurazione YAML |

**File prodotti:**

| File | Descrizione |
|------|-------------|
| `outs/results/model_comparison.csv` | Tabella comparativa con tutte le metriche |
| `outs/results/classification_reports.txt` | Sklearn classification report per ogni modello |
| `outs/imgs/confusion_matrix/cm_*.png` | Matrice di confusione per ogni modello |
| `outs/imgs/roc_curves/roc_*.png` | Curve ROC one-vs-rest per ogni modello |
| `outs/imgs/model_comparison/model_*_comparison.png` | Confronto per singola metrica |
| `outs/imgs/model_comparison/model_comparison_groups.png` | Confronto complessivo |

**Restituisce:** `pd.DataFrame` — la tabella comparativa.

---

## interpretability.py

### `run_shap_analysis`

```python
run_shap_analysis(all_results: dict, data: dict, config: dict)
```

Esegue l'analisi SHAP per ogni modello SVM presente in `all_results`.

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `all_results` | `dict` | Dizionario con risultati dei modelli; vengono processati solo quelli con `"svm"` nel nome |
| `data` | `dict` | Output di `load_and_preprocess()` |
| `config` | `dict` | Configurazione YAML (legge `interpretability.shap_samples` e `interpretability.background_clusters`) |

**Funzionamento interno:**
1. Carica le feature PCA del test set dall'SVM già addestrato
2. Costruisce un background ridotto via `shap.kmeans` (k = `background_clusters`)
3. Usa `shap.KernelExplainer` per calcolare i valori SHAP su `shap_samples` campioni
4. Chiama `plot_shap_results()` per generare i grafici

**File prodotti** in `outs/imgs/interpretability/`:

| File | Descrizione |
|------|-------------|
| `SHAP_summary_svm.png` | Summary plot aggregato (importanza per classe) |
| `SHAP_bar_svm.png` | Bar plot: importanza media assoluta di ogni componente PCA |
| `SHAP_svm_{class}.png` | Summary plot per singola classe (Normal, Pneumonia, ecc.) |

---

## uncertainty.py

### `_enable_mc_dropout`

```python
_enable_mc_dropout(model: nn.Module)
```

Imposta il modello in modalità `eval()` tranne i layer `Dropout`, che vengono mantenuti in `train()`. Questo permette di avere predizioni stocastiche durante l'inferenza (MC Dropout).

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `model` | `nn.Module` | Modello DL (ResNet o AlexNet) |

---

### `mc_dropout_predict`

```python
mc_dropout_predict(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    n_iterations: int = 50,
) -> dict
```

Esegue `n_iterations` forward pass stocastici con dropout attivo e aggrega i risultati.

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `model` | `nn.Module` | Modello con dropout (ResNet o AlexNet) |
| `dataloader` | `DataLoader` | DataLoader del test set |
| `device` | `torch.device` | Device di esecuzione |
| `n_iterations` | `int` | Numero di forward pass stocastici (default `50`) |

**Restituisce:**

```python
{
    "mean_probs":  np.ndarray,  # (n_samples, n_classes) — probabilità medie sulle N iterazioni
    "std_probs":   np.ndarray,  # (n_samples, n_classes) — deviazione standard delle probabilità
    "predictions": np.ndarray,  # (n_samples,)           — classe con probabilità media più alta
    "entropy":     np.ndarray,  # (n_samples,)           — entropia predittiva: -Σ p·log(p+ε)
    "y_true":      np.ndarray,  # (n_samples,)           — etichette vere
}
```

---

### `run_uncertainty_analysis`

```python
run_uncertainty_analysis(dl_results: dict, data: dict, config: dict)
```

Applica MC Dropout a tutti i modelli DL presenti in `dl_results` e genera i relativi grafici di uncertainty.

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `dl_results` | `dict` | Sottoinsieme di `all_results` contenente solo modelli DL, con chiavi `"model"` e `"test_dl"` |
| `data` | `dict` | Output di `load_and_preprocess()` |
| `config` | `dict` | Configurazione YAML (legge `uncertainty.mc_dropout_iterations` e `uncertainty.enabled`) |

**Comportamento:** se `config["uncertainty"]["enabled"]` è `False`, la funzione termina immediatamente senza produrre output.

**File prodotti** (via `plot_uncertainty_results`) in `outs/imgs/uncertainty/{model_name}/`:

| File | Descrizione |
|------|-------------|
| `uncertainty_entropy_{model}.png` | Istogramma dell'entropia: predizioni corrette vs errate |
| `rejection_curve_{model}.png` | Accuracy vs percentuale eventi accettati (soglia entropia) |
| `uncertainty_per_class_{model}.png` | Boxplot dell'entropia per ogni classe |
