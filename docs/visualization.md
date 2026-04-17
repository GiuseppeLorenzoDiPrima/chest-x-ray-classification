# plot/visualization.py

Tutti i grafici del progetto seguono lo stile IEEE-ready definito in `setup_publication_style()`. I file vengono salvati in sottocartelle di `outs/imgs/` con DPI e dimensioni configurabili.

---

## Stile IEEE

### `setup_publication_style`

```python
setup_publication_style(config: dict)
```

Imposta globalmente i parametri `rcParams` di matplotlib per grafici adatti a pubblicazioni scientifiche.

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `config` | `dict` | Configurazione YAML (legge `visualization.dpi`) |

**Palette colori:**

```python
IEEE_PALETTE = [
    "#2166AC",  # blu
    "#B2182B",  # rosso scuro
    "#1B7837",  # verde foresta
    "#D6604D",  # arancione mattone
    "#762A83",  # viola
    "#4D4D4D",  # grigio antracite
]
```

**Impostazioni applicate:** font serif, dimensioni coerenti (11/13 pt), spines top/right rimossi, griglia tratteggiata con alpha 0.7, colori testo `#333333`.

---

### `_class_labels`

```python
_class_labels(classification_type: str) -> list[str]
```

Restituisce le etichette di classe in base al tipo di classificazione.

| `classification_type` | Output |
|-----------------------|--------|
| `"binary"` | `["Normal", "Pneumonia"]` |
| qualsiasi altro | `["Bacteria", "Normal", "Virus"]` |

---

## Pre-processing

### `plot_class_distribution`

```python
plot_class_distribution(data: dict, config: dict)
```

Genera grafici a barre della distribuzione delle classi per train, val, test e per l'intero dataset (Overall).

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `data` | `dict` | Output di `load_and_preprocess()` |
| `config` | `dict` | Configurazione YAML |

**File prodotti** in `outs/imgs/pre-processing/`:

| File | Descrizione |
|------|-------------|
| `class_distribution_training.png` | Distribuzione sul training set |
| `class_distribution_validation.png` | Distribuzione sul validation set |
| `class_distribution_test.png` | Distribuzione sul test set |
| `class_distribution_overall.png` | Distribuzione sull'intero dataset (train+val+test) |

---

### `plot_scree_graph`

```python
plot_scree_graph(pca, config: dict = None, show: bool = False)
```

Visualizza la varianza spiegata per componente PCA e la varianza cumulativa.

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `pca` | `sklearn.decomposition.PCA` | Oggetto PCA già fittato |
| `config` | `dict` | Configurazione YAML (opzionale; usa valori di default se `None`) |
| `show` | `bool` | Se `True` mostra la finestra interattiva (default `False`) |

**File prodotto** in `outs/imgs/pre-processing/`: `pca_scree.png`

---

## Training

### `plot_training_history`

```python
plot_training_history(
    training_metrics: list[dict],
    validation_metrics: list[dict],
    model_name: str,
    config: dict,
)
```

Genera un grafico per ogni metrica configurata (default: loss e accuracy) con le curve di train e validation per epoch.

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `training_metrics` | `list[dict]` | Lista di metriche per ogni epoca di training; ogni dict ha chiavi come `"loss"`, `"accuracy"` |
| `validation_metrics` | `list[dict]` | Lista di metriche per ogni epoca di validazione |
| `model_name` | `str` | Nome del modello (es. `"ResNet"`) — usato per il titolo e il nome file |
| `config` | `dict` | Configurazione YAML |

**File prodotti** in `outs/imgs/training/{model_name.lower()}/`:

| File | Descrizione |
|------|-------------|
| `{model_name}_loss.png` | Curva di loss per epoch |
| `{model_name}_accuracy.png` | Curva di accuracy per epoch |

---

## Valutazione

### `plot_confusion_matrix`

```python
plot_confusion_matrix(
    y_true,
    y_pred,
    labels: list[str],
    title: str,
    config: dict,
    filename: str,
    subdir: str = "confusion_matrix",
)
```

Salva la matrice di confusione come heatmap IEEE-style con griglia tra le celle.

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `y_true` | `array-like` | Etichette vere (`int`) |
| `y_pred` | `array-like` | Predizioni del modello (`int`) |
| `labels` | `list[str]` | Nomi delle classi (es. `["Normal", "Pneumonia"]`) |
| `title` | `str` | Titolo del grafico |
| `config` | `dict` | Configurazione YAML |
| `filename` | `str` | Nome del file di output (es. `"cm_resnet.png"`) |
| `subdir` | `str` | Sottocartella di `outs/imgs/` (default `"confusion_matrix"`) |

**File prodotto** in `outs/imgs/{subdir}/{filename}`.

---

### `plot_roc_curves`

```python
plot_roc_curves(
    y_true,
    y_score,
    labels: list[str],
    title: str,
    config: dict,
    filename: str,
    subdir: str = "roc_curves",
)
```

Genera le curve ROC one-vs-rest per ogni classe con il relativo AUC. Gestisce correttamente sia il caso binario (espandendo `label_binarize` da shape `(n, 1)` a `(n, 2)`) sia il caso multi-classe.

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `y_true` | `array-like` | Etichette vere (`int`) |
| `y_score` | `np.ndarray` | Probabilità per classe, shape `(n_samples, n_classes)` |
| `labels` | `list[str]` | Nomi delle classi |
| `title` | `str` | Titolo del grafico |
| `config` | `dict` | Configurazione YAML |
| `filename` | `str` | Nome del file di output (es. `"roc_resnet.png"`) |
| `subdir` | `str` | Sottocartella di `outs/imgs/` (default `"roc_curves"`) |

**File prodotto** in `outs/imgs/{subdir}/{filename}`.

---

## Confronto modelli

### `plot_metrics_comparison`

```python
plot_metrics_comparison(comparison: pd.DataFrame, config: dict)
```

Genera un grafico a barre orizzontali per ogni metrica, un file per metrica.

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `comparison` | `pd.DataFrame` | Output di `build_comparison_table()` |
| `config` | `dict` | Configurazione YAML (legge `visualization.comparison_metrics`) |

**File prodotti** in `outs/imgs/model_comparison/`: `model_{metric}_comparison.png` per ogni metrica.

---

### `plot_metric_groups_comparison`

```python
plot_metric_groups_comparison(comparison: pd.DataFrame, config: dict)
```

Genera un grafico a barre raggruppate con tutti i modelli sull'asse X e una barra per ogni metrica.

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `comparison` | `pd.DataFrame` | Output di `build_comparison_table()` |
| `config` | `dict` | Configurazione YAML (legge `visualization.comparison_group_metrics`) |

**File prodotto** in `outs/imgs/model_comparison/`: `model_comparison_groups.png`.

---

## Uncertainty (MC Dropout)

### `plot_uncertainty_results`

```python
plot_uncertainty_results(
    mc_results: dict,
    y_test: np.ndarray,
    model_name: str,
    config: dict,
)
```

Genera tre grafici di uncertainty quantification a partire dai risultati MC Dropout.

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `mc_results` | `dict` | Output di `mc_dropout_predict()` |
| `y_test` | `np.ndarray` | Etichette vere del test set, shape `(n_samples,)` |
| `model_name` | `str` | Nome del modello — usato per titolo e sottocartella |
| `config` | `dict` | Configurazione YAML |

**File prodotti** in `outs/imgs/uncertainty/{model_name.lower()}/`:

| File | Descrizione |
|------|-------------|
| `uncertainty_entropy_{model}.png` | Istogramma densità dell'entropia: predizioni corrette vs errate |
| `rejection_curve_{model}.png` | Accuracy in funzione della percentuale di eventi accettati |
| `uncertainty_per_class_{model}.png` | Boxplot dell'entropia per ogni classe |

---

## Interpretabilità (SHAP)

### `plot_shap_results`

```python
plot_shap_results(
    shap_values,
    X_sample: np.ndarray,
    feature_names: list[str],
    class_labels: list[str],
    model_name: str,
    fig_dir: str,
    dpi: int,
    figsize: tuple,
)
```

Genera summary plot, bar plot di importanza media e plot per singola classe usando la libreria `shap`.

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `shap_values` | `list[np.ndarray]` o `np.ndarray` | Valori SHAP (lista di array per multi-classe, o array 3D `(n, features, classes)`) |
| `X_sample` | `np.ndarray` | Feature campionate, shape `(n_samples, n_features)` |
| `feature_names` | `list[str]` | Etichette delle feature (es. `["PC1", "PC2", ...]`) |
| `class_labels` | `list[str]` | Etichette delle classi |
| `model_name` | `str` | Nome del modello (usato nei titoli e nomi file) |
| `fig_dir` | `str` | Cartella di destinazione per i file PNG |
| `dpi` | `int` | Risoluzione dei grafici |
| `figsize` | `tuple` | Dimensioni in pollici `(width, height)` |

**File prodotti** in `fig_dir`:

| File | Descrizione |
|------|-------------|
| `SHAP_summary_{model}.png` | Summary plot aggregato su tutte le classi |
| `SHAP_bar_{model}.png` | Bar plot: importanza media assoluta per feature |
| `SHAP_{model}_{class}.png` | Summary plot per ogni singola classe |

---

## Cartelle di output

```
outs/imgs/
├── pre-processing/        ← distribuzione classi, PCA scree
├── training/
│   ├── resnet/            ← loss/accuracy per epoch (ResNet)
│   └── alexnet/           ← loss/accuracy per epoch (AlexNet)
├── confusion_matrix/      ← cm_*.png per ogni modello
├── roc_curves/            ← roc_*.png per ogni modello
├── model_comparison/      ← confronto metriche tra modelli
├── uncertainty/
│   ├── resnet/            ← grafici MC Dropout per ResNet
│   └── alexnet/           ← grafici MC Dropout per AlexNet
└── interpretability/      ← SHAP summary, bar, per-class
```
