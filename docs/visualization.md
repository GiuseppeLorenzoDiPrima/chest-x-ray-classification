# plot/visualization.py

Tutti i grafici del progetto seguono lo stile IEEE-ready definito in `setup_publication_style()`.

---

## Stile IEEE

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

Impostazioni rcParams: font serif, dimensioni coerenti, spines top/right rimossi,
griglia tratteggiata, colori testo `#333333`.

---

## Funzioni grafiche

### Pre-processing

| Funzione | Output |
|----------|--------|
| `plot_class_distribution(data, config)` | Bar chart distribuzione classi per train/val/test |
| `plot_scree_graph(pca, config, show)` | Varianza spiegata + cumulativa per componente PCA |

### Training

| Funzione | Output |
|----------|--------|
| `plot_training_history(train_hist, val_hist, model_name, config)` | Loss e accuracy per epoch |

### Valutazione

| Funzione | Output |
|----------|--------|
| `plot_confusion_matrix(y_true, y_pred, labels, title, config, filename)` | Heatmap Blues con griglia |
| `plot_roc_curves(y_true, y_score, labels, title, config, filename)` | ROC one-vs-rest con AUC |

### Confronto modelli

| Funzione | Output |
|----------|--------|
| `plot_metrics_comparison(comparison, config)` | Barre orizzontali per ogni metrica |
| `plot_metric_groups_comparison(comparison, config)` | Barre raggruppate per modello |

### Uncertainty (MC Dropout)

| Funzione | Output |
|----------|--------|
| `plot_uncertainty_results(mc_results, y_test, model_name, config)` | Entropia, rejection curve, boxplot per classe |

### Interpretabilità (SHAP)

| Funzione | Output |
|----------|--------|
| `plot_shap_results(shap_values, X_sample, feature_names, class_labels, ...)` | Summary, bar plot, plot per classe |

---

## Cartelle di output

Tutti i grafici vengono salvati in sottocartelle di `outs/imgs/`:

```
outs/imgs/
├── pre-processing/
├── training/
├── confusion_matrix/
├── roc_curves/
├── model_comparison/
├── uncertainty/
└── interpretability/
```
