# models/

Architetture e pipeline di training per i tre modelli del progetto: ResNet, AlexNet e SVM.

---

## ResNet (`resnet_model.py`)

Implementazione personalizzata di una rete residua con connessioni skip configurabili.

---

### `ResidualBlock`

```python
ResidualBlock(in_channels: int, out_channels: int, stride: int = 1)
```

Blocco base della rete residua con due strati convoluzionali 3×3.

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `in_channels` | `int` | Numero di canali in ingresso |
| `out_channels` | `int` | Numero di canali in uscita |
| `stride` | `int` | Stride della prima convoluzione (default `1`) |

**Architettura interna:**
```
Conv2d(3×3) → BatchNorm2d → ReLU → Conv2d(3×3) → BatchNorm2d
      ↕ shortcut (1×1 Conv se stride ≠ 1 o canali cambiano)
ReLU
```

**`forward(x) → Tensor`**

| Parametro | Tipo | Shape |
|-----------|------|-------|
| `x` | `torch.Tensor` | `(B, in_channels, H, W)` |

**Restituisce:** `torch.Tensor` — `(B, out_channels, H', W')`

---

### `ResNet`

```python
ResNet(num_classes: int, layers: list[int], dropout: float)
```

Rete residua completa. Segue la struttura ResNet con 4 gruppi di blocchi, Global Average Pooling e layer fully-connected finale.

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `num_classes` | `int` | Numero di classi in output (`2` binario, `3` ternario) |
| `layers` | `list[int]` | Numero di blocchi per gruppo (es. `[3, 4, 6, 3]` per ResNet-50) |
| `dropout` | `float` | Probabilità di dropout prima del layer FC |

**`forward(x) → Tensor`**

| Parametro | Shape | Descrizione |
|-----------|-------|-------------|
| `x` | `(B, 3, 224, 224)` | Batch di immagini normalizzate |

**Restituisce:** `torch.Tensor` — `(B, num_classes)` — logit non normalizzati.

---

### `build_resnet`

```python
build_resnet(config: dict) -> ResNet
```

Factory che istanzia `ResNet` leggendo la sezione `resnet` del config YAML.

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `config` | `dict` | Configurazione YAML caricata da `load_config()` |

**Restituisce:** istanza di `ResNet` inizializzata con i parametri del config.

---

## AlexNet (`alexnet_model.py`)

Architettura classica con 5 layer convoluzionali e 3 fully-connected.

---

### `AlexNet`

```python
AlexNet(num_classes: int, dropout: float)
```

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `num_classes` | `int` | Numero di classi in output |
| `dropout` | `float` | Probabilità di dropout nei layer FC (default `0.5`) |

**Architettura:**

```
Feature extractor:
  Conv(96, 11×11, s=4) → ReLU → MaxPool(3×3, s=2)
  Conv(256, 5×5, p=2)  → ReLU → MaxPool(3×3, s=2)
  Conv(384, 3×3, p=1)  → ReLU
  Conv(384, 3×3, p=1)  → ReLU
  Conv(256, 3×3, p=1)  → ReLU → MaxPool(3×3, s=2)

Classifier:
  AdaptiveAvgPool(6×6) → Flatten
  Dropout → FC(4096) → ReLU
  Dropout → FC(4096) → ReLU
  FC(num_classes)
```

**`forward(x) → Tensor`**

| Parametro | Shape | Descrizione |
|-----------|-------|-------------|
| `x` | `(B, 3, 224, 224)` | Batch di immagini normalizzate |

**Restituisce:** `torch.Tensor` — `(B, num_classes)` — logit non normalizzati.

---

### `build_alexnet`

```python
build_alexnet(config: dict) -> AlexNet
```

Factory che istanzia `AlexNet` leggendo la sezione `alexnet` del config YAML.

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `config` | `dict` | Configurazione YAML caricata da `load_config()` |

**Restituisce:** istanza di `AlexNet` inizializzata con i parametri del config.

---

## SVM (`svm_model.py`)

Pipeline in 4 passi: **ViT embeddings → PCA → SMOTE → SVM**.

---

### `VisionEmbeddings`

```python
VisionEmbeddings(model_name: str, device: torch.device)
```

Estrae embedding da immagini usando un Vision Transformer pre-addestrato da Hugging Face.

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `model_name` | `str` | Identificatore HuggingFace (default `"google/vit-base-patch16-224"`) |
| `device` | `torch.device` | Device su cui eseguire il modello (`cpu` o `cuda`) |

**Metodo `extract_all`**

```python
extract_all(
    train_dl: DataLoader,
    val_dl: DataLoader,
    test_dl: DataLoader,
    pca_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray,
           PCA]
```

Estrae embedding per tutti gli split, fitta PCA su train, applica SMOTE sul training set.

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `train_dl` | `DataLoader` | DataLoader del training set |
| `val_dl` | `DataLoader` | DataLoader del validation set |
| `test_dl` | `DataLoader` | DataLoader del test set |
| `pca_components` | `int` | Numero di componenti PCA da conservare |

**Restituisce:** `(X_train, y_train, X_val, y_val, X_test, y_test, pca)` dove le X hanno shape `(n_samples, pca_components)`.

---

**Metodo `extract_single`**

```python
extract_single(
    dataloader: DataLoader,
    pca: PCA,
) -> tuple[np.ndarray, np.ndarray]
```

Estrae embedding per un singolo split usando una PCA già fittata.

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `dataloader` | `DataLoader` | DataLoader dello split da processare |
| `pca` | `PCA` | Trasformazione PCA già fittata su train |

**Restituisce:** `(X, y)` — array di feature ridotte e relative etichette.

---

### `train_svm`

```python
train_svm(data: dict, config: dict) -> dict
```

Esegue la pipeline completa di training SVM.

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `data` | `dict` | Output di `load_and_preprocess()` |
| `config` | `dict` | Configurazione YAML |

**Salva su disco:** `SVM_best_model.pkl` e `pca.joblib` in `config["paths"]["models_dir"]`.

**Restituisce:**

```python
{
    "model":    SVC,           # modello addestrato
    "pca":      PCA,           # trasformazione PCA fittata su train
    "test_svm": tuple,         # (X_test, y_test) per la valutazione
    "metrics":  dict,          # metriche sul validation set
}
```

---

### `evaluate_svm`

```python
evaluate_svm(data: dict, config: dict, svm_model: SVC = None) -> dict
```

Valuta il modello SVM sul test set. Se `svm_model` non è fornito, carica il modello da disco.

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `data` | `dict` | Output di `load_and_preprocess()` |
| `config` | `dict` | Configurazione YAML |
| `svm_model` | `SVC` | Modello già addestrato (opzionale; se `None` viene caricato da disco) |

**Restituisce:**

```python
{
    "metrics":          dict,         # accuracy, precision, recall, f1, loss
    "confusion_matrix": np.ndarray,   # (n_classes, n_classes)
    "y_true":           np.ndarray,   # etichette vere del test set
    "y_pred":           np.ndarray,   # predizioni del modello
    "y_score":          np.ndarray,   # probabilità per classe (n_samples, n_classes)
    "model":            SVC,          # il modello usato
}
```

---

## Training DL (in `main.py`)

Entrambi i modelli DL condividono lo stesso loop di training implementato in `main.py`.

### `_train_dl_model`

```python
_train_dl_model(
    model: nn.Module,
    model_name: str,
    config: dict,
    train_dl: DataLoader,
    val_dl: DataLoader,
    device: torch.device,
) -> tuple[dict, list[dict], list[dict]]
```

Loop di training con early stopping. Ad ogni epoca chiama `_train_one_epoch` e `evaluate_dl_model`, salva il miglior stato del modello in memoria.

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `model` | `nn.Module` | Modello da addestrare (ResNet o AlexNet) |
| `model_name` | `str` | Nome del modello (per log) |
| `config` | `dict` | Configurazione YAML |
| `train_dl` | `DataLoader` | DataLoader del training set |
| `val_dl` | `DataLoader` | DataLoader del validation set |
| `device` | `torch.device` | Device di esecuzione |

**Restituisce:** `(best_val_metrics, train_history, val_history)`

| Elemento | Tipo | Descrizione |
|----------|------|-------------|
| `best_val_metrics` | `dict` | Metriche della migliore epoca di validazione |
| `train_history` | `list[dict]` | Lista di metriche per ogni epoca (train) |
| `val_history` | `list[dict]` | Lista di metriche per ogni epoca (val) |

**Componenti del training:**

| Componente | Valore / Configurazione |
|-----------|------------------------|
| Loss | `CrossEntropyLoss` |
| Optimizer | `Adam` / `SGD` / `RMSprop` (da config) |
| Scheduler | `LambdaLR` con warmup lineare + decay lineare |
| Oversampling | `WeightedRandomSampler` (bilanciamento classi su train) |
| Early stopping | Pazienza configurabile su metrica scelta |
| Salvataggio | Miglior `state_dict` → `{model_name}_best_model.pt` |
