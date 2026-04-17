# Chest X-Ray Classification

Classificazione di radiografie toraciche pediatriche tramite Deep Learning e SVM,
con confronto sistematico tra modelli, analisi di interpretabilità (SHAP)
e quantificazione dell'incertezza (MC Dropout).

| | |
|---|---|
| **Autori** | Barbera Antonino · Di Prima Giuseppe Lorenzo |
| **Corso** | Machine Learning @ UniKore |
| **Dataset** | [Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |
| **Licenza** | [MIT](LICENSE) |

---

## Indice

- [Introduzione](#introduzione)
- [Requisiti](#requisiti)
- [Installazione](#installazione)
- [Dataset](#dataset)
- [Struttura del progetto](#struttura-del-progetto)
- [Configurazione](#configurazione)
- [Utilizzo](#utilizzo)
- [Risultati](#risultati)
- [Documentazione](#documentazione)
- [Licenza](#licenza)

---

## Introduzione

Il progetto si propone di classificare immagini radiografiche del torace in:

- **Classificazione binaria** — NORMAL vs PNEUMONIA (2 classi)
- **Classificazione ternaria** — BACTERIA vs NORMAL vs VIRUS (3 classi)

Le immagini provengono da pazienti pediatrici (1–5 anni) del Guangzhou Women and
Children's Medical Center, Guangzhou. Il dataset conta circa 5.856 immagini JPEG
suddivise in train, val e test.

I modelli confrontati sono:

| Modello | Tipo | Note |
|---------|------|------|
| **ResNet** | Deep Learning (CNN) | Architettura residua personalizzata |
| **AlexNet** | Deep Learning (CNN) | Architettura classica multi-layer |
| **SVM** | Machine Learning classico | ViT embeddings + PCA + SMOTE + RBF kernel |

La pipeline include inoltre:
- Analisi **SHAP** sull'SVM per l'interpretabilità delle componenti PCA
- **MC Dropout** sui modelli DL per la quantificazione dell'incertezza

---

## Requisiti

Il progetto richiede **Python ≥ 3.11**.

Le dipendenze principali sono:

```
torch · torchvision · transformers · scikit-learn · imbalanced-learn
shap · matplotlib · seaborn · kaggle · pyyaml · tqdm · tabulate
```

L'elenco completo è in [`requirements.txt`](requirements.txt).

---

## Installazione

### Windows

```bat
prepare.bat
```

### Linux / macOS

```bash
bash prepare.sh
```

Entrambi gli script eseguono automaticamente:
1. Creazione del virtual environment `.venv`
2. Aggiornamento di pip
3. Installazione dei requirements
4. Reinstallazione di PyTorch con supporto CUDA 12.8

> [!IMPORTANT]
> Per il download automatico del dataset è necessario configurare le credenziali
> Kaggle. Crea il file `~/.kaggle/kaggle.json` con il tuo API token
> ([istruzioni](https://www.kaggle.com/docs/api)).

> [!TIP]
> Per velocizzare il download dei pesi ViT da Hugging Face (usati dall'SVM per
> l'estrazione degli embeddings), puoi autenticarti con un token HF:
>
> 1. Crea un token su [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
>    (tipo: *Read*)
> 2. Esegui da terminale (con il venv attivo):
>    ```bash
>    huggingface-cli login
>    ```
> 3. Incolla il token quando richiesto.
>
> Senza autenticazione il download funziona comunque, ma potrebbe essere soggetto
> a rate limiting.

---

## Dataset

Il dataset viene scaricato **automaticamente** alla prima esecuzione tramite
le Kaggle API (`paultimothymooney/chest-xray-pneumonia`) e salvato in `data/chest_xray/`.

La struttura originale del dataset è:

```
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/      ← contiene sia file bacteria_* che virus_*
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Il preprocessore converte automaticamente la struttura in base al tipo di
classificazione scelto:

- **Binaria** — mantiene `NORMAL` e `PNEUMONIA` (2 classi)
- **Ternaria** — spezza `PNEUMONIA` in `BACTERIA` e `VIRUS` (3 classi)

La conversione è **reversibile**: cambiando `classification.type` nel config,
la struttura viene riadattata alla successiva esecuzione.

---

## Struttura del progetto

```
Progetto_Barbera_DiPrima/
│
├── main.py                  # Entry point — pipeline a 5 fasi
├── requirements.txt
├── prepare.bat              # Setup automatico Windows
├── prepare.sh               # Setup automatico Linux/macOS
├── README.md
├── LICENSE
├── .gitignore
│
├── config/
│   └── config.yaml          # Configurazione centralizzata
│
├── data_classes/
│   └── data_loader.py       # Download Kaggle + preprocessing + splits
│
├── models/
│   ├── resnet_model.py      # Architettura ResNet + factory
│   ├── alexnet_model.py     # Architettura AlexNet + factory
│   └── svm_model.py         # ViT embeddings + PCA + SMOTE + SVM
│
├── utils/
│   ├── evaluation.py        # Metriche, tabella comparativa, report
│   ├── interpretability.py  # Analisi SHAP (SVM + componenti PCA)
│   └── uncertainty.py       # MC Dropout e uncertainty quantification
│
├── plot/
│   └── visualization.py     # Grafici IEEE-ready
│
├── data/                    # Dataset (generato automaticamente)
│
├── docs/                    # Documentazione del progetto
│
└── outs/
    ├── imgs/                # Grafici generati (.png)
    ├── models/              # Pesi salvati (*.pt, *.pkl, pca.joblib)
    ├── logs/                # Log di esecuzione (run.log)
    └── results/             # CSV e report testuali
```

---

## Configurazione

Tutte le impostazioni sono centralizzate in [`config/config.yaml`](config/config.yaml).

Le più rilevanti:

```yaml
# Tipo di classificazione
classification:
  type: "binary"   # "binary" o "ternary"

# Modelli da addestrare
models:
  to_train: ["ResNet", "AlexNet", "SVM"]

# Parametri deep learning
deep_learning:
  epochs:     20
  batch_size: 32
  optimizer:  "adam"

# SVM
svm:
  C:              1.0
  kernel:         "rbf"
  pca_components: 8

# Visualizzazione
visualization:
  graph: true    # genera i grafici
  show:  false   # mostra le finestre durante l'esecuzione
  dpi:   300
```

> [!CAUTION]
> Un modello addestrato in modalità **binaria** non può essere valutato su un
> dataset ternario e viceversa. Assicurati di mantenere coerente il valore di
> `classification.type` tra addestramento e valutazione.

---

## Utilizzo

### Pipeline completa

```bash
python main.py
```

### Singola fase

```bash
python main.py --phase 1    # Solo caricamento dati e visualizzazioni
python main.py --phase 2    # Solo training DL (ResNet, AlexNet)
python main.py --phase 3    # Solo training SVM
python main.py --phase 4    # Solo SHAP + MC Dropout
python main.py --phase 5    # Solo report finale
```

### Fasi multiple

```bash
python main.py --phases 1 2 3
```

### Configurazione custom

```bash
python main.py --config config/my_config.yaml
```

### Run veloce (debug)

```bash
python main.py --quick      # 5 epoche, batch 64, SHAP ridotto
```

---

Le 5 fasi della pipeline:

| Fase | Descrizione |
|------|-------------|
| **1** | Download e preprocessing dataset; distribuzione classi |
| **2** | Training e valutazione ResNet + AlexNet con early stopping |
| **3** | Estrazione ViT embeddings, PCA, SMOTE, training e valutazione SVM |
| **4** | Analisi SHAP sull'SVM; MC Dropout su ResNet e AlexNet |
| **5** | Tabella comparativa, CSV, report, grafici di confronto |

---

## Risultati

Al termine dell'esecuzione vengono generati in `outs/`:

| Cartella | Contenuto |
|----------|-----------|
| `imgs/pre-processing/` | Distribuzione classi, PCA scree plot |
| `imgs/training/` | Loss/accuracy per epoch per ogni modello DL |
| `imgs/confusion_matrix/` | Matrici di confusione |
| `imgs/roc_curves/` | Curve ROC one-vs-rest |
| `imgs/model_comparison/` | Grafici comparativi tra modelli |
| `imgs/uncertainty/` | Entropia MC Dropout, rejection curve, boxplot |
| `imgs/interpretability/` | SHAP summary, bar plot, plot per classe |
| `models/` | Pesi `.pt` (ResNet, AlexNet), `SVM_best_model.pkl`, `pca.joblib` |
| `results/` | `model_comparison.csv`, `classification_reports.txt` |
| `logs/` | `run.log` con tutto il dettaglio dell'esecuzione |

---

## Documentazione

La documentazione tecnica dettagliata dei moduli si trova in [`docs/`](docs/):

- [`docs/data_loader.md`](docs/data_loader.md) — Preprocessing e gestione dataset
- [`docs/models.md`](docs/models.md) — Architetture e pipeline di training
- [`docs/visualization.md`](docs/visualization.md) — Grafici e stile IEEE
- [`docs/evaluation.md`](docs/evaluation.md) — Metriche e report

---

## Licenza

Questo progetto è distribuito con licenza MIT. Consulta il file [`LICENSE`](LICENSE)
per i dettagli.
