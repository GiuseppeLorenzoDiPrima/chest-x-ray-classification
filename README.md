# Chest X-Ray Classification
## Deep Learning e SVM applicati alla Diagnosi Medica

> Classificazione automatica di radiografie toraciche pediatriche in configurazione binaria (NORMAL / PNEUMONIA) e ternaria (BACTERIA / NORMAL / VIRUS) mediante reti neurali profonde (ResNet, AlexNet) e Support Vector Machine con rappresentazioni (embeddings) estratte da un Vision Transformer pre-addestrato.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![scikit--learn](https://img.shields.io/badge/scikit--learn-SVM-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

Pipeline sperimentale per la classificazione di radiografie toraciche, fondata su:
- ~5.856 immagini JPEG acquisite su pazienti pediatrici (1–5 anni, Guangzhou)
- Confronto sistematico tra approcci di Deep Learning e Support Vector Machine applicato agli embeddings estratti tramite ViT
- Modalità di classificazione **binaria** (NORMAL / PNEUMONIA) e **ternaria** (BACTERIA / NORMAL / VIRUS)

###### **Obiettivo: valutare e comparare quantitativamente le prestazioni dei modelli ResNet, AlexNet e SVM su dati radiologici reali. Analisi ulteriormente approfondita attraverso la valutazione dell'interpretabilità (SHAP) e dell'incertezza predittiva (MC Dropout)**.

##### Risultati ottenuti

###### Scenario di classificazione binaria

<div align="center">

<table>
  <tr>
    <th>Modello</th>
    <th>Accuracy</th>
    <th>Precision Macro</th>
    <th>Recall Macro</th>
    <th>F1-Score Macro</th>
    <th>Loss</th>
    <th>Train Time (s)</th>
  </tr>

  <tr>
    <td>ResNet</td>
    <td>0.9022</td>
    <td>0.9284</td>
    <td>0.8714</td>
    <td>0.8894</td>
    <td>0.4366</td>
    <td>1h 17min 51sec</td>
  </tr>

  <tr>
    <td>AlexNet</td>
    <td>0.8526</td>
    <td>0.8914</td>
    <td>0.8077</td>
    <td>0.8275</td>
    <td>0.5073</td>
    <td>1h 20min 43sec</td>
  </tr>

  <tr>
    <td>SVM (ViT + PCA + SMOTE)</td>
    <td>0.7981</td>
    <td>0.7844</td>
    <td>0.7897</td>
    <td>0.7867</td>
    <td>0.5769</td>
    <td>2min 45sec</td>
  </tr>

</table>

</div>

###### Scenario di classificazione ternaria

<div align="center">

<table>
  <tr>
    <th>Modello</th>
    <th>Accuracy</th>
    <th>Precision Macro</th>
    <th>Recall Macro</th>
    <th>F1-Score Macro</th>
    <th>Loss</th>
    <th>Train Time (s)</th>
  </tr>

  <tr>
    <td>ResNet</td>
    <td>0.7131</td>
    <td>0.7546</td>
    <td>0.7237</td>
    <td>0.6944</td>
    <td>0.9600</td>
    <td>1h 24min 55sec</td>
  </tr>

  <tr>
    <td>AlexNet</td>
    <td>0.8253</td>
    <td>0.8155</td>
    <td>0.8143</td>
    <td>0.8119</td>
    <td>0.5786</td>
    <td>1h 6min 10sec</td>
  </tr>

  <tr>
    <td>SVM (ViT + PCA + SMOTE)</td>
    <td>0.7292</td>
    <td>0.7169</td>
    <td>0.7189</td>
    <td>0.717</td>
    <td>0.7638</td>
    <td>2min 55sec</td>
  </tr>

</table>

</div>

###### Alcuni grafici significativi

<p align="center">
  <img src=".github/class_distribution_overall.png" height="220" width="45%">
  <img src=".github/pca_scree.png" height="220" width="45%">
</p>

<p align="center">
  <img src=".github/model_comparison_groups.png" height="220" width="45%">
  <img src=".github/SHAP_summary_svm.png" height="220" width="45%">
</p>

<p align="center">
  <em>Prima riga: distribuzione delle classi e PCA Scree Plot. Seconda riga: confronto tra modelli e SHAP Summary (SVM).</em>
</p>

##### Key Insights

- L'architettura residuale (ResNet) mitiga in modo più efficace il problema del vanishing gradient rispetto ad AlexNet, grazie all'impiego delle connessioni residuali (skip connection)
- L'SVM addestrato sulle rappresentazioni (embeddings) estratte tramite ViT consegue performance competitive limitando significativamente i tempi di addestramento (ViT pre-addestrato)
- Le componenti principali a maggiore varianza spiegata esibiscono il più elevato potere discriminante nella separazione tra classi

##### Configurabilità
Tutti i parametri sperimentali sono centralizzati e modificabili nel file `config/config.yaml`, senza necessità di intervenire sul codice sorgente.

---

## Introduzione

Benvenuta/o, il presente elaborato è stato sviluppato nell'ambito dell'esame di Machine Learning, previsto dal piano di studi del Corso di Laurea Magistrale in Ingegneria dell'Intelligenza Artificiale e Sicurezza Informatica presso l'Università degli Studi di Enna Kore.

Il lavoro propone un framework per la classificazione automatica di radiografie toraciche pediatriche mediante tecniche di Deep Learning e Machine Learning classico. Il sistema supporta due modalità operative: classificazione ternaria, finalizzata a distinguere le polmoniti batteriche (BACTERIA) da quelle virali (VIRUS) rispetto ai casi di soggetti sani (NORMAL); e classificazione binaria, volta a classificare tra soggetti sani (NORMAL) e pazienti affetti da polmonite (PNEUMONIA).

I dati impiegati provengono dal dataset pubblico [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), disponibile su Kaggle e raccolto presso il Guangzhou Women and Children's Medical Center. Il corpus comprende circa 5.856 immagini JPEG ripartite nei set di training, validazione e test.

Sono posti a confronto tre approcci metodologici: ResNet (rete neurale convoluzionale con connessioni residuali), AlexNet (architettura CNN classica) e SVM (Support Vector Machine addestrato sulle rappresentazioni estratte da un Vision Transformer pre-addestrato, riduzione dimensionale via PCA e bilanciamento del training set tramite SMOTE).

---

## Indice

1. [Prerequisiti](#1-prerequisiti)
2. [Installazione](#2-installazione)
3. [Esecuzione](#3-esecuzione)
4. [Struttura del progetto](#4-struttura-del-progetto)
5. [Le 5 fasi della pipeline](#5-le-5-fasi-della-pipeline)
6. [Configurazione](#6-configurazione)
7. [Output prodotti](#7-output-prodotti)
8. [Licenza MIT](#8-licenza-mit)
9. [Contatti](#9-contatti)

---

## 1. Prerequisiti

- **Python 3.11+** installato e disponibile da terminale (`python --version`)
- **Connessione internet** (per scaricare il dataset da Kaggle alla prima esecuzione)
- **GPU CUDA** raccomandata per il training DL (il codice viene eseguito in fallback su CPU)
- **Account Kaggle** gratuito (credenziali API per il download automatico del dataset)

Il dataset viene acquisito automaticamente mediante le API ufficiali Kaggle. Qualora il file **kaggle.json** non risulti ancora configurato nel proprio ambiente di lavoro, è necessario seguire la procedura riportata di seguito.

### Passo 1 — Creare un account Kaggle

Qualora non si disponga già di un account, è possibile registrarsi gratuitamente su [kaggle.com](https://www.kaggle.com/).

### Passo 2 — Generare la API Key

1. Creare una cartella al path:

```
C:\Users\<TUO_UTENTE>\.kaggle
```

2. Navigare alla pagina [kaggle.com/settings](https://www.kaggle.com/settings)
3. Scorrere fino alla sezione **API**
4. Cliccare su **"Create New Token"**
5. Conservare la chiave esadecimale appena generata

### Passo 3 — Configurare la API

6. Andare al path `C:\Users\<TUO_UTENTE>\.kaggle` e creare, al suo interno, un file `kaggle.json`
7. Inserire all'interno del file il seguente contenuto:

```json
{"username": "il_tuo_username", "key": "una_stringa_esadecimale"}
```

### Passo 4 — Verificare il funzionamento

Al termine della configurazione, è possibile verificare il corretto funzionamento eseguendo da terminale i seguenti comandi:

```bash
pip install kaggle
kaggle datasets list
```

La comparsa di una lista di dataset attesta la corretta configurazione delle credenziali.

>Qualora lo si preferisca, è possibile installare manualmente il dataset ed collocarlo al path: data\chest-xray

### Passo 6 (opzionale) — Token Hugging Face

Il modello SVM si avvale del Vision Transformer `google/vit-base-patch16-224` per l'estrazione di rappresentazioni dense (embeddings). Il download dei pesi avviene automaticamente da Hugging Face anche in assenza di autenticazione; tuttavia, il possesso di un token personale incrementa la velocità di trasferimento ed elimina il rischio di throttling da parte del servizio.

1. Qualora non si disponga già di un account, è possibile registrarsi gratuitamente su [Hugging Face.](https://huggingface.co)
2. Navigare alla pagina [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Creare un token *Read*
4. Eseguire da terminale (con il venv attivo):

```bash
hf auth login
```

5. Incollare il token copiato nella finestra di dialogo quando richiesto.

---

## 2. Installazione

### Clonazione della repository

Per prima cosa è necessario procedere con la clonazione della repository. Aprire un terminale, portarsi al path dove di vuole installare il progetto e clonare la repository.
Quanto sopra specificato si traduce in:
```bash
cd <tuo_percorso>
git clone https://www.github.com/GiuseppeLorenzoDiPrima/chest-x-ray-classification
```
Clonata la repository, sarà necessario installare le dipendenza secondo quanto meglio precisato nel seguito.

### Metodo rapido (Windows)

Il progetto include un file denominato **`prepare.bat`**. È sufficiente eseguire da terminale il seguente comando:

```bat
.\prepare.bat
```

In questo modo, lo script:

1. Crea un virtual environment in `.venv/`
2. Lo attiva
3. Installa tutte le dipendenze da `requirements.txt`
4. Reinstalla PyTorch con supporto CUDA 12.8

### Metodo manuale (Windows)

```bash
# Crea il virtual environment
python -m venv .venv

# Attiva il virtual environment
# Su Windows (cmd):
.venv\Scripts\activate
# Su Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Installa le dipendenze
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

### Metodo rapido (Linux / macOS)

```bash
bash prepare.sh
```

In questo modo, lo script:

1. Crea un virtual environment in `.venv/`
2. Lo attiva
3. Installa tutte le dipendenze da `requirements.txt`

### Metodo manuale (Linux / macOS)

```bash
# Crea il virtual environment
python -m venv .venv

# Attiva il virtual environment
source .venv/bin/activate

# Installa le dipendenze
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Esecuzione

> **Importante:** verificare che il virtual environment risulti attivo prima di lanciare qualsiasi comando. In assenza della cartella `.venv/` nella root del progetto, eseguire preventivamente la procedura di installazione descritta nella sezione "Metodo manuale" concorde al proprio sistema operativo presentata in precedenza.

```bash
# Su Windows (cmd):
.venv\Scripts\activate
# Su Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Su Linux/macOS:
source .venv/bin/activate
```

### Pipeline completa

Per avviare in sequenza l'intera pipeline (tutte e 5 le fasi) portarsi al path di installazione ed eseguire il comando `python main.py`.

```bash
cd <tuo_percorso>/chest-x-ray-classification
python main.py
```

### Pipeline veloce (consigliata per il primo test)

Esegue la pipeline completa limitando il training a 5 epoche. Raccomandata come test di sanità prima di procedere con l'addestramento definitivo.

```bash
cd <tuo_percorso>/chest-x-ray-classification
python main.py --quick
```

### Eseguire una singola fase

Dopo essersi portati al path di installazione attraverso:

```bash
cd <tuo_percorso>/chest-x-ray-classification
```

È possibile eseguire anche solo una specifica fase di addestramento.

```bash
python main.py --phase 1    # Solo caricamento dati e visualizzazione esplorativa
python main.py --phase 2    # Solo training Deep Learning (ResNet, AlexNet)
python main.py --phase 3    # Solo training SVM
python main.py --phase 4    # Solo interpretabilità (SHAP) e uncertainty (MC Dropout)
python main.py --phase 5    # Solo valutazione finale e confronto tra modelli
```

### Eseguire più fasi selezionate

Dopo essersi portati al path di installazione attraverso:

```bash
cd <tuo_percorso>/chest-x-ray-classification
```

È possibile eseguire più fasi di addestramento.

```bash
python main.py --phases 1 2 3    # Fasi 1, 2 e 3
```

### Usare una configurazione custom
Dopo essersi portati al path di installazione attraverso:

```bash
cd <tuo_percorso>/chest-x-ray-classification
```

È possibile eseguire in accordo ad una configurazione personalizzata di addestramento.

```bash
python main.py --config config/mia_config.yaml
```

---

## 4. Struttura del progetto

```
chest-x-ray-classification/
│
├── main.py                  # Entry point - pipeline a 5 fasi
├── requirements.txt         # Dipendenze Python
├── prepare.bat              # Script di setup automatico (Windows)
├── prepare.sh               # Script di setup automatico (Linux/macOS)
├── README.md                # Questo file
├── LICENSE                  # MIT License
├── .gitignore
│
├── config/
│   └── config.yaml          # Configurazione centralizzata
│
├── data_classes/
│   └── data_loader.py       # Download Kaggle + preprocessing + costruzione DataLoader
│
├── models/
│   ├── resnet_model.py      # Architettura ResNet con connessioni residuali
│   ├── alexnet_model.py     # Architettura AlexNet classica
│   └── svm_model.py         # ViT embeddings + PCA + SMOTE + SVM
│
├── utils/
│   ├── evaluation.py        # Metriche, tabella comparativa, report finale
│   ├── interpretability.py  # Analisi SHAP (KernelExplainer su SVM)
│   └── uncertainty.py       # MC Dropout e uncertainty quantification
│
├── plot/
│   └── visualization.py     # Grafici IEEE-ready
│
├── data/                    # Dataset (generato automaticamente al primo avvio)
│
├── docs/                    # Documentazione tecnica dei moduli
│
├── .github/                 # Immagini per questo file
│
└── outs/
    ├── imgs/                # Grafici generati (.png), suddivisi in sottocartelle
    ├── models/              # Pesi salvati (.pt, .pkl, pca.joblib)
    ├── logs/                # Log di esecuzione (run.log)
    └── results/             # CSV e report testuali
```

---

## 5. Le 5 fasi della pipeline

### Fase 1 — Caricamento dati e visualizzazione esplorativa

In questa fase il dataset viene scaricato da Kaggle (esclusivamente alla prima esecuzione) nella directory `data/chest_xray/`. Il sistema riorganizza automaticamente la struttura delle cartelle in funzione della modalità di classificazione selezionata (binaria o ternaria) e ridistribuisce i campioni tra training, validation e test set; tale ridistribuzione si rende necessaria in ragione del marcato sbilanciamento di classe presente nella suddivisione originale. La struttura è reversibile: la modifica del parametro `classification.type` nel file di configurazione `config/config.yaml` comporta la riesecuzione automatica della conversione alla successiva esecuzione.

### Fase 2 — Training modelli Deep Learning

Vengono addestrati i modelli di Deep Learning specificati in `models.to_train` (ResNet e/o AlexNet). Il bilanciamento delle classi durante il training è assicurato da un `WeightedRandomSampler` ponderato in base alla frequenza relativa di ciascuna classe. Il learning rate è gestito tramite uno scheduler con warmup lineare nella fase iniziale e decay lineare nelle epoche successive. È implementato l'early stopping sulla metrica di validazione, anch'essa configurabile in `config/config.yaml`. Il miglior checkpoint viene serializzato in `outs/models/` e per ogni modello vengono prodotti i grafici delle curve di training in una sottocartella dedicata.

### Fase 3 — Training modello SVM

Vengono estratte rappresentazioni dense a 768 dimensioni da ciascuna immagine mediante il Vision Transformer pre-addestrato `google/vit-base-patch16-224`. Alle rappresentazioni estratte viene applicata la Principal Component Analysis per la riduzione dimensionale (numero di componenti configurabile in `config/config.yaml`). Il training set viene successivamente bilanciato tramite SMOTE (Synthetic Minority Oversampling Technique). Ultimata la fase di preprocessing, si procede all'addestramento del classificatore SVM con kernel RBF. Il modello addestrato e la trasformazione PCA vengono serializzati e memorizzati in `outs/models/`.

### Fase 4 — Interpretabilità e Uncertainty

Viene condotta una **SHAP Analysis** (`KernelExplainer`) per identificare le componenti PCA che maggiormente determinano la predizione dell'SVM per ogni classe, producendo un summary plot aggregato, un bar plot di importanza media e un plot per singola classe. La quantificazione dell'incertezza predittiva è realizzata mediante **MC Dropout**, eseguendo N forward pass stocastici con dropout attivo (N configurabile in `config/config.yaml`) su ResNet e AlexNet per l'intero test set. Viene infine tracciata una **Rejection curve** che illustra la variazione dell'accuracy in funzione della soglia di entropia adottata per il filtraggio delle predizioni incerte. I grafici relativi all'incertezza sono organizzati in sottocartelle per modello.

### Fase 5 — Valutazione finale e confronto

Al termine di tutte le fasi di addestramento e valutazione, viene prodotta una tabella comparativa comprendente le seguenti metriche per ciascun modello: accuracy, precision (macro), recall (macro), F1-Score (macro) e loss. I file `model_comparison.csv` e `classification_reports.txt` vengono archiviati in `outs/results/`. Per ogni modello vengono prodotte la matrice di confusione e le curve ROC one-vs-rest, corredate da grafici di confronto complessivo a barre orizzontali per singola metrica e a barre raggruppate per modello.

---

## 6. Configurazione

Tutti i parametri sperimentali sono centralizzati in **`config/config.yaml`** e possono essere modificati senza intervenire sul codice sorgente.

| Sezione | Cosa controlla |
|---------|---------------|
| `paths` | Cartelle di input/output |
| `dataset` | Slug Kaggle, percentuale split train/val |
| `classification` | Tipo di classificazione (`binary` o `ternary`) |
| `models` | Modelli da addestrare |
| `deep_learning` | Epoche, batch size, optimizer, warmup, oversampling |
| `training` | Learning rate, early stopping, metrica di valutazione |
| `resnet` / `alexnet` | Iperparametri architettura |
| `svm` | C, kernel, numero componenti PCA |
| `interpretability` | Numero campioni SHAP, cluster di background |
| `uncertainty` | Iterazioni MC Dropout |
| `visualization` | DPI, dimensioni figure, metriche nei grafici |

### Esempio: cambiare tipo di classificazione

A titolo esemplificativo, di seguito sono riportate alcune delle modifiche più comuni apportabili al file di configurazione:

```yaml
classification:
  type: "ternary"       # Classificazione ternaria
```

### Esempio: addestrare solo ResNet

```yaml
models:
  to_train: ["ResNet"]  # Viene addestrata solo la rete ResNet
```

### Esempio: modificare l'architettura ResNet

```yaml
resnet:
  layers: [2, 2, 2, 2]  # ResNet-18 (modello meno computazionalmente oneroso)
  dropout: 0.3
```

### Esempio: aumentare i componenti PCA dell'SVM

```yaml
svm:
  pca_components: 16    # Più features per SVM
  C: 10.0
```

> [!CAUTION]
> Un modello addestrato in modalità **binaria** non è compatibile con dati in configurazione ternaria e viceversa. È indispensabile mantenere la coerenza del parametro `classification.type` tra le fasi di addestramento e valutazione; in caso di variazione della modalità operativa, i pesi precedentemente salvati devono essere eliminati prima di procedere.

---

## 7. Output prodotti

Al termine di un'esecuzione della pipeline (completa o parziale), i principali risultati prodotti in `outs/` sono i seguenti:

### Grafici (`outs/imgs/*/`)

| File | Descrizione |
|------|-------------|
| `pre-processing/class_distribution_*.png` | Distribuzione delle classi per train, val, test e dataset completo |
| `training/{model}/` | Loss e accuracy per epoca per ogni modello DL, in sottocartelle dedicate |
| `confusion_matrix/cm_*.png` | Matrice di confusione per ogni modello |
| `roc_curves/roc_*.png` | Curve ROC one-vs-rest per ogni modello |
| `model_comparison/model_*_comparison.png` | Confronto per singola metrica tra modelli |
| `model_comparison/model_comparison_groups.png` | Confronto complessivo tra modelli |
| `uncertainty/{model}/uncertainty_entropy_*.png` | Distribuzione entropia: predizioni corrette vs errate |
| `uncertainty/{model}/rejection_curve_*.png` | Accuracy vs percentuale eventi accettati |
| `uncertainty/{model}/uncertainty_per_class_*.png` | Boxplot incertezza per classe |
| `interpretability/SHAP_summary_svm.png` | SHAP summary aggregato (SVM) |
| `interpretability/SHAP_bar_svm.png` | Importanza media assoluta delle componenti PCA |
| `interpretability/SHAP_svm_*.png` | SHAP plot per singola classe |
| `pre-processing/pca_scree.png` | Varianza spiegata per componente PCA |

### Risultati (`outs/results/`)

| File | Descrizione |
|------|-------------|
| `model_comparison.csv` | Tabella comparativa con tutte le metriche per ogni modello |
| `classification_reports.txt` | Classification report dettagliato (sklearn) per ogni modello |

### Log (`outs/logs/`)

| File | Descrizione |
|------|-------------|
| `run.log` | Log completo dell'esecuzione con timestamp |

### Modelli (`outs/models/`)

| File | Descrizione |
|------|-------------|
| `ResNet_best_model.pt` | Pesi del miglior checkpoint ResNet (PyTorch) |
| `AlexNet_best_model.pt` | Pesi del miglior checkpoint AlexNet (PyTorch) |
| `SVM_best_model.pkl` | Modello SVM serializzato (joblib) |
| `pca.joblib` | Trasformazione PCA salvata per inferenza |

---

## 8. Licenza MIT

**🔓 MIT License**  
Il presente progetto è distribuito sotto licenza MIT, una licenza open source permissiva che autorizza chiunque ad utilizzare, modificare e redistribuire il codice, anche per finalità commerciali, a condizione di preservare la nota di copyright originale. Gli autori accolgono con favore la citazione del presente lavoro in eventuali elaborati accademici o progetti derivati.

---

## 9. Contatti

**👤 Antonino Barbera**<br>🎓 Corso di Laurea Magistrale in Ingegneria dell'Intelligenza Artificiale e della Sicurezza Informatica<br>[🏫 Università degli Studi di Enna Kore, Italy](https://www.uke.it)<br>✉️ [antonino.barbera001@unikorestudent.it](mailto:antonino.barbera001@unikorestudent.it)

---

**👤 Giuseppe Lorenzo Di Prima**, ORCID: [Giuseppe Lorenzo Di Prima](https://orcid.org/0009-0002-9470-9370)<br>🎓 Ph.D. in Sistemi Intelligenti per l'Ingegneria<br>[🏫 Università degli Studi di Enna Kore, Italy](https://www.uke.it)<br>✉️ [giuseppelorenzo.diprima@unikorestudent.it](mailto:giuseppelorenzo.diprima@unikorestudent.it)
