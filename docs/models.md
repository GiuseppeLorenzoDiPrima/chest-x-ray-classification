# models/

Architetture e pipeline di training per i tre modelli del progetto.

---

## ResNet (`resnet_model.py`)

Implementazione personalizzata con connessioni residue configurabili.

### `ResidualBlock`
Blocco con due Conv2d 3×3, BatchNorm2d e shortcut connection.  
Il downsample viene aggiunto automaticamente quando stride ≠ 1 o i canali cambiano.

### `ResNet`
- Layer groups: configurabili via `resnet.layers` (default `[3, 4, 6, 3]`)
- Output: 2 neuroni (binario) o 3 neuroni (ternario)
- Segue la struttura ResNet-50 con AvgPool finale e FC layer

### `build_resnet(config) → ResNet`
Factory che legge la sezione `resnet` del config YAML.

---

## AlexNet (`alexnet_model.py`)

Architettura classica con 5 layer convoluzionali e 3 fully-connected.

### `AlexNet`
- Feature extractor: Conv96 → Conv256 → Conv384 → Conv384 → Conv256 con MaxPool
- Classifier: Dropout → FC4096 → Dropout → FC4096 → FC(num_classes)
- Dropout standard (p=0.5) per regolarizzazione

### `build_alexnet(config) → AlexNet`
Factory che legge la sezione `alexnet` del config YAML.

---

## SVM (`svm_model.py`)

Pipeline in 4 passi: ViT embeddings → PCA → SMOTE → SVM.

### `VisionEmbeddings`
Usa `google/vit-base-patch16-224` (HuggingFace Transformers) per estrarre
embedding da 768 dimensioni (media sul sequence length del last hidden state).

| Metodo | Descrizione |
|--------|-------------|
| `extract_all()` | Estrae embeddings per train/val/test, fitta PCA su train, applica SMOTE |
| `extract_single()` | Estrae embeddings per un singolo split usando PCA già salvata |

### `train_svm(data, config) → dict`
Esegue la pipeline completa e salva `SVM_best_model.pkl` e `pca.joblib` in `models_dir`.

**Loss utilizzata:**
- Binaria → `hinge_loss`
- Ternaria → `log_loss` (richiede `probability=true`)

### `evaluate_svm(data, config) → dict`
Valuta il modello sul test set. Restituisce `metrics`, `confusion_matrix`,
`y_pred`, `y_true`, `y_score`.

---

## Training DL (in `main.py`)

Entrambi i modelli DL condividono lo stesso loop:

| Componente | Valore / Configurazione |
|-----------|------------------------|
| Loss | CrossEntropyLoss |
| Optimizer | Adam / SGD / RMSprop (config) |
| Scheduler | LambdaLR con warmup lineare + decay lineare |
| Oversampling | WeightedRandomSampler (bilanciamento classi) |
| Early stopping | Pazienza configurabile su metrica scelta |
| Salvataggio | Miglior modello su val metric → `.pt` |
