# Documentazione — Chest X-Ray Classification

Documentazione tecnica del progetto di Machine Learning per la classificazione
di radiografie toraciche pediatriche.

**Autori:** Barbera Antonino · Di Prima Giuseppe Lorenzo  
**Corso:** Machine Learning @ UniKore

---

## Moduli

| Modulo | File | Descrizione |
|--------|------|-------------|
| Data Loader | [`data_loader.md`](data_loader.md) | Download Kaggle, preprocessing, splits |
| Models | [`models.md`](models.md) | ResNet, AlexNet, SVM con ViT embeddings |
| Visualization | [`visualization.md`](visualization.md) | Grafici IEEE-ready |
| Evaluation | [`evaluation.md`](evaluation.md) | Metriche, confronto, report |

---

## Pipeline

```
main.py
  │
  ├── Fase 1: data_classes/data_loader.py
  │           Download Kaggle → struttura binaria/ternaria → DataLoader
  │
  ├── Fase 2: models/resnet_model.py
  │           models/alexnet_model.py
  │           Training DL con early stopping → salva .pt
  │
  ├── Fase 3: models/svm_model.py
  │           ViT embeddings → PCA → SMOTE → SVM → salva .pkl
  │
  ├── Fase 4: utils/interpretability.py  (SHAP)
  │           utils/uncertainty.py       (MC Dropout)
  │
  └── Fase 5: utils/evaluation.py
              Tabella comparativa → CSV + grafici
```

---

## Configurazione rapida

Editare [`config/config.yaml`](../config/config.yaml):

```yaml
classification:
  type: "binary"      # oppure "ternary"

models:
  to_train: ["ResNet", "AlexNet", "SVM"]
```
