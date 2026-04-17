# utils/

Moduli di valutazione, interpretabilità e uncertainty quantification.

---

## evaluation.py

### `compute_metrics(y_true, y_pred) → dict`
Calcola le metriche base con `average='macro'`:
- `accuracy`, `precision`, `recall`, `f1`

### `evaluate_dl_model(model, dataloader, criterion, device) → tuple`
Valuta un modello DL su un DataLoader.  
Restituisce `(metrics, confusion_matrix, y_true, y_pred)`.

### `build_comparison_table(all_results) → pd.DataFrame`
Costruisce un DataFrame con una riga per modello e colonne:
`Modello · accuracy · precision · recall · f1 · loss`

### `generate_full_report(all_results, data, config) → pd.DataFrame`
Pipeline completa di valutazione finale:
1. Salva `model_comparison.csv` in `results_dir`
2. Scrive `classification_reports.txt` con sklearn report per ogni modello
3. Genera confusion matrix, ROC curves e grafici di confronto

---

## interpretability.py

### `run_shap_analysis(all_results, data, config)`
Esegue l'analisi SHAP per ogni modello SVM nei risultati.

Utilizza `shap.KernelExplainer` con background ridotto tramite k-means
(`background_clusters` centroidi). Opera sulle componenti PCA del ViT,
restituendo l'importanza relativa di ogni PC nella predizione.

Produce:
- `SHAP_summary_svm.png` — summary plot aggregato
- `SHAP_bar_svm.png` — importanza media assoluta
- `SHAP_svm_normal.png`, `SHAP_svm_pneumonia.png`, ecc. — per singola classe

---

## uncertainty.py

### `mc_dropout_predict(model, dataloader, device, n_iterations) → dict`
Esegue `n_iterations` forward pass con dropout attivo.

**Restituisce:**
```python
{
    "mean_probs":  np.ndarray,   # (n_samples, n_classes)
    "std_probs":   np.ndarray,   # (n_samples, n_classes)
    "predictions": np.ndarray,   # (n_samples,)
    "entropy":     np.ndarray,   # (n_samples,) — -Σ p·log(p)
    "y_true":      np.ndarray,   # (n_samples,)
}
```

### `run_uncertainty_analysis(dl_results, data, config)`
Applica MC Dropout a tutti i modelli DL e genera i grafici di uncertainty:
- Distribuzione entropia (predizioni corrette vs errate)
- Rejection curve (accuracy al variare della soglia di entropia)
- Boxplot incertezza per classe
