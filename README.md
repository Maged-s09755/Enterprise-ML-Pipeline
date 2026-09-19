# Enterprise MLOps & Production-Ready ML Pipeline

An enterprise-grade, production-ready Machine Learning pipeline designed for robust classification tasks with rigorous data leakage prevention, custom memory optimization, automated outlier handling, and class imbalance mitigation.

## 🚀 Key Features
- **Zero Data Leakage Architecture:** All preprocessing steps (`KNNImputer`, `OutlierClipper`, `TargetEncoder`, `SMOTE`) are encapsulated within an `imbalanced-learn` pipeline, ensuring transformers are fitted strictly on training folds during Cross-Validation.
- **Custom Transformers:**
  - `MemoryOptimizer`: Automatically downcasts numeric types to optimize RAM usage while maintaining numerical stability (float32/integers).
  - `OutlierClipper`: Safely clips extreme outliers based on statistical quantiles (1% and 99%) without dropping data points.
- **Advanced Preprocessing & Encoding:** Handles mixed data types, missing values via KNN imputation, high-cardinality categorical variables via `TargetEncoder`, and low-cardinality via `OneHotEncoder`.
- **Automated Hyperparameter Optimization:** Uses `RandomizedSearchCV` with stratified cross-validation (`StratifiedKFold`) optimized for `ROC-AUC`.
- **Production-Ready Serialization:** Saves the entire preprocessing and classification pipeline as a single `.pkl` artifact for seamless deployment.

---

## 🛠️ Project Structure
```text
Enterprise-ML-Pipeline/
│
├── main.py              # Core pipeline implementation & simulation script
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
