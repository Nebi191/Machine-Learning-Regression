# Ames Housing Price Prediction: Stacking Ensemble 🏠📊

This project achieves a high-performance prediction of house prices in Ames, Iowa, by leveraging a sophisticated **Stacking Regressor** architecture. 

## 🚀 Performance Highlights
- **R² Score:** 0.9108 (Explaining 91% of the variance)
- **MAE:** $13,233 (Average absolute error)
- **RMSE:** 20,483

## 🧠 Model Architecture
The model uses a two-level hierarchy to minimize error and prevent overfitting:

1. **Base Models (Level 0):**
   - **XGBoost:** Fine-tuned for gradient boosting precision.
   - **LightGBM:** Optimized with leaf-wise growth for speed and accuracy.
   - **SVR:** Polynomial kernel to capture non-linear geometric trends.
2. **Meta-Model (Level 1):**
   - **Ridge Regression:** Acts as the "Chief Justice," weighting the base models' predictions to deliver the final output.



## 🛠️ Key Insights
- **XGBoost Dominance:** The meta-model assigned a 0.94 weight to XGBoost, identifying it as the most reliable predictor for this dataset.
- **Regularization:** Integrated L1 and L2 penalties to maintain a robust "Lionidas-like" discipline against overfitting.

## 📁 Repository Structure
- `Ames_Housing_Processed.csv`: Preprocessed dataset.
- `stacking_analysis.ipynb`: Full modeling workflow and evaluation.
- `requirements.txt`: List of dependencies.