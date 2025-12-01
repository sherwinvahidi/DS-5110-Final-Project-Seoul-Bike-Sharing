# DS 5110 Final Project: Seoul Bike Sharing Demand Prediction

**Authors:** Yuzhe Li, Sherwin Vahidimowlavi, Bolai Yin  
**Course:** DS 5110 - Essentials of Data Science
**Term:** Fall 2025

## Project Overview

This project implements binary classification models to predict high-demand periods in Seoul's bike-sharing system. We compare Logistic Regression, Random Forest, and XGBoost using temporal cross-validation and comprehensive feature engineering.

**Key findings:**
- XGBoost achieved best performance (F1=0.964, ROC-AUC=0.998)
- Temporal features (rolling averages, lag variables) critical for accuracy
- Ensemble methods substantially outperform linear baseline

## Dataset

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand)
- **Time Period:** December 2017 - November 2018
- **Size:** 8,760 hourly observations
- **Features:** Weather conditions (temperature, humidity, wind, etc.) + temporal features

## Repository Structure
```
├── data/
│   ├── SeoulBikeData.csv          # Original dataset
│   └── processed_data_full.csv     # Processed dataset with engineered features
├── notebooks/
│   ├── 01_data_preprocessing.ipynb # EDA and feature engineering
│   ├── 02_logistic_regression.ipynb # Logistic regression implementation
│   ├── 03_random_forest.ipynb      # Random forest implementation
│   └── 04_xgboost.ipynb            # XGBoost implementation
├── report/
│   └── Final_Report.pdf            # Written report
├── presentation/
│   └── Presentation_Slides.pdf     # Presentation slides
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Installation & Setup
```bash
# Clone the repository
git clone https://github.com/sherwinvahidi/DS-5110-Final-Project-Seoul-Bike-Sharing.git
cd DS-5110-Final-Project-Seoul-Bike-Sharing

# Install dependencies
pip install -r requirements.txt

# Run preprocessing
python src/data_preparation.py

# Run models (in notebooks/ directory)
jupyter notebook
```

## Models Implemented

### 1. Logistic Regression (Baseline)
- L2 regularization with C=10.0
- Two-stage GridSearchCV hyperparameter tuning
- F1-score: 0.857, ROC-AUC: 0.978

### 2. Random Forest
- 900 trees, max_features=0.6
- RandomizedSearchCV over 28 configurations
- F1-score: 0.955, ROC-AUC: 0.997

### 3. XGBoost (Best Performance)
- Gradient boosting with temporal cross-validation
- Optimized learning rate, depth, and regularization
- F1-score: 0.964, ROC-AUC: 0.998

## Key Features Engineered

- **Lag features:** 1h, 2h, 24h previous demand
- **Rolling statistics:** 3h, 6h, 24h windows (mean, std, max)
- **Interaction terms:** temp×hour, temp², humidity×temp, wind×rain
- **Cyclical encodings:** hour_sin, hour_cos

## Evaluation Methodology

- **Train-Test Split:** Temporal (first 10 months train, last 2 months test)
- **Cross-Validation:** TimeSeriesSplit (5 folds) to prevent information leakage
- **Primary Metric:** F1-score (balances precision and recall for imbalanced data)
- **Additional Metrics:** Accuracy, Precision, Recall, ROC-AUC, PR-AUC

## Results Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.920 | 0.777 | 0.954 | 0.857 | 0.978 |
| Random Forest | 0.974 | 0.924 | 0.988 | 0.955 | 0.997 |
| **XGBoost** | **0.979** | **0.968** | **0.960** | **0.964** | **0.998** |

## Key Insights

1. **Binary classification** aligns better with operational decisions than regression
2. **Temporal features** (rolling_mean_3h) dominate feature importance
3. **Transition hours** (4-7 AM, 19-22 PM) remain most challenging to predict
4. **Ensemble methods** capture non-linear weather-time interactions effectively

## References

See full citations in [Final_Report.pdf](report/Final_Report.pdf)

## License

This project is for educational purposes as part of DS 5110 coursework.
