# 🏠 Housing Price Prediction

A Python project that predicts property prices from listing features, with predictions generated for properties with contract dates from **January to March 2023**.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)

## 📌 Overview

This project builds a regression model to estimate housing prices and exports one prediction per property and contract date. House prices are right-skewed, so the pipeline is designed to handle large values and outliers without letting them dominate the model.

> **Note:** The original notebook was lost, and only the results file survived. `housing_price_prediction.py` is a reconstruction of the workflow that fits the saved results. Hyperparameters and features may differ from the original run.

## 📂 Files

| File | Description |
|---|---|
| `Recode-2by2.csv` | Final predictions (`property_key`, `contractDate`, `prediction`) |
| `housing_price_prediction.py` | Training and prediction pipeline (reconstructed) |
| `README.md` | Project documentation |

## 🔄 Workflow

1. **Load data**: read the training and test property data
2. **Preprocess**: impute missing values (median for numeric, most frequent for categorical) and encode categorical features
3. **Transform target**: train on log(price) to handle the right-skew
4. **Train and validate**: tree-based regressor with a train/validation split, scored with MAE, RMSE and R²
5. **Predict**: refit on all training data, predict the test set, and convert back to the original price scale
6. **Export**: save `property_key`, `contractDate` and `prediction` to CSV

## 📊 Results

The output file contains **2,331 predictions** across **2,220 unique properties**.

| Month | Predictions | Median | Mean |
|---|---|---|---|
| Jan 2023 | 645 | 1,462,601 | 1,968,588 |
| Feb 2023 | 844 | 1,483,043 | 2,061,204 |
| Mar 2023 | 842 | 1,498,921 | 2,055,933 |
| **Overall** | **2,331** | **1,479,917** | **2,033,673** |

- Predicted prices range from about **0.66M** to **19.5M**
- 50% of predictions fall between **1.16M** and **2.14M**
- The distribution is right-skewed, with a small number of high-value properties pulling the mean above the median

## 🚀 How to run

```bash
git clone https://github.com/<your-username>/Housing-Price-Prediction-.git
cd Housing-Price-Prediction-
pip install pandas numpy scikit-learn
python housing_price_prediction.py
```

Update `TRAIN_PATH`, `TEST_PATH` and `TARGET` at the top of the script to match your data files.

## 🛠️ Tech stack

- Python
- pandas, NumPy
- scikit-learn

## 👩‍💻 Author

**Rosa Hunahunan**: Research Analyst | Applied Statistics
