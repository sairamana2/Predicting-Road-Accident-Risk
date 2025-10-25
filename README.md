# Road Accident Risk Prediction 🚗💥
Predicts road accident risk using machine learning. Features like weather, road type, vehicle count, and time of day are preprocessed and used to train a LightGBM model, generating predictions for unseen data to help identify high-risk situations and improve road safety.
This is a machine learning project to predict the `accident_risk` (a score between 0 and 1) based on various road, weather, and time conditions. The project is based on a Kaggle dataset.

**My final submission file (`submission.csv`) is included in this repository.**

---

### # Project Goal

The goal was to build a regression model that accurately predicts accident risk. I used a LightGBM Regressor, which is a powerful model for tabular data.

---

### # Project Workflow

1.  **Data Loading:** Loaded `train.csv` and `test.csv`.
2.  **Feature Engineering:** Converted the `time_of_day` category (e.g., 'morning') into a numerical `Hour` feature.
3.  **Preprocessing:**
    * **Numerical Features:** Filled missing values with the `median` and scaled using `StandardScaler`.
    * **Categorical Features:** Filled missing values with the `most_frequent` value and encoded using `OneHotEncoder`.
4.  **Modeling:** Trained an `LGBMRegressor` on the full dataset.
5.  **Post-processing:** Clipped all final predictions to be within the valid range of [0, 1] using `np.clip`.

---

### # How to Run This Project

1.  Clone or download this repository.
2.  Download the original data from the [Kaggle Competition Page](https://www.kaggle.com/competitions/playground-series-s5e10/data).
3.  Place `train.csv`, `test.csv`, and `sample_submission.csv` in the same folder as the `road_accident_prediction.ipynb` notebook.
4.  Open the notebook and run all cells.
