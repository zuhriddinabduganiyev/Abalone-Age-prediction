
# Stacking Regressor Model for Age Prediction

This project focuses on predicting the `Age` of abalones using a stacking regressor model that combines multiple regression techniques. The main objective is to minimize the Mean Absolute Error (MAE) and maximize the R2 score.

## Project Overview

The dataset used in this project contains various features related to abalones, with the target variable being `Age`. The model leverages different regression techniques, including `HuberRegressor`, `LinearRegression`, `Ridge`, `TheilSenRegressor`, and `Lasso`. These models are combined using a `StackingRegressor` to improve prediction accuracy.

## Model Components

### 1. Huber Regressor
- **Pipeline:** Includes a `HuberRegressor` model.
- **Hyperparameter Tuning:** `epsilon` parameter is tuned using `GridSearchCV`.

### 2. Linear Regression
- **Pipeline:** Includes `PolynomialFeatures` and `LinearRegression`.
- **Hyperparameter Tuning:** Degree of polynomial features is tuned using `GridSearchCV`.

### 3. Ridge Regression
- **Pipeline:** Includes `PolynomialFeatures` and `Ridge`.
- **Hyperparameter Tuning:** Degree of polynomial features and `alpha` parameter are tuned using `GridSearchCV`.

### 4. TheilSen Regressor
- **Pipeline:** Includes `PolynomialFeatures` and `TheilSenRegressor`.
- **Hyperparameter Tuning:** Degree of polynomial features is tuned using `GridSearchCV`.

### 5. Lasso Regression
- **Pipeline:** Includes `PolynomialFeatures` and `Lasso`.
- **Hyperparameter Tuning:** Degree of polynomial features and `alpha` parameter are tuned using `GridSearchCV`.

### 6. Stacking Regressor
- Combines the best pipelines of the models mentioned above.
- **Final Estimator:** A `HuberRegressor` with `epsilon=1.1` is used as the final estimator in the stacking regressor.

## Evaluation Metrics

- **Mean Absolute Error (MAE):** Measures the average magnitude of the errors in a set of predictions, without considering their direction.
- **R2 Score:** Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.

## Results

After training the stacking regressor, the model was evaluated on the test set:

- **MAE:** [Add your MAE result here]
- **R2 Score:** [Add your R2 score result here]

## Getting Started

### Prerequisites

- Python 3.x
- Libraries: `scikit-learn`, `numpy`, `pandas`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/zuhriddinabduganiyev/Abalone-Age-prediction.git
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Model

1. Load your dataset.
2. Split the data into training and testing sets.
3. Run the stacking model with the best parameters.
4. Evaluate the model on the test set.

### Example Usage

```python
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge, Lasso, TheilSenRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler