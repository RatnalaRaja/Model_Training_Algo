
#  Simple Linear Regression 

This README is designed to help you revise and understand **Simple Linear Regression** using a real dataset: height and weight. It includes code breakdown, visual interpretation, and explanation of key metrics.

---

##  Regression Equation

The model tries to find the best straight line:
```
Y = mX + b
```
- `Y`: Predicted value (e.g., Height)
- `X`: Input feature (e.g., Weight)
- `m`: Slope (coefficient)
- `b`: Intercept

---


```
                ^
                |                       Actual Y
                |                        ●
  Height (Y)    |                   ●   ●
                |              ●   |   ●
                |         ●   |   ●
                |    ●    |  ●
                |_________|_____________________>
                        Weight (X)
                        |-----> Predicted Y
```

The model tries to **minimize vertical distances (errors)** between the actual points and the regression line.

---

##  What is R² (R-squared)?

**R² (Coefficient of Determination)** shows how well the line fits the data.

###  Formula:
```
R² = 1 - (SSR / SST)
```
- **SSR**: Sum of Squares of Residuals
- **SST**: Total Sum of Squares

###  Meaning:
- R² = 1 → perfect fit
- R² = 0 → model explains nothing
- R² = 0.8 → 80% of variation in Y is explained by X

---

##  What is Adjusted R²?

Adjusted R² considers the number of predictors and adjusts R² accordingly.

###  Formula:
```
Adjusted R² = 1 - [(1 - R²) * (n - 1) / (n - k - 1)]
```
- `n`: number of samples
- `k`: number of predictors

Use Adjusted R² when you have multiple features or want to prevent overfitting.

---

## 📏 Evaluation Metrics

You used:
- **MSE (Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**

These indicate how far off predictions are from the true values.

---
===================================================================================

#  Multiple Linear Regression 

This README is crafted to help you revise and understand **Multiple Linear Regression** using the California Housing dataset. It breaks down code, key metrics, and explains model behavior in simple terms.

---

##  What is Multiple Linear Regression?

It models the relationship between **one target variable (Y)** and **two or more input features (X₁, X₂, ..., Xₙ)**:
```
Y = b₀ + b₁X₁ + b₂X₂ + ... + bₙXₙ
```

Where:
- `Y`: Predicted output (House Price)
- `X₁, X₂, ..., Xₙ`: Input features (e.g., MedInc, AveRooms, etc.)
- `b₀`: Intercept
- `b₁...bₙ`: Coefficients (weights)

---

##  Dataset Used

- **Source**: California Housing dataset from `sklearn.datasets`
- **Target**: `Price` of the house
- **Features**: MedInc, HouseAge, AveRooms, AveOccup, etc.

---

##  Steps You Performed

### 1.  Imported Libraries
```python
from sklearn.datasets import fetch_california_housing
import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
```

### 2.  Loaded Data
```python
california = fetch_california_housing()
dataset = pd.DataFrame(california.data, columns=california.feature_names)
dataset['Price'] = california.target
```

### 3.  Exploratory Data Analysis
- Used `pairplot()` and `heatmap()` to visualize correlations.

### 4.  Splitting Features
```python
X = dataset.iloc[:, :-1]
Y = dataset.iloc[:, -1]
```

### 5.  Train-Test Split
```python
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=10)
```

### 6.  Feature Scaling
```python
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)
```

### 7.  Model Training
```python
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, Y_train)
```

### 8.  Prediction
```python
Y_pred = regression.predict(X_test)
```

---

##  Evaluation Metrics

###  Error Metrics
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
```
- **MSE**: Average squared error
- **MAE**: Average absolute error
- **RMSE**: Root Mean Squared Error

###  R² (Coefficient of Determination)
```python
from sklearn.metrics import r2_score
r2 = r2_score(Y_test, Y_pred)
```

###  Adjusted R²
```python
Adjusted = 1 - (1 - r2) * (len(Y_test) - 1) / (len(Y_test) - X_test.shape[1] - 1)
```

==============================================================================

#  Pickling 


Pickling is the process of **serializing** a Python object — converting it into a byte stream. You can save this stream to a file or transfer it over a network, then later **unpickle** it to restore the original object.

Simply "saving your progress" so you don't have to start over.

---

##  Pickling vs Unpickling

| Operation     | Description                          | Function          |
|---------------|--------------------------------------|-------------------|
| **Pickling**   | Save Python object → byte stream     | `pickle.dump()`   |
| **Unpickling** | Load byte stream → Python object     | `pickle.load()`   |

---

## Your Code Example

 Trained regression model:

```python
import pickle


pickle.dump(regression, open('regressor.pkl', 'wb')) #Saving the model

model = pickle.load(open('regressor.pkl', 'rb'))#Loading the model
```

---

## Why Use Pickling?

- Saves time — no need to retrain the model every time
-  Allows easy deployment and sharing of models
-  Works well with machine learning pipelines




