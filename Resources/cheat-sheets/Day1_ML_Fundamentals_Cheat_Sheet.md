# Day 1: Machine Learning Fundamentals Cheat Sheet

## Introduction to Machine Learning

### What is Machine Learning?
- **Definition**: A subset of AI that enables computers to learn and make decisions from data without being explicitly programmed
- **Goal**: Build models that can make predictions or decisions based on data patterns

### Why Machine Learning?
- **Automation**: Automate complex decision-making processes
- **Pattern Recognition**: Identify patterns in large datasets
- **Prediction**: Forecast future outcomes based on historical data
- **Scalability**: Handle massive amounts of data efficiently

## 🔍 Types of Machine Learning

### 1. Supervised Learning
- **Definition**: Learning with labeled training data
- **Types**:
  - **Classification**: Predict discrete categories/classes
  - **Regression**: Predict continuous numerical values
- **Examples**: Email spam detection, house price prediction, image recognition

### 2. Unsupervised Learning
- **Definition**: Learning from data without labeled outcomes
- **Types**:
  - **Clustering**: Group similar data points
  - **Association**: Find relationships between variables
  - **Dimensionality Reduction**: Reduce feature space
- **Examples**: Customer segmentation, market basket analysis

### 3. Reinforcement Learning
- **Definition**: Learning through interaction with environment using rewards/penalties
- **Examples**: Game playing, robotics, autonomous vehicles

## 🛠️ Getting Started with scikit-learn

### Installation
```python
pip install scikit-learn numpy pandas matplotlib
```

### Basic Import Structure
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```

## Data Loading Techniques

### Loading Data with Pandas
```python
# CSV files
df = pd.read_csv('data.csv')

# Excel files
df = pd.read_excel('data.xlsx')

# JSON files
df = pd.read_json('data.json')

# From URL
df = pd.read_csv('https://example.com/data.csv')
```

### Built-in Datasets (scikit-learn)
```python
from sklearn.datasets import load_iris, load_boston, load_digits

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Load Boston housing dataset
boston = load_boston()
X, y = boston.data, boston.target
```

## 📈 Linear Regression

### Implementation
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
```

### Key Concepts
- **Equation**: y = mx + b (simple), y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ (multiple)
- **Assumptions**: 
  - Linear relationship between features and target
  - Independence of errors
  - Homoscedasticity (constant variance)
  - Normal distribution of errors

## 📏 Evaluation Metrics for Regression

### Mean Squared Error (MSE)
```python
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
```
- **Formula**: MSE = (1/n) × Σ(yᵢ - ŷᵢ)²
- **Range**: 0 to ∞ (lower is better)

### Mean Absolute Error (MAE)
```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
```
- **Formula**: MAE = (1/n) × Σ|yᵢ - ŷᵢ|
- **Range**: 0 to ∞ (lower is better)

### Root Mean Squared Error (RMSE)
```python
import numpy as np
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
```
- **Formula**: RMSE = √MSE
- **Unit**: Same as target variable

### R² Score (Coefficient of Determination)
```python
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
```
- **Formula**: R² = 1 - (SS_res / SS_tot)
- **Range**: -∞ to 1 (closer to 1 is better)
- **Interpretation**: Proportion of variance explained by the model

## 🔧 Complete Linear Regression Example
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load data
# X = features, y = target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Visualize results
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values')
plt.show()
```

## Key Takeaways
- ML is about finding patterns in data to make predictions
- Supervised learning uses labeled data, unsupervised doesn't
- Linear regression is fundamental for understanding relationships
- Always evaluate your model using appropriate metrics
- R² score is often the most interpretable metric for regression

## 📝 Home Task Preparation
**Topic**: Non-Linear Regression
- **Key Concepts to Research**:
  - Polynomial regression
  - Regularization (Ridge, Lasso)
  - Feature transformation
  - Overfitting vs underfitting
