# Day 2: ML Workflow & Classification Cheat Sheet

## Machine Learning Workflow

### 1. Problem Definition
- **Classification**: Predict categories/classes
- **Regression**: Predict continuous values
- **Clustering**: Group similar data points

### 2. Data Collection & Understanding
- **Sources**: CSV, databases, APIs, web scraping
- **Exploration**: Shape, data types, missing values, distributions
```python
# Basic data exploration
df.shape
df.info()
df.describe()
df.isnull().sum()
df.head()
```

### 3. Data Preprocessing
```python
# Handle missing values
df.fillna(df.mean())  # Numerical
df.fillna(df.mode()[0])  # Categorical

# Remove duplicates
df.drop_duplicates()

# Handle outliers
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
```

### 4. Feature Engineering
- **Feature Selection**: Choose relevant features
- **Feature Creation**: Combine/transform existing features
- **Encoding**: Convert categorical to numerical
- **Scaling**: Normalize feature ranges

### 5. Model Selection & Training
- Choose appropriate algorithm
- Split data (train/validation/test)
- Train model
- Tune hyperparameters

### 6. Model Evaluation
- Use appropriate metrics
- Cross-validation
- Compare multiple models

### 7. Deployment & Monitoring
- Deploy model to production
- Monitor performance
- Retrain when necessary

## Data Types

### Numerical Data
- **Continuous**: Can take any value within a range (height, weight, temperature)
- **Discrete**: Countable values (number of children, cars sold)

### Categorical Data
- **Nominal**: No natural order (colors, names, gender)
- **Ordinal**: Natural order (ratings, education levels, sizes)

### Time Series Data
- Data points indexed by time
- Special handling required for trends, seasonality

## Classification Algorithms

### 1. Logistic Regression
```python
from sklearn.linear_model import LogisticRegression

# Create and train model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Make predictions
y_pred = log_reg.predict(X_test)
y_pred_proba = log_reg.predict_proba(X_test)
```

**Key Concepts**:
- Uses sigmoid function to map any real number to (0,1)
- Outputs probabilities
- Decision boundary at 0.5 probability
- Works well for linearly separable data

### 2. K-Nearest Neighbors (KNN)
```python
from sklearn.neighbors import KNeighborsClassifier

# Create and train model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)
```

**Key Concepts**:
- Lazy learning algorithm (no training phase)
- Classifies based on majority vote of k nearest neighbors
- Distance-based (usually Euclidean)
- Sensitive to feature scaling
- Choose odd k to avoid ties

### 3. Decision Trees
```python
from sklearn.tree import DecisionTreeClassifier

# Create and train model
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Make predictions
y_pred = dt.predict(X_test)
```

**Key Concepts**:
- Tree-like model of decisions
- Easy to interpret and visualize
- Can handle both numerical and categorical features
- Prone to overfitting
- Uses criteria like Gini impurity or entropy

## 📏 Evaluation Metrics for Classification

### 1. Accuracy
```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
```
- **Formula**: (TP + TN) / (TP + TN + FP + FN)
- **Range**: 0 to 1 (higher is better)
- **Good for**: Balanced datasets

### 2. Precision
```python
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred, average='weighted')
```
- **Formula**: TP / (TP + FP)
- **Interpretation**: Of all positive predictions, how many were correct?
- **Good for**: When false positives are costly

### 3. Recall (Sensitivity)
```python
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred, average='weighted')
```
- **Formula**: TP / (TP + FN)
- **Interpretation**: Of all actual positives, how many were correctly identified?
- **Good for**: When false negatives are costly

### 4. F1-Score
```python
from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred, average='weighted')
```
- **Formula**: 2 × (Precision × Recall) / (Precision + Recall)
- **Interpretation**: Harmonic mean of precision and recall
- **Good for**: Imbalanced datasets

### 5. Confusion Matrix
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
```

**Structure**:
```
           Predicted
         0    1
Actual 0 TN   FP
       1 FN   TP
```

## 🌸 Hands-on: Iris Dataset Classification

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models to compare
models = {
    'Logistic Regression': LogisticRegression(),
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

# Train and evaluate models
for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"{name}: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    print("-" * 50)
```

## 📧 Case Study: Spam Detection with Decision Trees

### Problem Setup
- **Objective**: Classify emails as spam or not spam
- **Features**: Word frequency, email length, special characters, etc.
- **Algorithm**: Decision Tree

### Implementation Steps
```python
# Assuming preprocessed email data
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create decision tree
dt_spam = DecisionTreeClassifier(
    max_depth=10,           # Limit tree depth to prevent overfitting
    min_samples_split=20,   # Minimum samples to split a node
    min_samples_leaf=10,    # Minimum samples in a leaf
    random_state=42
)

# Train model
dt_spam.fit(X_train, y_train)

# Make predictions
y_pred = dt_spam.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Spam Detection Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam']))
```

### Feature Importance
```python
# Get feature importance
feature_importance = dt_spam.feature_importances_
feature_names = ['word_freq_spam', 'word_freq_free', 'char_freq_!', 'capital_run_length_avg']

# Create feature importance DataFrame
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(importance_df)
```

## 🛠️ Feature Engineering Techniques

### 1. Encoding Categorical Variables
```python
# One-Hot Encoding
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
encoded_features = encoder.fit_transform(categorical_data.reshape(-1, 1))

# Label Encoding
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(categorical_data)

# Using pandas
pd.get_dummies(df, columns=['categorical_column'])
```

### 2. Feature Scaling
```python
# StandardScaler (mean=0, std=1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# MinMaxScaler (range 0-1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

### 3. Creating New Features
```python
# Polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Mathematical transformations
df['log_feature'] = np.log(df['feature'] + 1)
df['sqrt_feature'] = np.sqrt(df['feature'])
df['feature_squared'] = df['feature'] ** 2
```

## Model Selection Best Practices

### 1. Cross-Validation
```python
from sklearn.model_selection import cross_val_score

# 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

### 2. Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance']
}

# Grid search
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

## Key Takeaways
- Data preprocessing is crucial for model performance
- Different algorithms work better for different types of problems
- Always evaluate using multiple metrics
- Confusion matrix provides detailed insight into classification errors
- Feature engineering can significantly improve model performance
- Cross-validation gives more reliable performance estimates

## 📝 Home Task Preparation
**Topic**: Feature Selection Methods
- **Key Concepts to Research**:
  - Filter methods (correlation, chi-square)
  - Wrapper methods (forward/backward selection)
  - Embedded methods (Lasso, tree-based importance)
  - Dimensionality reduction techniques
