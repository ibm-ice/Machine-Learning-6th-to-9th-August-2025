# Day 3: Advanced Supervised Learning & Unsupervised Learning Cheat Sheet

## Encoding Methods & Requirements

### 1. Categorical Encoding Overview
- **Purpose**: Convert categorical variables to numerical format for ML algorithms
- **Choice depends on**: Data type, algorithm requirements, cardinality

### 2. One-Hot Encoding
```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Using pandas
df_encoded = pd.get_dummies(df, columns=['category_column'])

# Using sklearn
encoder = OneHotEncoder(sparse=False, drop='first')  # drop='first' to avoid multicollinearity
encoded = encoder.fit_transform(df[['category_column']])
```
- **Best for**: Nominal categories with low cardinality (<10-15 categories)
- **Pros**: No ordinal assumption, works well with linear models
- **Cons**: High dimensionality, sparse matrices

### 3. Label Encoding
```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df['encoded_column'] = encoder.fit_transform(df['category_column'])
```
- **Best for**: Ordinal categories or tree-based algorithms
- **Pros**: Low dimensionality, preserves memory
- **Cons**: Implies false ordering for nominal data

### 4. Target Encoding
```python
# Calculate mean target value for each category
target_means = df.groupby('category_column')['target'].mean()
df['target_encoded'] = df['category_column'].map(target_means)
```
- **Best for**: High cardinality categorical variables
- **Pros**: Captures relationship with target
- **Cons**: Risk of overfitting, requires careful cross-validation

### 5. Binary Encoding
```python
import category_encoders as ce

encoder = ce.BinaryEncoder(cols=['category_column'])
df_encoded = encoder.fit_transform(df)
```
- **Best for**: High cardinality categorical variables
- **Pros**: Reduces dimensionality compared to one-hot
- **Cons**: Less interpretable

## Advanced Supervised Learning Algorithms

### 1. Support Vector Machine (SVM)
```python
from sklearn.svm import SVC

# Create SVM classifier
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)
```

**Key Concepts**:
- **Objective**: Find optimal hyperplane that separates classes
- **Kernels**: Linear, polynomial, RBF (radial basis function), sigmoid
- **Parameters**:
  - `C`: Regularization parameter (higher C = less regularization)
  - `gamma`: Kernel coefficient (higher gamma = more complex boundary)
- **Pros**: Effective in high dimensions, memory efficient
- **Cons**: Slow on large datasets, no probability estimates by default

### 2. Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

# Create Random Forest classifier
rf_model = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Maximum depth of trees
    min_samples_split=5,   # Minimum samples to split
    random_state=42
)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Feature importance
importance = rf_model.feature_importances_
```

**Key Concepts**:
- **Ensemble method**: Combines multiple decision trees
- **Bootstrap Aggregating**: Each tree trained on random subset of data
- **Random Feature Selection**: Each split considers random subset of features
- **Pros**: Handles overfitting well, provides feature importance, handles mixed data types
- **Cons**: Less interpretable than single decision tree, can overfit with very noisy data

### 3. Algorithm Comparison
| Algorithm | Type | Pros | Cons | Best Use Case |
|-----------|------|------|------|---------------|
| Logistic Regression | Linear | Fast, interpretable, probabilistic | Assumes linear relationship | Binary classification, baseline model |
| KNN | Instance-based | Simple, no assumptions | Sensitive to scale, slow prediction | Small datasets, non-linear boundaries |
| Decision Tree | Tree-based | Interpretable, handles mixed data | Prone to overfitting | When interpretability is crucial |
| SVM | Kernel-based | Effective in high dimensions | Slow on large data | Text classification, high-dimensional data |
| Random Forest | Ensemble | Robust, feature importance | Less interpretable | General purpose, tabular data |

## Case Study: Titanic Survival Prediction

### Problem Overview
- **Objective**: Predict passenger survival on the Titanic
- **Features**: Age, sex, passenger class, fare, embarkation port, etc.
- **Type**: Binary classification problem

### Complete Implementation
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load data (assuming titanic.csv is available)
df = pd.read_csv('titanic.csv')

# Data exploration
print(df.info())
print(df.isnull().sum())

# Data preprocessing
def preprocess_titanic(df):
    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # Feature engineering
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Encode categorical variables
    le = LabelEncoder()
    df['Sex_encoded'] = le.fit_transform(df['Sex'])
    df['Embarked_encoded'] = le.fit_transform(df['Embarked'])
    
    # Select features
    features = ['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 
                'Fare', 'Embarked_encoded', 'FamilySize', 'IsAlone']
    
    return df[features], df['Survived']

# Preprocess data
X, y = preprocess_titanic(df)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Feature Importance - Titanic Survival Prediction')
plt.show()
```

### Key Insights from Titanic Analysis
- **Gender**: Strong predictor (women had higher survival rate)
- **Passenger Class**: Higher class passengers had better survival chances
- **Age**: Children had higher survival rates
- **Family Size**: Medium family sizes had optimal survival rates

## 🔍 Introduction to Unsupervised Learning

### What is Unsupervised Learning?
- **Definition**: Learning patterns from data without labeled outcomes
- **Goal**: Discover hidden structures in data
- **No target variable**: Only input features available

### Types of Unsupervised Learning

#### 1. Clustering
- **Purpose**: Group similar data points together
- **Applications**: Customer segmentation, image segmentation, gene sequencing
- **Algorithms**: K-Means, Hierarchical, DBSCAN

#### 2. Association Rule Learning
- **Purpose**: Find relationships between different items
- **Applications**: Market basket analysis, recommendation systems
- **Example**: "People who buy bread also buy butter"

#### 3. Dimensionality Reduction
- **Purpose**: Reduce number of features while preserving information
- **Applications**: Data visualization, noise reduction, feature extraction
- **Algorithms**: PCA, t-SNE, LDA

### Use Cases for Unsupervised Learning

#### Customer Segmentation
```python
# Example: Segment customers based on purchasing behavior
features = ['annual_spending', 'visit_frequency', 'avg_order_value']
# Use clustering to identify customer groups
```

#### Anomaly Detection
```python
# Example: Detect fraudulent transactions
# Use clustering or isolation forest to identify outliers
```

#### Data Exploration
```python
# Example: Understand structure of new dataset
# Use dimensionality reduction for visualization
```

#### Feature Engineering
```python
# Example: Create new features from clustering results
# Add cluster labels as new categorical features
```

## K-Means Clustering Deep Dive

### Algorithm Overview
1. **Initialize**: Choose number of clusters (k) and place centroids randomly
2. **Assign**: Assign each point to nearest centroid
3. **Update**: Move centroids to center of assigned points
4. **Repeat**: Steps 2-3 until convergence

### Implementation
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 2) * 10

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# Visualize results
plt.figure(figsize=(10, 8))
colors = ['red', 'blue', 'green', 'purple', 'orange']

for i in range(3):
    plt.scatter(X[cluster_labels == i, 0], X[cluster_labels == i, 1], 
                c=colors[i], label=f'Cluster {i+1}', alpha=0.7)

plt.scatter(centroids[:, 0], centroids[:, 1], 
            c='black', marker='x', s=200, linewidths=3, label='Centroids')
plt.title('K-Means Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
```

### Choosing Optimal Number of Clusters

#### 1. Elbow Method
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Calculate WCSS for different k values
wcss = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Within Cluster Sum of Squares)')
plt.show()
```

#### 2. Silhouette Analysis
```python
from sklearn.metrics import silhouette_score

silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Analysis for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Average Silhouette Score')
plt.show()
```

### K-Means Assumptions and Limitations
- **Assumptions**:
  - Clusters are spherical and similar size
  - Features are on similar scales
  - Number of clusters is known beforehand

- **Limitations**:
  - Sensitive to initialization
  - Struggles with non-spherical clusters
  - Affected by outliers
  - Requires preprocessing for categorical data

### Improving K-Means Performance
```python
# Feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use scaled data for clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
```

## 🛠️ Practical Tips for Clustering

### Data Preprocessing for Clustering
```python
# 1. Handle missing values
df.fillna(df.mean(), inplace=True)

# 2. Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Handle categorical variables
# Use appropriate encoding methods

# 4. Remove outliers if necessary
from sklearn.preprocessing import RobustScaler
robust_scaler = RobustScaler()
X_robust = robust_scaler.fit_transform(X)
```

### Evaluating Clustering Results
```python
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Internal validation (no ground truth needed)
silhouette = silhouette_score(X, cluster_labels)
inertia = kmeans.inertia_

# External validation (if ground truth available)
# ari = adjusted_rand_score(true_labels, cluster_labels)

print(f"Silhouette Score: {silhouette:.4f}")
print(f"Inertia: {inertia:.4f}")
```

## Key Takeaways
- Choose encoding methods based on data type and algorithm requirements
- Advanced algorithms like SVM and Random Forest can handle complex patterns
- Random Forest provides built-in feature importance
- Unsupervised learning discovers hidden patterns without labels
- K-Means is simple but has assumptions about cluster shape and size
- Always preprocess data appropriately for clustering
- Use multiple methods to determine optimal number of clusters

## Home Task Preparation
**Topic**: K-Means Clustering Revision
- **Key Concepts to Review**:
  - K-Means algorithm steps
  - Choosing optimal k (elbow method, silhouette analysis)
  - Handling different data types in clustering
  - Alternative clustering algorithms (hierarchical, DBSCAN)
  - Cluster validation techniques
