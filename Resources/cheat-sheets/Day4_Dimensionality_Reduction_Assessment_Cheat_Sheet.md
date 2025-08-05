# Day 4: Dimensionality Reduction & Assessment Cheat Sheet

## Principal Component Analysis (PCA)

### What is PCA?
- **Definition**: Dimensionality reduction technique that transforms data to lower dimensions while preserving maximum variance
- **Goal**: Find principal components (directions of maximum variance) in the data
- **Linear transformation**: Creates new features that are linear combinations of original features

### Key Concepts
- **Principal Components**: New axes that capture maximum variance
- **Eigenvalues**: Amount of variance captured by each component
- **Eigenvectors**: Direction of each principal component
- **Explained Variance Ratio**: Proportion of total variance explained by each component

### When to Use PCA
- **High-dimensional data**: Reduce curse of dimensionality
- **Data visualization**: Project to 2D/3D for plotting
- **Noise reduction**: Remove less important components
- **Feature extraction**: Create new meaningful features
- **Storage/computation**: Reduce memory and processing requirements

### PCA Implementation
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Load sample data
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Standardize the data (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Get explained variance ratio
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("Explained Variance Ratio:", explained_variance)
print("Cumulative Variance:", cumulative_variance)

# Visualize explained variance
plt.figure(figsize=(12, 5))

# Subplot 1: Individual explained variance
plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained_variance) + 1), explained_variance)
plt.title('Explained Variance by Component')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')

# Subplot 2: Cumulative explained variance
plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
plt.legend()

plt.tight_layout()
plt.show()
```

### Choosing Number of Components
```python
# Method 1: Explained variance threshold (e.g., 95%)
def find_components_for_variance(pca, threshold=0.95):
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumsum >= threshold) + 1
    return n_components

n_comp_95 = find_components_for_variance(pca, 0.95)
print(f"Components needed for 95% variance: {n_comp_95}")

# Method 2: Kaiser criterion (eigenvalues > 1)
eigenvalues = pca.explained_variance_
n_comp_kaiser = np.sum(eigenvalues > 1)
print(f"Components with eigenvalues > 1: {n_comp_kaiser}")

# Method 3: Scree plot analysis
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.axhline(y=1, color='r', linestyle='--', label='Eigenvalue = 1')
plt.legend()
plt.show()
```

### PCA for Data Visualization
```python
# Reduce to 2D for visualization
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

# Create visualization
plt.figure(figsize=(10, 8))
colors = ['red', 'green', 'blue']
target_names = iris.target_names

for i, target_name in enumerate(target_names):
    plt.scatter(X_pca_2d[y == i, 0], X_pca_2d[y == i, 1], 
                c=colors[i], label=target_name, alpha=0.7)

plt.xlabel(f'First Principal Component ({pca_2d.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'Second Principal Component ({pca_2d.explained_variance_ratio_[1]:.1%} variance)')
plt.title('PCA - Iris Dataset')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Print component interpretations
components_df = pd.DataFrame(
    pca_2d.components_.T,
    columns=['PC1', 'PC2'],
    index=feature_names
)
print("Component Loadings:")
print(components_df)
```

### PCA Best Practices
```python
# Complete PCA workflow
def perform_pca_analysis(X, feature_names=None, target=None):
    # 1. Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Fit PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # 3. Analyze components
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    # 4. Find optimal number of components
    n_comp_95 = np.argmax(cumulative_var >= 0.95) + 1
    
    # 5. Create results dictionary
    results = {
        'pca_model': pca,
        'scaler': scaler,
        'transformed_data': X_pca,
        'explained_variance': explained_var,
        'cumulative_variance': cumulative_var,
        'n_components_95': n_comp_95
    }
    
    return results

# Example usage
pca_results = perform_pca_analysis(X, feature_names, y)
print(f"Components for 95% variance: {pca_results['n_components_95']}")
```

## 🛍️ Customer Segmentation Case Study: Mall Customer Dataset

### Problem Overview
- **Objective**: Segment mall customers based on spending behavior and demographics
- **Features**: Age, Annual Income, Spending Score
- **Method**: K-Means clustering with PCA visualization

### Complete Implementation
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load data (assuming mall_customers.csv is available)
# Sample data structure: CustomerID, Gender, Age, Annual Income, Spending Score
df = pd.read_csv('mall_customers.csv')

# Data exploration
print("Dataset Info:")
print(df.info())
print("\nDataset Description:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Visualize data distribution
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Age distribution
axes[0,0].hist(df['Age'], bins=20, alpha=0.7)
axes[0,0].set_title('Age Distribution')
axes[0,0].set_xlabel('Age')

# Annual Income distribution
axes[0,1].hist(df['Annual Income (k$)'], bins=20, alpha=0.7)
axes[0,1].set_title('Annual Income Distribution')
axes[0,1].set_xlabel('Annual Income (k$)')

# Spending Score distribution
axes[1,0].hist(df['Spending Score (1-100)'], bins=20, alpha=0.7)
axes[1,0].set_title('Spending Score Distribution')
axes[1,0].set_xlabel('Spending Score')

# Gender distribution
df['Gender'].value_counts().plot(kind='bar', ax=axes[1,1])
axes[1,1].set_title('Gender Distribution')
axes[1,1].set_xlabel('Gender')

plt.tight_layout()
plt.show()

# Prepare data for clustering
# Select features for clustering
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features].copy()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal number of clusters using elbow method
def find_optimal_clusters(X, max_k=10):
    wcss = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, cluster_labels))
    
    return k_range, wcss, silhouette_scores

k_range, wcss, silhouette_scores = find_optimal_clusters(X_scaled)

# Plot elbow curve and silhouette scores
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Elbow curve
axes[0].plot(k_range, wcss, marker='o')
axes[0].set_title('Elbow Method for Optimal k')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('WCSS')
axes[0].grid(True)

# Silhouette scores
axes[1].plot(k_range, silhouette_scores, marker='o', color='orange')
axes[1].set_title('Silhouette Analysis')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Average Silhouette Score')
axes[1].grid(True)

plt.tight_layout()
plt.show()

# Apply K-Means with optimal number of clusters (let's say k=5)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to dataframe
df['Cluster'] = cluster_labels

# Cluster analysis
print(f"\nCluster Analysis (k={optimal_k}):")
cluster_summary = df.groupby('Cluster')[features].mean()
print(cluster_summary)

# Cluster sizes
print(f"\nCluster Sizes:")
print(df['Cluster'].value_counts().sort_index())

# Apply PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# PCA visualization
axes[0,0].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
axes[0,0].set_title('Customer Segments (PCA)')
axes[0,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
axes[0,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')

# Income vs Spending Score
scatter = axes[0,1].scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], 
                           c=cluster_labels, cmap='viridis', alpha=0.7)
axes[0,1].set_title('Income vs Spending Score')
axes[0,1].set_xlabel('Annual Income (k$)')
axes[0,1].set_ylabel('Spending Score (1-100)')

# Age vs Spending Score
axes[0,2].scatter(df['Age'], df['Spending Score (1-100)'], 
                 c=cluster_labels, cmap='viridis', alpha=0.7)
axes[0,2].set_title('Age vs Spending Score')
axes[0,2].set_xlabel('Age')
axes[0,2].set_ylabel('Spending Score (1-100)')

# Age vs Income
axes[1,0].scatter(df['Age'], df['Annual Income (k$)'], 
                 c=cluster_labels, cmap='viridis', alpha=0.7)
axes[1,0].set_title('Age vs Annual Income')
axes[1,0].set_xlabel('Age')
axes[1,0].set_ylabel('Annual Income (k$)')

# Cluster characteristics heatmap
cluster_chars = df.groupby('Cluster')[features].mean()
sns.heatmap(cluster_chars.T, annot=True, cmap='YlOrRd', ax=axes[1,1])
axes[1,1].set_title('Cluster Characteristics Heatmap')

# 3D visualization (if using 3 features)
ax_3d = fig.add_subplot(2, 3, 6, projection='3d')
scatter_3d = ax_3d.scatter(df['Age'], df['Annual Income (k$)'], 
                          df['Spending Score (1-100)'], 
                          c=cluster_labels, cmap='viridis', alpha=0.7)
ax_3d.set_xlabel('Age')
ax_3d.set_ylabel('Annual Income (k$)')
ax_3d.set_zlabel('Spending Score')
ax_3d.set_title('3D Customer Segments')

plt.tight_layout()
plt.show()

# Interpret clusters
def interpret_clusters(df, cluster_col='Cluster'):
    interpretations = {}
    
    for cluster in df[cluster_col].unique():
        cluster_data = df[df[cluster_col] == cluster]
        
        avg_age = cluster_data['Age'].mean()
        avg_income = cluster_data['Annual Income (k$)'].mean()
        avg_spending = cluster_data['Spending Score (1-100)'].mean()
        size = len(cluster_data)
        
        # Create interpretation based on characteristics
        if avg_income > 70 and avg_spending > 70:
            interpretation = "High Value Customers"
        elif avg_income < 40 and avg_spending < 40:
            interpretation = "Low Value Customers"
        elif avg_income > 70 and avg_spending < 40:
            interpretation = "Conservative High Earners"
        elif avg_income < 40 and avg_spending > 70:
            interpretation = "Impulsive Spenders"
        else:
            interpretation = "Moderate Customers"
        
        interpretations[cluster] = {
            'name': interpretation,
            'avg_age': avg_age,
            'avg_income': avg_income,
            'avg_spending': avg_spending,
            'size': size,
            'percentage': (size / len(df)) * 100
        }
    
    return interpretations

cluster_interpretations = interpret_clusters(df)

print("\nCluster Interpretations:")
for cluster, info in cluster_interpretations.items():
    print(f"\nCluster {cluster}: {info['name']}")
    print(f"  Size: {info['size']} customers ({info['percentage']:.1f}%)")
    print(f"  Average Age: {info['avg_age']:.1f}")
    print(f"  Average Income: ${info['avg_income']:.1f}k")
    print(f"  Average Spending Score: {info['avg_spending']:.1f}")
```

### Business Insights and Recommendations
```python
def generate_business_recommendations(cluster_interpretations):
    recommendations = {}
    
    for cluster, info in cluster_interpretations.items():
        name = info['name']
        
        if "High Value" in name:
            rec = [
                "Provide premium products and services",
                "Offer VIP loyalty programs",
                "Focus on retention strategies",
                "Upsell premium brands"
            ]
        elif "Low Value" in name:
            rec = [
                "Offer budget-friendly options",
                "Implement discount campaigns",
                "Focus on value proposition",
                "Target with affordable product lines"
            ]
        elif "Conservative" in name:
            rec = [
                "Highlight quality and durability",
                "Offer investment-worthy products",
                "Focus on practical benefits",
                "Provide detailed product information"
            ]
        elif "Impulsive" in name:
            rec = [
                "Create attractive promotions",
                "Use emotional marketing appeals",
                "Implement flash sales",
                "Focus on immediate gratification"
            ]
        else:
            rec = [
                "Balanced marketing approach",
                "Mix of value and premium options",
                "Seasonal promotions",
                "Focus on customer journey optimization"
            ]
        
        recommendations[cluster] = rec
    
    return recommendations

business_recs = generate_business_recommendations(cluster_interpretations)

print("\nBusiness Recommendations by Cluster:")
for cluster, recs in business_recs.items():
    cluster_name = cluster_interpretations[cluster]['name']
    print(f"\nCluster {cluster} ({cluster_name}):")
    for i, rec in enumerate(recs, 1):
        print(f"  {i}. {rec}")
```

## 📊 Mini Project Guidelines

### Project Structure
```
mini_project/
├── data/
│   └── dataset.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── evaluation.py
├── results/
│   ├── models/
│   ├── plots/
│   └── reports/
└── README.md
```

### Project Checklist
1. **Problem Definition** ✓
   - [ ] Clear problem statement
   - [ ] Define success metrics
   - [ ] Identify stakeholders

2. **Data Collection & Understanding** ✓
   - [ ] Load and explore dataset
   - [ ] Check data quality
   - [ ] Identify missing values and outliers
   - [ ] Understand feature relationships

3. **Data Preprocessing** ✓
   - [ ] Handle missing values
   - [ ] Remove/treat outliers
   - [ ] Encode categorical variables
   - [ ] Scale numerical features
   - [ ] Feature engineering

4. **Exploratory Data Analysis** ✓
   - [ ] Univariate analysis
   - [ ] Bivariate analysis
   - [ ] Feature correlation analysis
   - [ ] Visualization of key patterns

5. **Model Development** ✓
   - [ ] Split data appropriately
   - [ ] Try multiple algorithms
   - [ ] Hyperparameter tuning
   - [ ] Cross-validation

6. **Model Evaluation** ✓
   - [ ] Use appropriate metrics
   - [ ] Compare model performance
   - [ ] Analyze feature importance
   - [ ] Validate on test set

7. **Results Interpretation** ✓
   - [ ] Business insights
   - [ ] Model limitations
   - [ ] Recommendations
   - [ ] Future improvements

### Sample Project Template
```python
# Mini Project Template: End-to-End ML Pipeline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

class MLProject:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.models = {}
        self.results = {}
    
    def load_data(self):
        """Load and initial data exploration"""
        self.df = pd.read_csv(self.data_path)
        print("Dataset Shape:", self.df.shape)
        print("\nDataset Info:")
        print(self.df.info())
        print("\nMissing Values:")
        print(self.df.isnull().sum())
        return self.df
    
    def explore_data(self):
        """Comprehensive EDA"""
        # Summary statistics
        print(self.df.describe())
        
        # Visualizations
        # Add your visualization code here
        pass
    
    def preprocess_data(self, target_column):
        """Data preprocessing pipeline"""
        # Separate features and target
        self.X = self.df.drop(target_column, axis=1)
        self.y = self.df[target_column]
        
        # Handle missing values
        # Add preprocessing steps here
        
        return self.X, self.y
    
    def train_models(self):
        """Train multiple models"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Initialize models
        self.models = {
            'RandomForest': RandomForestClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42)
        }
        
        # Train and evaluate models
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Store results
            self.results[name] = {
                'model': model,
                'predictions': y_pred,
                'test_data': (X_test, y_test)
            }
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        for name, result in self.results.items():
            model = result['model']
            y_pred = result['predictions']
            X_test, y_test = result['test_data']
            
            print(f"\n{name} Results:")
            print(classification_report(y_test, y_pred))
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X, self.y, cv=5)
            print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    def visualize_results(self):
        """Create result visualizations"""
        # Add visualization code here
        pass

# Usage example
# project = MLProject('your_dataset.csv')
# project.load_data()
# project.explore_data()
# project.preprocess_data('target_column')
# project.train_models()
# project.evaluate_models()
```

## key Takeaways
- PCA is powerful for dimensionality reduction and visualization
- Always standardize data before applying PCA
- Customer segmentation provides valuable business insights
- K-Means + PCA is an effective combination for clustering
- End-to-end ML projects require systematic approach
- Proper evaluation and interpretation are crucial
- Peer feedback enhances learning and improvement

## Final Workshop Summary
1. **Day 1**: ML fundamentals and linear regression
2. **Day 2**: Classification algorithms and evaluation metrics
3. **Day 3**: Advanced algorithms and unsupervised learning introduction
4. **Day 4**: Dimensionality reduction, clustering applications, and project assessment

**Congratulations on completing the 4-day Machine Learning Training! 🎉**
