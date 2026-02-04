# Data Science Cheat Sheet

A comprehensive reference guide for data science concepts, tools, and techniques.

## Table of Contents
- [Python Basics](#python-basics)
- [NumPy](#numpy)
- [Pandas](#pandas)
- [Data Visualization](#data-visualization)
- [Statistics & Probability](#statistics--probability)
- [Data Preprocessing](#data-preprocessing)
- [Machine Learning with Scikit-learn](#machine-learning-with-scikit-learn)
- [Model Evaluation](#model-evaluation)
- [Deep Learning Basics](#deep-learning-basics)

---

## Python Basics

### Essential Data Structures
```python
# Lists
my_list = [1, 2, 3, 4, 5]
my_list.append(6)
my_list[0]  # Access first element

# Dictionaries
my_dict = {'key1': 'value1', 'key2': 'value2'}
my_dict['key1']  # Access value

# Sets
my_set = {1, 2, 3}
my_set.add(4)

# Tuples (immutable)
my_tuple = (1, 2, 3)
```

### List Comprehensions
```python
# Basic list comprehension
squares = [x**2 for x in range(10)]

# With condition
even_squares = [x**2 for x in range(10) if x % 2 == 0]

# Dictionary comprehension
square_dict = {x: x**2 for x in range(5)}
```

### Lambda Functions
```python
# Lambda function
square = lambda x: x**2

# With map
result = list(map(lambda x: x**2, [1, 2, 3, 4]))

# With filter
even = list(filter(lambda x: x % 2 == 0, [1, 2, 3, 4, 5, 6]))
```

---

## NumPy

### Array Creation
```python
import numpy as np

# Create arrays
arr = np.array([1, 2, 3, 4, 5])
zeros = np.zeros((3, 4))
ones = np.ones((2, 3))
identity = np.eye(3)
random = np.random.random((3, 3))
arange = np.arange(0, 10, 2)  # Start, stop, step
linspace = np.linspace(0, 1, 5)  # Start, stop, number of points
```

### Array Operations
```python
# Basic operations
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

addition = arr1 + arr2
subtraction = arr1 - arr2
multiplication = arr1 * arr2
division = arr1 / arr2

# Matrix operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

matrix_mult = np.dot(A, B)  # or A @ B
transpose = A.T
inverse = np.linalg.inv(A)
determinant = np.linalg.det(A)
```

### Array Indexing and Slicing
```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Indexing
element = arr[0, 1]  # Row 0, Column 1

# Slicing
row = arr[0, :]  # First row
col = arr[:, 1]  # Second column
subarray = arr[0:2, 1:3]  # Rows 0-1, Columns 1-2

# Boolean indexing
mask = arr > 5
filtered = arr[mask]
```

### Useful Functions
```python
# Statistical functions
mean = np.mean(arr)
median = np.median(arr)
std = np.std(arr)
var = np.var(arr)
min_val = np.min(arr)
max_val = np.max(arr)
sum_val = np.sum(arr)

# Reshape
reshaped = arr.reshape(3, 3)

# Concatenate
concat = np.concatenate([arr1, arr2])
vstack = np.vstack([arr1, arr2])
hstack = np.hstack([arr1, arr2])
```

---

## Pandas

### DataFrame Creation
```python
import pandas as pd

# From dictionary
data = {'col1': [1, 2, 3], 'col2': [4, 5, 6]}
df = pd.DataFrame(data)

# From CSV
df = pd.read_csv('file.csv')

# From Excel
df = pd.read_excel('file.xlsx')

# From JSON
df = pd.read_json('file.json')
```

### DataFrame Basics
```python
# View data
df.head()  # First 5 rows
df.tail()  # Last 5 rows
df.info()  # DataFrame info
df.describe()  # Statistical summary
df.shape  # (rows, columns)
df.columns  # Column names
df.dtypes  # Data types
```

### Data Selection
```python
# Select columns
df['col1']  # Single column (Series)
df[['col1', 'col2']]  # Multiple columns (DataFrame)

# Select rows by index
df.loc[0]  # By label
df.iloc[0]  # By integer position

# Boolean indexing
df[df['col1'] > 2]

# Query
df.query('col1 > 2 and col2 < 10')
```

### Data Manipulation
```python
# Add column
df['new_col'] = df['col1'] + df['col2']

# Drop column
df = df.drop('col1', axis=1)

# Drop row
df = df.drop(0, axis=0)

# Rename columns
df = df.rename(columns={'old_name': 'new_name'})

# Sort values
df = df.sort_values('col1', ascending=False)

# Group by
grouped = df.groupby('category').mean()

# Merge/Join
merged = pd.merge(df1, df2, on='key', how='inner')  # inner, outer, left, right

# Concatenate
concatenated = pd.concat([df1, df2], axis=0)  # Vertically
```

### Handling Missing Data
```python
# Check for missing values
df.isnull()
df.isnull().sum()

# Drop missing values
df = df.dropna()  # Drop rows with any NaN
df = df.dropna(axis=1)  # Drop columns with any NaN

# Fill missing values
df = df.fillna(0)  # Fill with 0
df = df.fillna(df.mean())  # Fill with mean
df = df.fillna(method='ffill')  # Forward fill
df = df.fillna(method='bfill')  # Backward fill
```

### Data Transformation
```python
# Apply function
df['col1'] = df['col1'].apply(lambda x: x**2)

# Map values
df['category'] = df['category'].map({'A': 1, 'B': 2, 'C': 3})

# Pivot table
pivot = df.pivot_table(values='value', index='row', columns='col', aggfunc='mean')

# Melt (unpivot)
melted = pd.melt(df, id_vars=['id'], value_vars=['col1', 'col2'])
```

---

## Data Visualization

### Matplotlib
```python
import matplotlib.pyplot as plt

# Line plot
plt.plot(x, y)
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Title')
plt.legend()
plt.show()

# Scatter plot
plt.scatter(x, y, c='red', marker='o')
plt.show()

# Bar plot
plt.bar(categories, values)
plt.show()

# Histogram
plt.hist(data, bins=20)
plt.show()

# Subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].plot(x, y)
axes[0, 1].scatter(x, y)
plt.tight_layout()
plt.show()
```

### Seaborn
```python
import seaborn as sns

# Set style
sns.set_style('whitegrid')

# Distribution plot
sns.histplot(data, kde=True)

# Box plot
sns.boxplot(x='category', y='value', data=df)

# Violin plot
sns.violinplot(x='category', y='value', data=df)

# Heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

# Pair plot
sns.pairplot(df, hue='target')

# Count plot
sns.countplot(x='category', data=df)
```

---

## Statistics & Probability

### Descriptive Statistics
```python
import numpy as np
from scipy import stats

# Measures of central tendency
mean = np.mean(data)
median = np.median(data)
mode = stats.mode(data)

# Measures of spread
variance = np.var(data)
std_dev = np.std(data)
range_val = np.ptp(data)  # Peak to peak (max - min)

# Quartiles and percentiles
q1 = np.percentile(data, 25)
q2 = np.percentile(data, 50)  # Median
q3 = np.percentile(data, 75)
iqr = q3 - q1  # Interquartile range
```

### Probability Distributions
```python
from scipy import stats

# Normal distribution
mean, std = 0, 1
normal = stats.norm(mean, std)
pdf = normal.pdf(x)  # Probability density function
cdf = normal.cdf(x)  # Cumulative distribution function

# Binomial distribution
n, p = 10, 0.5
binomial = stats.binom(n, p)

# Poisson distribution
mu = 3
poisson = stats.poisson(mu)
```

### Hypothesis Testing
```python
# T-test
t_stat, p_value = stats.ttest_ind(group1, group2)

# Chi-square test
chi2, p_value = stats.chisquare(observed, expected)

# Correlation
correlation, p_value = stats.pearsonr(x, y)  # Pearson
correlation, p_value = stats.spearmanr(x, y)  # Spearman
```

---

## Data Preprocessing

### Feature Scaling
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization (Z-score normalization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Min-Max scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

### Encoding Categorical Variables
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

# Label Encoding
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])

# One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['category'])

# OneHotEncoder from sklearn
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
encoded = encoder.fit_transform(df[['category']])
```

### Train-Test Split
```python
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Handling Imbalanced Data
```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# SMOTE (Synthetic Minority Over-sampling)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Random Under-sampling
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)
```

---

## Machine Learning with Scikit-learn

### Linear Regression
```python
from sklearn.linear_model import LinearRegression

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Coefficients
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
```

### Logistic Regression
```python
from sklearn.linear_model import LogisticRegression

# Create and train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)
```

### Decision Trees
```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Classification
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Regression
reg = DecisionTreeRegressor(max_depth=5, random_state=42)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
```

### Random Forest
```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Classification
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)

# Feature importance
importances = rf_clf.feature_importances_
```

### Support Vector Machines
```python
from sklearn.svm import SVC, SVR

# Classification
svm_clf = SVC(kernel='rbf', C=1.0, random_state=42)
svm_clf.fit(X_train, y_train)
y_pred = svm_clf.predict(X_test)

# Regression
svm_reg = SVR(kernel='rbf', C=1.0)
svm_reg.fit(X_train, y_train)
y_pred = svm_reg.predict(X_test)
```

### K-Nearest Neighbors
```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Classification
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
```

### K-Means Clustering
```python
from sklearn.cluster import KMeans

# Create and fit model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Predict clusters
labels = kmeans.predict(X)
centroids = kmeans.cluster_centers_
```

### Principal Component Analysis (PCA)
```python
from sklearn.decomposition import PCA

# Dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Explained variance
explained_variance = pca.explained_variance_ratio_
```

### Cross-Validation
```python
from sklearn.model_selection import cross_val_score, KFold

# Cross-validation
scores = cross_val_score(model, X, y, cv=5)
print(f"Mean score: {scores.mean()}")
print(f"Std deviation: {scores.std()}")

# K-Fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold)
```

### Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)

# Best parameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Random Search
from scipy.stats import randint
param_distributions = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(5, 20)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions,
    n_iter=10,
    cv=5,
    random_state=42
)
random_search.fit(X_train, y_train)
```

---

## Model Evaluation

### Classification Metrics
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# Basic metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Classification report
report = classification_report(y_test, y_pred)

# ROC AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
```

### Regression Metrics
```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_squared_log_error
)

# MSE and RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# MAE
mae = mean_absolute_error(y_test, y_pred)

# R² Score
r2 = r2_score(y_test, y_pred)

# MSLE
msle = mean_squared_log_error(y_test, y_pred)
```

### Plotting Metrics
```python
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Confusion Matrix Plot
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()

# ROC Curve
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

---

## Deep Learning Basics

### TensorFlow/Keras
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Sequential model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    layers.Dropout(0.5),
    layers.Dense(32, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)

# Predict
predictions = model.predict(X_test)
```

### PyTorch
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
```

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.