# 1

```
import pandas as pd
csv_file = 'Titanic.csv'  # Replace with your file path

# Reading data from CSV
df = pd.read_csv(csv_file)
print("Data loaded from CSV:")
print(df.head())

selected_data = df[['Name', 'Age']]
print("\nSelected Columns (Name and Age):")
print(selected_data.head())

# Indexing - Set 'PassengerId' as index
indexed_data = df.set_index('PassengerId')
print("\nData with PassengerId as Index:")
print(indexed_data.head())

# Sorting data by 'Age' (ascending order)
sorted_data = df.sort_values(by='Age', ascending=True)
print("\nData sorted by Age:")
print(sorted_data[['Name', 'Age']].head())

# Describe attributes of the data
description = df.describe()
print("\nStatistical Description of the Data:")
print(description)

# Check data types of each column
data_types = df.dtypes
print("\nData Types of Each Column:")
print(data_types)
```
______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

# 2

```
import pandas as pd
file_path = 'Telecom Churn.csv'  # Replace with your file path if needed
df = pd.read_csv(file_path)

# Displaying the dataset structure
print("Dataset Loaded Successfully. Here's a preview:")
print(df.head())

numeric_columns = df.select_dtypes(include=['number']).columns

# Calculations for each statistic
print("\nSummary Statistics for Numeric Features:\n")

for column in numeric_columns:
    print(f"Feature: {column}")
    min_value = df[column].min()
    max_value = df[column].max()
    mean_value = df[column].mean()
    range_value = max_value - min_value
    std_dev = df[column].std()
    variance = df[column].var()
    percentiles = df[column].quantile([0.25, 0.5, 0.75])

    print(f"  Minimum: {min_value}")
    print(f"  Maximum: {max_value}")
    print(f"  Mean: {mean_value}")
    print(f"  Range: {range_value}")
    print(f"  Standard Deviation: {std_dev}")
    print(f"  Variance: {variance}")
    print(f"  25th Percentile: {percentiles[0.25]}")
    print(f"  50th Percentile (Median): {percentiles[0.5]}")
    print(f"  75th Percentile: {percentiles[0.75]}")
    print("\n")
```
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

# 3

```
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'House Data.csv'  # Replace with your dataset path
df = pd.read_csv(file_path)

# Display dataset preview
print("Dataset Loaded Successfully. Here's a preview:")
print(df.head())

# List numeric columns
numeric_columns = df.select_dtypes(include=['number']).columns

# Compute standard deviation, variance, and percentiles
print("\nStatistics for Numeric Features:\n")
for column in numeric_columns:
    std_dev = df[column].std()
    variance = df[column].var()
    percentiles = np.percentile(df[column].dropna(), [25, 50, 75])  # 25th, 50th, 75th percentiles

    print(f"Feature: {column}")
    print(f"  Standard Deviation: {std_dev}")
    print(f"  Variance: {variance}")
    print(f"  25th Percentile: {percentiles[0]}")
    print(f"  50th Percentile (Median): {percentiles[1]}")
    print(f"  75th Percentile: {percentiles[2]}")
    print("\n")

# Create histograms for all features
print("Generating histograms...")
for column in df.columns:
    plt.figure(figsize=(8, 5))
    plt.hist(df[column].dropna(), bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
```
________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
# 4

```
import pandas as pd
import numpy as np

# Load the dataset
file_path = "Datasets/Lipstick.csv"
data = pd.read_csv(file_path)

# Drop unnecessary columns (e.g., 'Id')
data = data.drop(columns=["Id"])

# Function to calculate entropy
def entropy(column):
    """Calculate entropy for a column."""
    probabilities = column.value_counts(normalize=True)
    return -np.sum(probabilities * np.log2(probabilities))

# Function to calculate information gain
def information_gain(data, split_attribute, target_attribute):
    """Calculate information gain for a split attribute."""
    total_entropy = entropy(data[target_attribute])
    values, counts = np.unique(data[split_attribute], return_counts=True)
    weighted_entropy = 0
    for i in range(len(values)):
        subset = data[data[split_attribute] == values[i]]
        weighted_entropy += (counts[i] / np.sum(counts)) * entropy(subset[target_attribute])
    return total_entropy - weighted_entropy

# Target variable entropy
target_entropy = entropy(data['Buys'])

# Calculate Information Gain for all features
features = ['Age', 'Income', 'Gender', 'Ms']
ig_values = {feature: information_gain(data, feature, 'Buys') for feature in features}

# Identify the root node
root_node = max(ig_values, key=ig_values.get)

# Output the results
print("Entropy of target variable (Buys):", target_entropy)
print("Information Gain for each feature:", ig_values)
print("Root node of the decision tree:", root_node)
```
_____________________________________________________________________________________________________________________________________________________________________

# 5

```
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
file_path = 'Datasets/Lipstick.csv'
data = pd.read_csv(file_path)

# Preprocess: Drop 'Id' and encode categorical features
data = data.drop(columns=["Id"])
label_encoders = {}
for column in data.columns:
    if data[column].dtype == 'object':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

# Separate features and target
X = data.drop(columns=["Buys"])
y = data["Buys"]

# Train decision tree classifier
model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X, y)

# Define test data
test_data = pd.DataFrame([["<21", "Low", "Female", "Married"]], columns=["Age", "Income", "Gender", "Ms"])

# Encode test data using the same label encoders
for column in test_data.columns:
    test_data[column] = label_encoders[column].transform(test_data[column])

# Predict the result
prediction = model.predict(test_data)
predicted_label = label_encoders["Buys"].inverse_transform(prediction)

# Output the prediction
print("Test Data:")
print( test_data)

print("Prediction for the test data:", predicted_label[0])
```
_____________________________________________________________________________________________________________________________________________________________________


# 6

```
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset from a CSV file
csv_file = 'Lipstick.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_file)

print("Dataset loaded:")
print(df.head())


df = df.drop(columns=['Id'])

# Encode categorical variables using LabelEncoder
encoders = {}  # To store LabelEncoders for each column
encoded_df = df.copy()

for column in encoded_df.columns:
    encoders[column] = LabelEncoder()
    encoded_df[column] = encoders[column].fit_transform(encoded_df[column])

# Split data into features (X) and target (y)
X = encoded_df.drop('Buys', axis=1)  # Features (all except 'Buys')
y = encoded_df['Buys']  # Target (Buys column)

# Train the decision tree classifier
model = DecisionTreeClassifier(criterion='entropy', random_state=0)
model.fit(X, y)

# Prepare the test data
# Test case: [Age < 21, Income = Low, Gender = Female, Marital Status = Married]
test_data_dict = {
    'Age': '>35',
    'Income': 'Medium',
    'Gender': 'Female',
    'Ms': 'Married'
}

# Encode the test data using the same encoders
test_data = {key: encoders[key].transform([value])[0] for key, value in test_data_dict.items()}
test_data_df = pd.DataFrame([test_data])

# Predict using the trained model
prediction = model.predict(test_data_df)
predicted_label = encoders['Buys'].inverse_transform(prediction)[0]  # Convert numeric prediction back to label

# Output the prediction
print(f"The decision for the test data is: {predicted_label}")
```
________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

# 7

```
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset from a CSV file
csv_file = 'Lipstick.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_file)

print("Dataset loaded:")
print(df.head())

df = df.drop(columns=['Id'])

# Encode categorical variables using LabelEncoder
encoders = {}  # To store LabelEncoders for each column
encoded_df = df.copy()

for column in encoded_df.columns:
    encoders[column] = LabelEncoder()
    encoded_df[column] = encoders[column].fit_transform(encoded_df[column])

# Split data into features (X) and target (y)
X = encoded_df.drop('Buys', axis=1)  # Features (all except 'Buys')
y = encoded_df['Buys']  # Target (Buys column)

# Train the decision tree classifier
model = DecisionTreeClassifier(criterion='entropy', random_state=0)
model.fit(X, y)

# Prepare the test data
# Test case: [Age < 21, Income = Low, Gender = Female, Marital Status = Married]
test_data_dict = {
    'Age': '<21',
    'Income': 'Low',
    'Gender': 'Female',
    'Ms': 'Married'
}

# Encode the test data using the same encoders
test_data = {key: encoders[key].transform([value])[0] for key, value in test_data_dict.items()}
test_data_df = pd.DataFrame([test_data])

# Predict using the trained model
prediction = model.predict(test_data_df)
predicted_label = encoders['Buys'].inverse_transform(prediction)[0]  # Convert numeric prediction back to label

# Output the prediction
print(f"The decision for the test data is: {predicted_label}")
```
________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

# 8

```
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset from a CSV file
csv_file = 'Lipstick.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_file)

print("Dataset loaded:")
print(df.head())

df = df.drop(columns=['Id'])

# Encode categorical variables using LabelEncoder
encoders = {}  # Store LabelEncoders for each column
encoded_df = df.copy()

for column in encoded_df.columns:
    encoders[column] = LabelEncoder()
    encoded_df[column] = encoders[column].fit_transform(encoded_df[column])

# Split data into features (X) and target (y)
X = encoded_df.drop('Buys', axis=1)  # Features (all except 'Buys')
y = encoded_df['Buys']  # Target (Buys column)

# Train the decision tree classifier
model = DecisionTreeClassifier(criterion='entropy', random_state=0)
model.fit(X, y)

# Test data
test_data_dict = {
    'Age': '21-35',
    'Income': 'Low',
    'Gender': 'Male',
    'Ms': 'Married'
}

# Encode the test data using the same encoders
test_data = {key: encoders[key].transform([value])[0] for key, value in test_data_dict.items()}
test_data_df = pd.DataFrame([test_data])

# Predict using the trained model
prediction = model.predict(test_data_df)
predicted_label = encoders['Buys'].inverse_transform(prediction)[0]  # Convert numeric prediction back to label

# Output the prediction
print(f"The decision for the test data is: {predicted_label}")
```

________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

# 9

```
import numpy as np

# Given points
points = np.array([
    [0.1, 0.6],  # P1
    [0.15, 0.71],  # P2
    [0.08, 0.9],  # P3
    [0.16, 0.85],  # P4
    [0.2, 0.3],  # P5
    [0.25, 0.5],  # P6
    [0.24, 0.1],  # P7
    [0.3, 0.2]   # P8
])

# Initial centroids
m1 = points[0]  # P1
m2 = points[7]  # P8

# Helper function to calculate Euclidean distance
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

# Perform clustering
clusters = {1: [], 2: []}

for point in points:
    # Calculate distance from m1 and m2
    d1 = euclidean_distance(point, m1)
    d2 = euclidean_distance(point, m2)
    # Assign point to the nearest cluster
    if d1 < d2:
        clusters[1].append(point)
    else:
        clusters[2].append(point)

# Convert cluster lists to numpy arrays for calculations
clusters[1] = np.array(clusters[1])
clusters[2] = np.array(clusters[2])

# Calculate updated centroids
new_m1 = np.mean(clusters[1], axis=0)
new_m2 = np.mean(clusters[2], axis=0)

# Determine answers to the questions
p6_cluster = 1 if any((clusters[1] == points[5]).all(axis=1)) else 2
m2_population = len(clusters[2])

# Print results
print(f"1] P6 belongs to Cluster #{p6_cluster}")
print(f"2] Population around m2 (Cluster #2): {m2_population}")
print(f"3] Updated centroids: m1 = {new_m1}, m2 = {new_m2}")
```
________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

# 10

```
import numpy as np

# Given points
points = np.array([
    [2, 10],  # P1
    [2, 5],   # P2
    [8, 4],   # P3
    [5, 8],   # P4
    [7, 5],   # P5
    [6, 4],   # P6
    [1, 2],   # P7
    [4, 9]    # P8
])

# Initial centroids
m1 = points[0]  # P1
m2 = points[3]  # P4
m3 = points[6]  # P7

# Helper function to calculate Euclidean distance
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

# Perform clustering
clusters = {1: [], 2: [], 3: []}

for point in points:
    # Calculate distance from each centroid
    d1 = euclidean_distance(point, m1)
    d2 = euclidean_distance(point, m2)
    d3 = euclidean_distance(point, m3)
    # Assign point to the nearest cluster
    if d1 < d2 and d1 < d3:
        clusters[1].append(point)
    elif d2 < d1 and d2 < d3:
        clusters[2].append(point)
    else:
        clusters[3].append(point)

# Convert cluster lists to numpy arrays for calculations
clusters[1] = np.array(clusters[1])
clusters[2] = np.array(clusters[2])
clusters[3] = np.array(clusters[3])

# Calculate updated centroids
new_m1 = np.mean(clusters[1], axis=0)
new_m2 = np.mean(clusters[2], axis=0)
new_m3 = np.mean(clusters[3], axis=0)

# Determine answers to the questions
p6_cluster = (
    1 if any((clusters[1] == points[5]).all(axis=1))
    else 2 if any((clusters[2] == points[5]).all(axis=1))
    else 3
)
m3_population = len(clusters[3])

# Print results
print(f"1] P6 belongs to Cluster #{p6_cluster}")
print(f"2] Population around m3 (Cluster #3): {m3_population}")
print(f"3] Updated centroids: m1 = {new_m1}, m2 = {new_m2}, m3 = {new_m3}")
```
________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

# 11

```
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = "IRIS.csv"  # Update with the correct path if needed
iris_data = pd.read_csv(file_path)

# Step 1: List features and their types
features = iris_data.columns
feature_types = iris_data.dtypes

print("Features and their types:")
for feature, ftype in zip(features, feature_types):
    print(f"{feature}: {'Numeric' if ftype in ['float64', 'int64'] else 'Nominal'}")

# Step 2: Create histograms for numeric features
numeric_features = iris_data.select_dtypes(include=['float64', 'int64']).columns

for feature in numeric_features:
    plt.figure(figsize=(6, 4))
    plt.hist(iris_data[feature], bins=15, color='skyblue', edgecolor='black')
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
```

________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

# 12

```
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = "IRIS.csv"  # Update with the correct path if needed
iris_data = pd.read_csv(file_path)

# Step 1: Create box plots for numeric features
numeric_features = iris_data.select_dtypes(include=['float64', 'int64']).columns

for feature in numeric_features:
    plt.figure(figsize=(6, 4))
    plt.boxplot(iris_data[feature], vert=False, patch_artist=True, 
                boxprops=dict(facecolor='skyblue', color='black'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
                medianprops=dict(color='red'))
    plt.title(f"Box Plot of {feature}")
    plt.xlabel(feature)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()

# Step 2: Analyze distributions and identify outliers
for feature in numeric_features:
    Q1 = iris_data[feature].quantile(0.25)
    Q3 = iris_data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = iris_data[(iris_data[feature] < lower_bound) | (iris_data[feature] > upper_bound)]
    
    print(f"\nFeature: {feature}")
    print(f"  Q1: {Q1}, Q3: {Q3}")
    print(f"  IQR: {IQR}")
    print(f"  Outlier Bounds: {lower_bound} to {upper_bound}")
    print(f"  Number of Outliers: {outliers.shape[0]}")
```
________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

# 13

```
import pandas as pd

# Load the dataset
data = pd.read_csv('/content/Covid Vaccine Statewise.csv')

# Convert 'Updated On' to datetime and sort the data by state and date
data['Updated On'] = pd.to_datetime(data['Updated On'], format='%d/%m/%Y')
data = data.sort_values(by=['State', 'Updated On'])

# Calculate daily doses for first and second doses
data['Daily First Dose'] = data.groupby('State')['First Dose Administered'].diff().fillna(0)
data['Daily Second Dose'] = data.groupby('State')['Second Dose Administered'].diff().fillna(0)

# b. Number of persons state-wise vaccinated for the first dose (daily total)
statewise_daily_first_dose = data.groupby('State')['Daily First Dose'].sum()

# c. Number of persons state-wise vaccinated for the second dose (daily total)
statewise_daily_second_dose = data.groupby('State')['Daily Second Dose'].sum()

# Print results
print("\nState-wise daily total of first doses administered:")
print(statewise_daily_first_dose)

print("\nState-wise daily total of second doses administered:")
print(statewise_daily_second_dose)
```
________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

# 14

```
import pandas as pd

# Load the dataset
file_path = '/content/Covid Vaccine Statewise.csv'  # Update with your file path
df = pd.read_csv(file_path)

# A. Describe the dataset
print("Dataset Description:")
print(df.describe(include='all'))  # Provides summary statistics for all columns

# Information about the dataset structure
print("\nDataset Info:")
print(df.info())  # Provides information about columns and data types

# Convert 'Updated On' to datetime and sort the data by state and date
df['Updated On'] = pd.to_datetime(df['Updated On'], format='%d/%m/%Y')
df = df.sort_values(by=['State', 'Updated On'])

# Calculate daily doses for males and females
df['Daily Male Doses'] = df.groupby('State')['Male (Doses Administered)'].diff().fillna(0)
df['Daily Female Doses'] = df.groupby('State')['Female (Doses Administered)'].diff().fillna(0)

# B. Total number of males vaccinated (daily total)
total_male_vaccinated = df['Daily Male Doses'].sum()
print(f"\nTotal number of males vaccinated: {total_male_vaccinated}")

# C. Total number of females vaccinated (daily total)
total_female_vaccinated = df['Daily Female Doses'].sum()
print(f"Total number of females vaccinated: {total_female_vaccinated}")
```
________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

# 15

```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset from the CSV file
file_path = 'Titanic.csv'  # Update with the path to your Titanic CSV file
titanic = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
print(titanic.head())

# A. Heatmap of Missing Values to see which columns have missing data
plt.figure(figsize=(10, 6))
sns.heatmap(titanic.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# B. Distribution of Age of passengers
plt.figure(figsize=(8, 5))
sns.histplot(titanic['Age'].dropna(), kde=True, color='blue')
plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# C. Count plot of Survival based on Sex
plt.figure(figsize=(8, 5))
sns.countplot(data=titanic, x='Survived', hue='Sex')
plt.title("Survival Count by Sex")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.show()

# D. Boxplot of Age vs Survived to see age distribution for survivors and non-survivors
plt.figure(figsize=(8, 5))
sns.boxplot(data=titanic, x='Survived', y='Age')
plt.title("Age Distribution by Survival Status")
plt.xlabel("Survived")
plt.ylabel("Age")
plt.show()

# E. Pairplot to check relationships between features like Age, Fare, and Pclass
sns.pairplot(titanic[['Age', 'Fare', 'Pclass', 'Survived']], hue='Survived', palette='coolwarm')
plt.show()

# F. Heatmap to show correlation between numerical features

numeric_cols = titanic.select_dtypes(include=['number'])
# Plot the heatmap with only numeric columns
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Between Numerical Features")
plt.show()

# G. Count plot of Pclass vs Survived to see survival rates by class
plt.figure(figsize=(8, 5))
sns.countplot(data=titanic, x='Pclass', hue='Survived')
plt.title("Survival Count by Pclass")
plt.xlabel("Pclass")
plt.ylabel("Count")
plt.show()

# H. Bar plot for Embarked vs Survival to see survival rates based on embarkation port
plt.figure(figsize=(8, 5))
sns.barplot(data=titanic, x='Embarked', y='Survived')
plt.title("Survival Rate by Embarked Location")
plt.xlabel("Embarked")
plt.ylabel("Survival Rate")
plt.show()
```
________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

# 16

```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset from the CSV file
file_path = 'Titanic.csv'  # Update with the path to your Titanic CSV file
titanic = pd.read_csv(file_path)

# Plotting a histogram for the 'fare' column
plt.figure(figsize=(10, 6))  # Set the size of the figure
sns.histplot(titanic['Fare'], kde=True, color='skyblue', bins=30)  # Plot the histogram with KDE (Kernel Density Estimation)
plt.title('Distribution of Ticket Fare for Titanic Passengers')  # Title of the plot
plt.xlabel('Fare')  # X-axis label
plt.ylabel('Frequency')  # Y-axis label
plt.show()
```
________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________


# 17

```
# Given values
TP = 1
FP = 1
FN = 8
TN = 90

# Calculating metrics
accuracy = (TP + TN) / (TP + TN + FP + FN)
error_rate = 1 - accuracy
precision = TP / (TP + FP)
recall = TP / (TP + FN)

# Printing results
print(f"Accuracy: {accuracy:.2f}")
print(f"Error Rate: {error_rate:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
```
________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

# 18

```
import pandas as pd

# Load the dataset
file_path = '/content/House Data.csv'
data = pd.read_csv(file_path)

# Inspect the dataset to identify categorical and quantitative variables
print("Dataset Overview:")
print(data.info())

# Separate categorical and quantitative variables
categorical_cols = data.select_dtypes(include=['object']).columns
quantitative_cols = data.select_dtypes(include=['int64', 'float64']).columns

# Group quantitative variables by each categorical variable and calculate summary statistics
results = {}
for cat_col in categorical_cols:
    grouped_stats = data.groupby(cat_col)[quantitative_cols].agg(['mean', 'median', 'min', 'max', 'std'])
    results[cat_col] = grouped_stats

# Print results for each categorical variable
for cat_col, stats in results.items():
    print(f"\nSummary statistics for categorical variable '{cat_col}':")
    print(stats)
```
________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

# 19

```
import pandas as pd

# Load the Iris dataset (assuming iris.csv is in the same directory)
df = pd.read_csv("/content/IRIS.csv")

# Group by species and calculate basic statistics
grouped = df.groupby("species").agg({
    "sepal_length": ["mean", "std", lambda x: x.quantile(0.25), lambda x: x.quantile(0.5), lambda x: x.quantile(0.75)],
    "sepal_width": ["mean", "std", lambda x: x.quantile(0.25), lambda x: x.quantile(0.5), lambda x: x.quantile(0.75)],
    "petal_length": ["mean", "std", lambda x: x.quantile(0.25), lambda x: x.quantile(0.5), lambda x: x.quantile(0.75)],
    "petal_width": ["mean", "std", lambda x: x.quantile(0.25), lambda x: x.quantile(0.5), lambda x: x.quantile(0.75)]
})

# Rename columns for clarity
grouped.columns = ["_".join(col).strip() for col in grouped.columns]
grouped.rename(columns={
    "sepal_length_<lambda_0>": "sepal_length_25%",
    "sepal_length_<lambda_1>": "sepal_length_50%",
    "sepal_length_<lambda_2>": "sepal_length_75%",
    "sepal_width_<lambda_0>": "sepal_width_25%",
    "sepal_width_<lambda_1>": "sepal_width_50%",
    "sepal_width_<lambda_2>": "sepal_width_75%",
    "petal_length_<lambda_0>": "petal_length_25%",
    "petal_length_<lambda_1>": "petal_length_50%",
    "petal_length_<lambda_2>": "petal_length_75%",
    "petal_width_<lambda_0>": "petal_width_25%",
    "petal_width_<lambda_1>": "petal_width_50%",
    "petal_width_<lambda_2>": "petal_width_75%"
}, inplace=True)

# Display the statistical details
print(grouped)
```
________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

# 20

```
import pandas as pd
import numpy as np
import random

# Load the Iris dataset
df = pd.read_csv("/content/IRIS.csv")

# Extract only the numeric features for clustering
data = df.iloc[:, :-1].values  # Exclude the species column

# Set K and initialize cluster means randomly
K = 3
np.random.seed(42)  # For reproducibility
initial_means_indices = random.sample(range(len(data)), K)
cluster_means = data[initial_means_indices]

# Function to compute Euclidean distance
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

# K-means algorithm
for iteration in range(10):
    # Step 1: Assign each point to the nearest cluster
    clusters = [[] for _ in range(K)]
    for point in data:
        distances = [euclidean_distance(point, mean) for mean in cluster_means]
        cluster_index = np.argmin(distances)
        clusters[cluster_index].append(point)

    # Step 2: Update cluster means
    new_cluster_means = []
    for cluster in clusters:
        if cluster:  # Avoid division by zero
            new_cluster_means.append(np.mean(cluster, axis=0))
        else:  # Handle empty clusters by retaining the old mean
            new_cluster_means.append(cluster_means[len(new_cluster_means)])
    cluster_means = np.array(new_cluster_means)

# Print the final cluster means
print("Final Cluster Means:")
for i, mean in enumerate(cluster_means, start=1):
    print(f"Cluster {i} Mean: {mean}")
```
________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________


# 21

```
import pandas as pd
import numpy as np
import random

# Load the Iris dataset
df = pd.read_csv("/content/IRIS.csv")

# Extract only the numeric features for clustering
data = df.iloc[:, :-1].values  # Exclude the species column

# Set K and initialize cluster means randomly
K = 4
np.random.seed(42)  # For reproducibility
initial_means_indices = random.sample(range(len(data)), K)
cluster_means = data[initial_means_indices]

# Function to compute Euclidean distance
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

# K-means algorithm
for iteration in range(10):
    # Step 1: Assign each point to the nearest cluster
    clusters = [[] for _ in range(K)]
    for point in data:
        distances = [euclidean_distance(point, mean) for mean in cluster_means]
        cluster_index = np.argmin(distances)
        clusters[cluster_index].append(point)

    # Step 2: Update cluster means
    new_cluster_means = []
    for cluster in clusters:
        if cluster:  # Avoid division by zero
            new_cluster_means.append(np.mean(cluster, axis=0))
        else:  # Handle empty clusters by retaining the old mean
            new_cluster_means.append(cluster_means[len(new_cluster_means)])
    cluster_means = np.array(new_cluster_means)

# Print the final cluster means
print("Final Cluster Means:")
for i, mean in enumerate(cluster_means, start=1):
    print(f"Cluster {i} Mean: {mean}")
```
________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

# 22

```
# Given confusion matrix values
TP = 90    # True Positive
TN = 9560  # True Negative
FP = 140   # False Positive
FN = 210   # False Negative

# Calculate Accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN)

# Calculate Error Rate
error_rate = 1 - accuracy

# Calculate Precision
precision = TP / (TP + FP)

# Calculate Recall
recall = TP / (TP + FN)

# Print results
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Error Rate: {error_rate * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
```
________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

# 23

```
import pandas as pd
import numpy as np

# Step 1: Construct the dataset from the provided table
data = {
    'Age': ['Young', 'Young', 'Middle', 'Old', 'Old', 'Old', 'Middle', 'Young', 'Young', 'Old', 'Young', 'Middle', 'Middle', 'Old'],
    'Income': ['High', 'High', 'High', 'Medium', 'Low', 'Low', 'Low', 'Medium', 'Low', 'Medium', 'Medium', 'Medium', 'High', 'Medium'],
    'Married': ['No', 'No', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes'],
    'Health': ['Fair', 'Good', 'Fair', 'Fair', 'Fair', 'Good', 'Good', 'Fair', 'Fair', 'Fair', 'Good', 'Good', 'Fair', 'Good'],
    'Class': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Step 2: Calculate the Frequency Table for 'Age'
age_freq_table = df['Age'].value_counts()
print("Frequency table for Age:")
print(age_freq_table)

# Step 3: Calculate the Entropy Before Split (Entropy of the target variable 'Class')
def entropy(values):
    total = len(values)
    value_counts = values.value_counts()
    prob = value_counts / total
    return -np.sum(prob * np.log2(prob))

# Calculate entropy of the target variable 'Class'
entropy_before_split = entropy(df['Class'])
print(f"Entropy before the split: {entropy_before_split:.4f}")

# Step 4: Calculate the Entropy After Split on 'Age' (Weighted Entropy)
def weighted_entropy(data, attribute, target):
    total = len(data)
    weighted_entropy = 0.0
    for value in data[attribute].unique():
        subset = data[data[attribute] == value]
        subset_entropy = entropy(subset[target])
        weighted_entropy += (len(subset) / total) * subset_entropy
    return weighted_entropy

# Calculate the weighted entropy after the split based on 'Age'
entropy_after_split = weighted_entropy(df, 'Age', 'Class')
print(f"Entropy after the split on 'Age': {entropy_after_split:.4f}")

# Step 5: Calculate Information Gain
information_gain = entropy_before_split - entropy_after_split
print(f"Information Gain when splitting on 'Age': {information_gain:.4f}")
```
________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________


# 24

```
import pandas as pd

# Load the IRIS dataset (replace 'IRIS.csv' with the correct path if needed)
df = pd.read_csv('IRIS.csv')

# 1. Counting unique values in each column
print("Unique values in each column:")
print(df.nunique())
print("\n")

# 2. Format (data types) of each column
print("Data types of each column:")
print(df.dtypes)
print("\n")

# 3. Converting variable data types (e.g., from float to int, or vice versa)
# Example: Convert 'sepal_length' (or the actual column name) from float to integer (if appropriate)
# Note: We only convert if it's safe to do so and doesn't affect the accuracy of the data
# Replace 'sepal_length', 'sepal_width', etc. with the actual column names
df['sepal_length'] = df['sepal_length'].astype('float')  # Ensuring that it's float for precision
df['sepal_width'] = df['sepal_width'].astype('float')
df['petal_length'] = df['petal_length'].astype('float')
df['petal_width'] = df['petal_width'].astype('float')

print("Data types after conversion (if any changes):")
print(df.dtypes)
print("\n")

# 4. Identifying missing values in each column
print("Missing values in each column:")
print(df.isnull().sum())
print("\n")

# 5. Filling missing values
# If there are any missing values, we'll fill them with the column mean (for numeric columns)
# For categorical columns, we can fill with the mode (most frequent value)
# Replace 'sepal_length', 'sepal_width', etc. with the actual column names
df['sepal_length'] = df['sepal_length'].fillna(df['sepal_length'].mean())
df['sepal_width'] = df['sepal_width'].fillna(df['sepal_width'].mean())
df['petal_length'] = df['petal_length'].fillna(df['petal_length'].mean())
df['petal_width'] = df['petal_width'].fillna(df['petal_width'].mean())

# If there were categorical columns (e.g., species), we could fill with the mode
# Since 'Species' is categorical, let's fill it with the most frequent value (mode)
df['species'] = df['species'].fillna(df['species'].mode()[0])  # Replace 'species' with the actual column name

# Checking the data after filling missing values
print("Data after filling missing values:")
print(df.head())
```
________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

# 25

```
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the IRIS dataset
df = pd.read_csv('IRIS.csv')

# 1. Data Cleaning

# a. Remove duplicate rows if any
df.drop_duplicates(inplace=True)

# b. Check for missing values
missing_values = df.isnull().sum()

# c. Fill missing values (if any) with the mean of the column (for numeric columns)
# Here, we'll assume the data does not have missing values, but if it did:
# Corrected column names to match the CSV file (likely 'sepal_length', etc.)
df['sepal_length'].fillna(df['sepal_length'].mean(), inplace=True)
df['sepal_width'].fillna(df['sepal_width'].mean(), inplace=True)
df['petal_length'].fillna(df['petal_length'].mean(), inplace=True)
df['petal_width'].fillna(df['petal_width'].mean(), inplace=True)

# d. Remove irrelevant columns (if any)
# For IRIS dataset, we don't have irrelevant columns, but you could drop any column like:
# df.drop(columns=['SomeColumn'], inplace=True)

# e. Correct data types (if necessary)
# Let's ensure that numerical columns are in the correct format (floats)
df['sepal_length'] = df['sepal_length'].astype('float')
df['sepal_width'] = df['sepal_width'].astype('float')
df['petal_length'] = df['petal_length'].astype('float')
df['petal_width'] = df['petal_width'].astype('float')

# Displaying the cleaned data
print("Cleaned DataFrame:")
print(df.head())
print("\n")

# 2. Data Transformation

# a. Scaling/Normalization of numeric columns using StandardScaler (Z-Score Normalization)
scaler = StandardScaler()

# Selecting numeric columns for scaling
# Corrected column names
numeric_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

print("Data after Scaling/Normalization:")
print(df.head())
print("\n")

# b. Label Encoding for the 'Species' column (categorical data)
# 'Species' column is categorical, so we will use label encoding
label_encoder = LabelEncoder()
df['species'] = label_encoder.fit_transform(df['species'])  # Corrected column name

print("Data after Label Encoding for 'Species' column:")
print(df.head())
print("\n")

# c. Feature Engineering: Creating a new feature 'Sepal Area' (for example)
df['SepalArea'] = df['sepal_length'] * df['sepal_width']  # Corrected column names

print("Data with New Feature 'SepalArea':")
print(df.head())
print("\n")

# 3. Split the dataset into training and testing sets (if needed)
X = df.drop(columns=['species'])  # Features - Corrected column name
y = df['species']  # Target - Corrected column name
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Train and Test Split:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
```
