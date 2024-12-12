import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load dataset
file_path = "bank-full.csv"
data = pd.read_csv(file_path, sep=';')

# Step 1: Inspect and Clean Data
print("Dataset Info:")
data.info()

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Check for duplicate rows
duplicates = data.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

# Step 2: Analyze and Handle Categorical Variables
# Explicitly identify categorical columns
categorical_columns = [col for col in data.columns if data[col].dtype == 'object']
print(f"\nCategorical Columns: {categorical_columns}")

# Display unique values in categorical columns before encoding
for col in categorical_columns:
    print(f"\nUnique values in '{col}': {data[col].unique()}")

# Encode categorical variables
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Store encoders for reference

# Step 3: Analyze Numerical Variables
# Explicitly identify numerical columns
numerical_columns = [col for col in data.columns if col not in categorical_columns and col != 'y']
print(f"\nNumerical Columns: {numerical_columns}")

# Check basic statistics of numerical columns
print("\nNumerical Column Statistics:")
print(data[numerical_columns].describe())

# Step 4: Address Outliers (Optional: Based on domain knowledge)
# Example: Clipping balance values to remove extreme outliers (domain-specific threshold can be set)
data['balance'] = data['balance'].clip(lower=data['balance'].quantile(0.01), upper=data['balance'].quantile(0.99))

# Step 5: Final Dataset Summary
print("\nCleaned Dataset Info:")
data.info()
print("\nPreview of Cleaned Data:")
print(data.head())

# Save cleaned data for modeling
cleaned_file_path = "bank-full-cleaned.csv"
data.to_csv(cleaned_file_path, index=False)
print(f"\nCleaned data saved to {cleaned_file_path}")

# Note: Encoded values can be decoded using 'label_encoders' dictionary for interpretation.
