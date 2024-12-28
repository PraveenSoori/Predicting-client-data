# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

# Step 1: Load the Dataset
file_path = '/content/drive/MyDrive/ML CW/bank-full.csv'
data = pd.read_csv(file_path, delimiter=';')

# Step 2: Handle Missing Values
print("Missing values per column:")
print(data.isnull().sum())

# Step 3: Check for duplicate rows
duplicates = data.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

# Step 4: Encode Categorical Variables
categorical_cols = data.select_dtypes(include=['object']).columns.drop('y')  # Exclude target variable

# For Random Forest: Label Encoding
label_encoded_data = data.copy()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    label_encoded_data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# For Neural Networks: One-Hot Encoding
one_hot_encoded_data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Encode the target variable
label_encoder_y = LabelEncoder()
label_encoded_data['y'] = label_encoder_y.fit_transform(data['y'])
one_hot_encoded_data['y'] = label_encoder_y.transform(data['y'])

# Step 4: Feature Scaling for Neural Networks
scaler = StandardScaler()
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
scaled_data = one_hot_encoded_data.copy()
scaled_data[numerical_cols] = scaler.fit_transform(one_hot_encoded_data[numerical_cols])

# Step 5: Train-Test Split
X_rf = label_encoded_data.drop('y', axis=1)  # Features for Random Forest
y_rf = label_encoded_data['y']

X_nn = scaled_data.drop('y', axis=1)  # Features for Neural Networks
y_nn = scaled_data['y']

# Split the data
X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(
    X_rf, y_rf, test_size=0.2, random_state=42, stratify=y_rf)
X_nn_train, X_nn_test, y_nn_train, y_nn_test = train_test_split(
    X_nn, y_nn, test_size=0.2, random_state=42, stratify=y_nn)

# Step 6: Address Class Imbalance with SMOTE
smote = SMOTE(random_state=42)

# Balancing the Random Forest dataset
X_rf_train_balanced, y_rf_train_balanced = smote.fit_resample(X_rf_train, y_rf_train)

# Balancing the Neural Network dataset
X_nn_train_balanced, y_nn_train_balanced = smote.fit_resample(X_nn_train, y_nn_train)

# Step 7: Display Results
print("Balanced Random Forest Training Set Class Distribution:")
print(pd.Series(y_rf_train_balanced).value_counts(normalize=True))

print("Balanced Neural Network Training Set Class Distribution:")
print(pd.Series(y_nn_train_balanced).value_counts(normalize=True))

print("Random Forest Train Shape:", X_rf_train_balanced.shape)
print("Random Forest Test Shape:", X_rf_test.shape)
print("Neural Network Train Shape:", X_nn_train_balanced.shape)
print("Neural Network Test Shape:", X_nn_test.shape)
