import csv
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# Load dataset
with open("data/raw/creditcard.csv", "r") as file:
    reader = csv.DictReader(file)
    data = list(reader)


# Shuffle dataset
random.seed(42)
random.shuffle(data)


# Split data
split_index = int(0.8 * len(data))

train_data = data[:split_index]
test_data = data[split_index:]


# Feature columns
feature_columns = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8",
    "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16",
    "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24",
    "V25", "V26", "V27", "V28",
    "Amount"
]


# Convert data into numbers
X_train = [
    [float(row[column]) for column in feature_columns]
    for row in train_data
]

y_train = [
    int(row["Class"])
    for row in train_data
]

X_test = [
    [float(row[column]) for column in feature_columns]
    for row in test_data
]

y_test = [
    int(row["Class"])
    for row in test_data
]


# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# Train model
print("--- Training Random Forest Model ---")

model.fit(X_train, y_train)

print("Training complete!")


# Get fraud probabilities
fraud_probabilities = model.predict_proba(X_test)[:, 1]


# Test different thresholds
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

print("\n--- Random Forest Threshold Comparison ---")

for threshold in thresholds:

    y_pred = [
        1 if probability >= threshold else 0
        for probability in fraud_probabilities
    ]

    print(f"\nThreshold: {threshold}")

    print(classification_report(
        y_test,
        y_pred,
        digits=2
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))