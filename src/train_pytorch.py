import csv
import random

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# 1. LOAD DATASET
# ============================================================

with open("data/raw/creditcard.csv", "r") as file:
    reader = csv.DictReader(file)
    data = list(reader)


# Shuffle dataset
random.seed(42)
random.shuffle(data)


# ============================================================
# 2. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# 60% training
# 20% validation
# 20% testing

train_end = int(0.60 * len(data))
validation_end = int(0.80 * len(data))

train_data = data[:train_end]
validation_data = data[train_end:validation_end]
test_data = data[validation_end:]


print("--- Dataset Split ---")
print("Training:", len(train_data))
print("Validation:", len(validation_data))
print("Testing:", len(test_data))


# ============================================================
# 3. FEATURE COLUMNS
# ============================================================

feature_columns = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8",
    "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16",
    "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24",
    "V25", "V26", "V27", "V28",
    "Amount"
]


# ============================================================
# 4. CONVERT DATA INTO NUMBERS
# ============================================================

X_train = [
    [float(row[column]) for column in feature_columns]
    for row in train_data
]

y_train = [
    int(row["Class"])
    for row in train_data
]


X_validation = [
    [float(row[column]) for column in feature_columns]
    for row in validation_data
]

y_validation = [
    int(row["Class"])
    for row in validation_data
]


X_test = [
    [float(row[column]) for column in feature_columns]
    for row in test_data
]

y_test = [
    int(row["Class"])
    for row in test_data
]


# ============================================================
# 5. SCALE FEATURES
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_validation = scaler.transform(X_validation)

X_test = scaler.transform(X_test)


# ============================================================
# 6. CONVERT TO PYTORCH TENSORS
# ============================================================

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_validation = torch.tensor(
    X_validation,
    dtype=torch.float32
)

y_validation = torch.tensor(
    y_validation,
    dtype=torch.float32
)

X_test = torch.tensor(X_test, dtype=torch.float32)

y_test = torch.tensor(
    y_test,
    dtype=torch.float32
)


# ============================================================
# 7. NEURAL NETWORK
# ============================================================

class FraudNeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(30, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


model = FraudNeuralNetwork()


# ============================================================
# 8. CLASS IMBALANCE HANDLING
# ============================================================

fraud_count = sum(y_train == 1)
legitimate_count = sum(y_train == 0)

print("\n--- Training Class Distribution ---")
print("Legitimate:", legitimate_count)
print("Fraud:", fraud_count)


# Moderate class weighting
# This prevents the model from predicting everything as fraud.

pos_weight = torch.tensor(
    [50.0],
    dtype=torch.float32
)


loss_function = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)


# ============================================================
# 9. OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# 10. TRAIN MODEL
# ============================================================

epochs = 20

print("\n--- Training PyTorch Neural Network ---")

for epoch in range(epochs):

    model.train()

    optimizer.zero_grad()

    outputs = model(X_train).squeeze()

    loss = loss_function(
        outputs,
        y_train
    )

    loss.backward()

    optimizer.step()

    print(
        f"Epoch {epoch + 1}/{epochs} - "
        f"Loss: {loss.item():.4f}"
    )


print("Training complete!")


# ============================================================
# 11. VALIDATION PREDICTIONS
# ============================================================

model.eval()

with torch.no_grad():

    validation_outputs = model(
        X_validation
    ).squeeze()

    validation_probabilities = torch.sigmoid(
        validation_outputs
    )


# ============================================================
# 12. FIND BEST THRESHOLD USING VALIDATION DATA
# ============================================================

thresholds = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]


best_threshold = 0.50
best_f1 = 0.0

print("\n--- Validation Threshold Comparison ---")

for threshold in thresholds:

    validation_predictions = (
        validation_probabilities >= threshold
    ).int()

    report = classification_report(
        y_validation.int().tolist(),
        validation_predictions.tolist(),
        output_dict=True,
        zero_division=0
    )

    fraud_f1 = report["1"]["f1-score"]
    fraud_precision = report["1"]["precision"]
    fraud_recall = report["1"]["recall"]

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {fraud_precision:.2f} | "
        f"Recall: {fraud_recall:.2f} | "
        f"F1: {fraud_f1:.2f}"
    )

    if fraud_f1 > best_f1:

        best_f1 = fraud_f1
        best_threshold = threshold


print("\nBest validation threshold:", best_threshold)
print("Best validation fraud F1:", round(best_f1, 2))


# ============================================================
# 13. FINAL TEST EVALUATION
# ============================================================

with torch.no_grad():

    test_outputs = model(
        X_test
    ).squeeze()

    test_probabilities = torch.sigmoid(
        test_outputs
    )


test_predictions = (
    test_probabilities >= best_threshold
).int()


# ============================================================
# 14. FINAL RESULTS
# ============================================================

print("\n--- FINAL TEST RESULTS ---")

print(
    classification_report(
        y_test.int().tolist(),
        test_predictions.tolist(),
        digits=2,
        zero_division=0
    )
)


print("--- Final Confusion Matrix ---")

print(
    confusion_matrix(
        y_test.int().tolist(),
        test_predictions.tolist()
    )
)