import csv
import random

# Load dataset
with open("data/raw/creditcard.csv", "r") as file:
    reader = csv.DictReader(file)
    data = list(reader)

print("--- Dataset Loaded ---")
print("Total transactions:", len(data))

# Shuffle data
random.seed(42)
random.shuffle(data)

# 80% training, 20% testing
split_index = int(0.8 * len(data))

train_data = data[:split_index]
test_data = data[split_index:]

# Count classes
train_fraud = sum(row["Class"] == "1" for row in train_data)
train_legitimate = sum(row["Class"] == "0" for row in train_data)

test_fraud = sum(row["Class"] == "1" for row in test_data)
test_legitimate = sum(row["Class"] == "0" for row in test_data)

print("\n--- Training Data ---")
print("Training transactions:", len(train_data))
print("Legitimate:", train_legitimate)
print("Fraud:", train_fraud)

print("\n--- Testing Data ---")
print("Testing transactions:", len(test_data))
print("Legitimate:", test_legitimate)
print("Fraud:", test_fraud)

print("\n--- Preprocessing Complete ---")