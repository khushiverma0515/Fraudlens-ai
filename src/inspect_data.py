import pandas as pd

df = pd.read_csv("data/raw/creditcard.csv")

print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- Columns Names ---")
print(df.columns.tolist())

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Fraud vs Legitimate ---")
print(df['Class'].value_counts())
