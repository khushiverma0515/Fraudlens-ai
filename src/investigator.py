
import os
import csv
import pickle

from openai import OpenAI
from rag_pipeline import retrieve_context


# Connect to OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Feature order must match train_random_forest.py
feature_columns = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8",
    "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16",
    "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24",
    "V25", "V26", "V27", "V28",
    "Amount"
]


# Load the saved Random Forest model
with open("models/random_forest_model.pkl", "rb") as file:
    model = pickle.load(file)

print("Random Forest model loaded successfully!")


def investigate_transaction(amount, dataset_time, fraud_probability, rag_context):
    prompt = f"""
You are an AI fraud investigation assistant.

Analyze this transaction:

Transaction amount (dataset units): {amount:.2f}
Dataset time (seconds elapsed): {dataset_time}
Fraud probability from Random Forest: {fraud_probability:.2%}

Knowledge base information:
{rag_context}

Provide:
1. Risk level
2. Fraud probability
3. Main reasons for possible suspicion
4. Recommended investigation action

Important:
- Do not claim the transaction is definitely fraudulent.
- Treat the ML probability as a risk signal, not proof.
- Do not invent transaction details or explain anonymized
  PCA features as if their meanings are known.
- Give a concise professional report.
"""

    try:
        response = client.responses.create(
            model="gpt-5-mini",
            input=prompt
        )
        return response.output_text

    except Exception as error:
        print(f"\nGPT unavailable: {error}")

        risk_level = (
            "HIGH" if fraud_probability >= 0.80
            else "MEDIUM" if fraud_probability >= 0.50
            else "LOW"
        )

        return (
            f"Risk Level: {risk_level}\n"
            f"Fraud Probability: {fraud_probability:.2%}\n\n"
            "Investigation Notes:\n"
            f"- Transaction amount: {amount:.2f} dataset units\n"
            f"- Dataset time: {dataset_time} seconds elapsed\n"
            "- Probability calculated by the Random Forest model.\n\n"
            f"Relevant RAG Guidance:\n{rag_context[:2000]}\n\n"
            "Recommended action: Review the transaction and "
            "supporting account information. This score is not "
            "proof of fraud."
        )



# INTERACTIVE DEMO: Select a transaction from the dataset

with open("data/raw/creditcard.csv", "r", newline="") as file:
    reader = csv.DictReader(file)
    transactions = list(reader)

print("\n--- FRAUD DETECTION + AI INVESTIGATOR ---")
print(f"Dataset transactions available: {len(transactions)}")
print("Enter a row number from 1 to", len(transactions))
print("(Enter 0 to exit.)")

while True:
    try:
        choice = int(input("\nTransaction row number: "))

        if choice == 0:
            print("Investigator closed.")
            break

        if choice < 1 or choice > len(transactions):
            print(f"Please enter a number from 1 to {len(transactions)}.")
            continue

        transaction = transactions[choice - 1]

        features = [
            float(transaction[column])
            for column in feature_columns
        ]

        amount = float(transaction["Amount"])
        dataset_time = float(transaction["Time"])

        fraud_probability = float(
            model.predict_proba([features])[0][1]
        )

        print("\n--- MODEL PREDICTION ---")
        print(f"Selected row: {choice}")
        print(f"Transaction amount: {amount:.2f}")
        print(f"Dataset time: {dataset_time} seconds")
        print(f"Fraud probability: {fraud_probability:.2%}")

        rag_context = retrieve_context(
            "What transaction characteristics may indicate fraud?"
        )

        result = investigate_transaction(
            amount,
            dataset_time,
            fraud_probability,
            rag_context
        )

        print("\n--- AI INVESTIGATOR REPORT ---")
        print(result)

        actual_label = transaction["Class"]
        actual_name = "Fraud" if actual_label == "1" else "Legitimate"
        print(f"\nDataset reference label: {actual_name}")

    except ValueError:
        print("Please enter a valid whole number.")
