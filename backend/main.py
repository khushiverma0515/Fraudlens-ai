
from pathlib import Path
import io

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "random_forest_model.pkl"

app = FastAPI(title="FraudLens AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
MAX_FILE_SIZE = 5 * 1024 * 1024


@app.get("/")
def home():
    return {
        "status": "ok",
        "service": "FraudLens AI API",
        "model_loaded": True,
        "supported_uploads": [".csv", ".xlsx"],
        "max_file_size_mb": 5,
    }


@app.post("/api/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    filename = (file.filename or "").lower()
    extension = Path(filename).suffix

    if extension not in {".csv", ".xlsx"}:
        raise HTTPException(
            status_code=400,
            detail="For now, upload a CSV (.csv) or Excel (.xlsx) file.",
        )

    content = await file.read(MAX_FILE_SIZE + 1)

    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File exceeds the 5 MB upload limit.",
        )

    try:
        if extension == ".csv":
            data = pd.read_csv(io.BytesIO(content))
        else:
            data = pd.read_excel(io.BytesIO(content), engine="openpyxl")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read the file. Check that it is a valid CSV or XLSX file.",
        )

    if data.empty:
        raise HTTPException(status_code=400, detail="No transaction rows were found.")

    if len(data) > 10000:
        raise HTTPException(
            status_code=400,
            detail="Upload no more than 10,000 transactions at a time.",
        )

    missing = [column for column in FEATURES if column not in data.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                "The file is missing required model features: "
                + ", ".join(missing)
            ),
        )

    try:
        features = data[FEATURES].apply(pd.to_numeric, errors="raise")

        if features.isnull().any().any():
            raise ValueError("Missing values found.")

        if not features.map(lambda value: pd.notna(value) and abs(value) != float("inf")).all().all():
            raise ValueError("Non-finite values found.")

        probabilities = model.predict_proba(features)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not analyze transaction data: {exc}",
        )

    results = []

    for index, (_, row) in enumerate(data.iterrows()):
        probability = float(probabilities[index])
        prediction = int(predictions[index])

        if probability >= 0.8:
            risk = "HIGH"
        elif probability >= 0.5:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        result = {
            "transaction_number": index + 1,
            "amount": float(features.iloc[index]["Amount"]),
            "predicted_class": prediction,
            "prediction": "Fraudulent" if prediction == 1 else "Legitimate",
            "fraud_probability": round(probability * 100, 2),
            "risk_level": risk,
        }

        if "Class" in data.columns and pd.notna(data.iloc[index]["Class"]):
            result["actual_dataset_label"] = (
                "Fraudulent" if int(data.iloc[index]["Class"]) == 1
                else "Legitimate"
            )

        results.append(result)

    return {
        "filename": file.filename,
        "transactions_analyzed": len(results),
        "fraudulent_count": sum(r["predicted_class"] == 1 for r in results),
        "legitimate_count": sum(r["predicted_class"] == 0 for r in results),
        "results": results,
        "notice": (
            "Predictions are model estimates, not proof of fraud. "
            "Input features must use the same format and preprocessing "
            "as the model's training data."
        ),
    }
