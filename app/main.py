from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import sys
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

# Get the CyberGuard project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Path to src folder
SRC_DIR = BASE_DIR / "src"

# Add src folder to Python path
sys.path.append(str(SRC_DIR))


# ============================================================
# IMPORT PREDICTION FUNCTION
# ============================================================

from predict import predict_attack


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="CyberGuard API",
    description=(
        "ML-based IoT Network Attack Detection API. "
        "Accepts network traffic features and predicts "
        "the corresponding attack type."
    ),
    version="1.0.0"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class NetworkTraffic(BaseModel):
    data: dict = Field(
        ...,
        description=(
            "Network traffic features required by the "
            "CyberGuard Random Forest model. "
            "The input should contain all required "
            "network traffic features."
        )
    )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    """
    Check whether the CyberGuard API is running.
    """

    return {
        "message": "CyberGuard API is running",
        "version": "1.0.0"
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(data: NetworkTraffic):
    """
    Predict the attack type of network traffic.
    """

    try:

        # Send input data to ML prediction function
        result = predict_attack(data.data)

    except ValueError as e:

        # Return a proper API error
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        # Handle unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    # Return prediction result
    return {
        "predicted_attack_type": result["attack_type"],
        "confidence": result["confidence"],
        "message": "Network traffic classified successfully"
    }