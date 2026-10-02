import joblib
import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

# Get the CyberGuard project root directory
BASE_DIR = Path(__file__).resolve().parent.parent


# Model files
MODEL_PATH = BASE_DIR / "models" / "random_forest.pkl"
ENCODER_PATH = BASE_DIR / "models" / "encoder.pkl"
CONFIG_PATH = BASE_DIR / "models" / "feature_config.pkl"


# ============================================================
# LOAD SAVED MODEL FILES
# ============================================================

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
feature_config = joblib.load(CONFIG_PATH)


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

numerical_features = feature_config["numerical_features"]
categorical_features = feature_config["categorical_features"]


# ============================================================
# PREPROCESS INPUT DATA
# ============================================================

def preprocess_input(data):
    """
    Prepare new network traffic data for the ML model.
    """

    # All features required by the trained model
    required_features = (
        numerical_features + categorical_features
    )

    # Check for missing features
    missing_features = [
        feature
        for feature in required_features
        if feature not in data
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    # Convert input dictionary into DataFrame
    input_df = pd.DataFrame([data])

    # ========================================================
    # ENCODE CATEGORICAL FEATURES
    # ========================================================

    encoded_data = encoder.transform(
        input_df[categorical_features]
    )

    encoded_columns = encoder.get_feature_names_out(
        categorical_features
    )

    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoded_columns
    )

    # ========================================================
    # COMBINE NUMERICAL + ENCODED CATEGORICAL FEATURES
    # ========================================================

    final_data = pd.concat(
        [
            input_df[numerical_features].reset_index(drop=True),
            encoded_df.reset_index(drop=True)
        ],
        axis=1
    )

    # ========================================================
    # KEEP SAME FEATURE ORDER AS TRAINING
    # ========================================================

    final_data = final_data.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    return final_data


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_attack(data):
    """
    Predict the attack type and confidence
    for new network traffic.
    """

    # Preprocess input
    processed_data = preprocess_input(data)

    # Make prediction
    prediction = model.predict(processed_data)

    # Get prediction probabilities
    probabilities = model.predict_proba(
        processed_data
    )

    # Get predicted class
    predicted_class = prediction[0]

    # Create class-probability dictionary
    class_probabilities = dict(
        zip(
            model.classes_,
            probabilities[0]
        )
    )

    # Get probability of predicted class
    confidence = class_probabilities[
        predicted_class
    ]

    # Return prediction result
    return {
        "attack_type": predicted_class,
        "confidence": round(
            float(confidence),
            4
        )
    }