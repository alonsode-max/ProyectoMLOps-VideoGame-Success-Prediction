import os
import joblib
import pandas as pd

MODEL_PATH = "src/model.pkl"


def test_model_artifact_exists():
    """Verifica que el artefacto del pipeline entrenado existe en el repositorio."""
    assert os.path.exists(MODEL_PATH), f"No se encontró el modelo en {MODEL_PATH}"


def test_model_inference_and_unknown_categories():
    """
    Comprueba que el pipeline predice probabilidades válidas [0, 1]
    y tolera categorías nuevas sin fallar (handle_unknown='ignore').
    """
    pipeline = joblib.load(MODEL_PATH)

    sample = pd.DataFrame([
        {
            "Platform": "PS4",
            "Year": 2018,
            "Genre": "Action",
            "Publisher": "Ubisoft",
            "Publisher_Grouped": "Ubisoft",
            "Platform_Vendor": "Sony",
        },
        {
            "Platform": "Consola_Ficticia_2026",
            "Year": 2026,
            "Genre": "Indie_Experimental",
            "Publisher": "Nuevo_Estudio_Indie",
            "Publisher_Grouped": "Other_Publisher",
            "Platform_Vendor": "Other_Vendor",
        },
    ])

    probas = pipeline.predict_proba(sample)[:, 1]
    assert len(probas) == 2
    assert all(0.0 <= p <= 1.0 for p in probas)