import os
from contextlib import asynccontextmanager
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from fastapi.responses import RedirectResponse

# 1. CONSTANTES Y CONFIGURACIÓN
MODEL_PATH = os.getenv("MODEL_PATH", "src/model.pkl")
CONSERVATIVE_THRESHOLD = 0.65

NINTENDO = {'Wii', 'NES', 'GB', 'DS', 'GBA', '3DS', 'N64', 'GC', 'WiiU', 'SNES'}
SONY = {'PS2', 'PS3', 'PS', 'PS4', 'PSP', 'PSV'}
MICROSOFT = {'X360', 'XB', 'XOne'}

TOP_PUBLISHERS = {
    'Electronic Arts', 'Activision', 'Namco Bandai Games', 'Ubisoft', 'Konami Digital Entertainment',
    'THQ', 'Nintendo', 'Sony Computer Entertainment', 'Sega', 'Take-Two Interactive',
    'Capcom', 'Atari', 'Tecmo Koei', 'Square Enix', 'Warner Bros. Interactive Entertainment',
    'Disney Interactive Studios', 'Eidos Interactive', 'Midway Games', '505 Games', 'Microsoft Game Studios'
}

# 2. CARGA DEL ARTEFACTO (MODELO EMPAQUETADO)
pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"No se encontró el modelo en la ruta: {MODEL_PATH}")
        pipeline = joblib.load(MODEL_PATH)
    return pipeline


# 3. CICLO DE VIDA DE LA APLICACIÓN (LIFESPAN)
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_pipeline()
        print(f"[OK] Modelo cargado correctamente desde {MODEL_PATH}")
    except Exception as e:
        print(f"[ADVERTENCIA] No se pudo precargar el modelo al arrancar: {e}")
    yield


# 4. SCHEMAS DE ENTRADA Y SALIDA (PYDANTIC V2)
class GamePredictionInput(BaseModel):
    Platform: str = Field(..., examples=["X360"], description="Consola o plataforma del videojuego")
    Year: int = Field(..., examples=[2015], description="Año de lanzamiento estimado (1980 - 2026)")
    Genre: str = Field(..., examples=["Action"], description="Género principal del juego")
    Publisher: str = Field(..., examples=["Ubisoft"], description="Distribuidora u organizador del lanzamiento")

    @field_validator('Year')
    @classmethod
    def validate_year(cls, v: int) -> int:
        if v < 1980 or v > 2026:
            raise ValueError('El año de lanzamiento debe estar comprendido entre 1980 y 2026.')
        return v

    @field_validator('Platform', 'Genre', 'Publisher')
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('Los campos de texto no pueden estar vacíos.')
        return v.strip()


class PredictionOutput(BaseModel):
    is_hit: int = Field(..., description="Predicción binaria: 1 si supera 0.5M ventas, 0 si no")
    probability: float = Field(..., description="Probabilidad estimada de éxito comercial")
    threshold: float = Field(..., description="Umbral de decisión aplicado")
    recommendation: str = Field(..., description="Interpretación de negocio orientada a reducir riesgo")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    threshold: float


# 5. INICIALIZACIÓN DE FASTAPI
app = FastAPI(
    title="Video Game Hit Predictor API",
    description="API MLOps para predecir si un videojuego superará el medio millón de ventas globales.",
    version="1.0.0",
    lifespan=lifespan
)


# 6. ENDPOINTS
@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK, tags=["Monitorización"])
def health_check():
    """Endpoint de verificación de estado e integridad del servicio."""
    model_ok = False
    try:
        get_pipeline()
        model_ok = True
    except Exception:
        model_ok = False

    return {
        "status": "healthy" if model_ok else "unhealthy",
        "model_loaded": model_ok,
        "threshold": CONSERVATIVE_THRESHOLD
    }


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.post("/predict", response_model=PredictionOutput, status_code=status.HTTP_200_OK, tags=["Inferencia"])
def predict_hit(game: GamePredictionInput):
    """
    Recibe las características básicas de un juego y devuelve la probabilidad
    y recomendación de inversión aplicando el umbral conservador.
    """
    model = get_pipeline()

    platform = game.Platform
    publisher = game.Publisher

    if platform in NINTENDO:
        vendor = 'Nintendo'
    elif platform in SONY:
        vendor = 'Sony'
    elif platform in MICROSOFT:
        vendor = 'Microsoft'
    elif platform == 'PC':
        vendor = 'PC'
    else:
        vendor = 'Other_Vendor'

    publisher_grouped = publisher if publisher in TOP_PUBLISHERS else 'Other_Publisher'

    input_df = pd.DataFrame([{
        'Platform': platform,
        'Year': game.Year,
        'Genre': game.Genre,
        'Publisher': publisher,
        'Publisher_Grouped': publisher_grouped,
        'Platform_Vendor': vendor
    }])

    try:
        proba = float(model.predict_proba(input_df)[0, 1])
        is_hit = int(proba >= CONSERVATIVE_THRESHOLD)

        if is_hit == 1:
            recommendation = f"RECOMENDADO: Alta probabilidad de éxito comercial ({proba:.1%}). Riesgo de inversión controlado."
        else:
            recommendation = f"NO RECOMENDADO: Probabilidad de hit insuficiente ({proba:.1%}). Riesgo de pérdidas elevado."

        return PredictionOutput(
            is_hit=is_hit,
            probability=round(proba, 4),
            threshold=CONSERVATIVE_THRESHOLD,
            recommendation=recommendation
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al procesar la predicción: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)