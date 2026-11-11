import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# 1. CONFIGURACIÓN Y CONSTANTES
SEED = 42
DATA_PATH = "data/processed/vgsales_cleaned.csv"
MODEL_PATH = "src/model.pkl"
# Umbral conservador (0.65) para minimizar Falsos Positivos y proteger la inversión del estudio
CONSERVATIVE_THRESHOLD = 0.65


# 2. CARGA DE DATOS
def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['is_hit', 'Name'])
    y = df['is_hit']
    return X, y


# 3. PIPELINE DE PREPROCESADO
def get_preprocessor():
    categorical_cols = ['Platform', 'Genre', 'Publisher_Grouped', 'Platform_Vendor']
    numerical_cols = ['Year']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    return preprocessor


# 4. EVALUACIÓN Y LOGGING EN MLFLOW
def evaluate_and_log_model(model_name, pipeline, X_train, y_train, X_test, y_test, threshold=0.65):
    with mlflow.start_run(run_name=model_name):
        # Ajuste exclusivo sobre train
        pipeline.fit(X_train, y_train)
        
        # Obtención de probabilidades de inferencia
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_proba = pipeline.predict(X_test).astype(float)

        # Aplicación del umbral conservador personalizado
        y_pred = (y_proba >= threshold).astype(int)

        # Métricas enfocadas en minimizar Falsos Positivos
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        f05 = fbeta_score(y_test, y_pred, beta=0.5, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            roc_auc = 0.5

        print(f"\nResultados para {model_name} (Umbral Conservador = {threshold}):")
        print(f"Precisión (Seguridad de Hit): {prec:.4f}")
        print(f"F0.5-Score (Prioriza Precisión): {f05:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Accuracy: {acc:.4f}")

        # Registro de hiperparámetros y métricas en MLflow
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("decision_threshold", threshold)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("f05_score", f05)
        mlflow.log_metric("roc_auc", roc_auc)

        # Generación y guardado de Matriz de Confusión
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Hit', 'Hit'])
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, cmap='Blues')
        plt.title(f'Matriz de Confusión - {model_name} (Umbral {threshold})')
        
        os.makedirs("artifacts", exist_ok=True)
        cm_path = f"artifacts/cm_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(cm_path)
        plt.close()
        
        mlflow.log_artifact(cm_path)

        # Registro del modelo con tipos seguros autorizados
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree",
                "collections.OrderedDict",
                "lightgbm.basic.Booster",
                "lightgbm.sklearn.LGBMClassifier"
            ]
        )

        return prec, f05, roc_auc, pipeline


# 5. ORQUESTACIÓN Y COMPARACIÓN DE MODELOS
def run_training():
    mlflow.set_experiment("VideoGame_Sales_Hit_Prediction")

    X, y = load_data()
    
    # Split estratificado 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    preprocessor = get_preprocessor()

    models = {
        "Baseline Dummy": DummyClassifier(strategy="most_frequent"),
        "Regresion Logistica": LogisticRegression(class_weight="balanced", random_state=SEED, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=SEED),
        "LightGBM": LGBMClassifier(n_estimators=100, class_weight="balanced", random_state=SEED, verbose=-1)
    }

    best_score = -1.0
    best_pipeline = None
    best_model_name = ""

    for name, clf in models.items():
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        prec, f05, roc_auc, pipeline = evaluate_and_log_model(
            name, full_pipeline, X_train, y_train, X_test, y_test, threshold=CONSERVATIVE_THRESHOLD
        )

        # Selección basada en F0.5-Score (prioriza la reducción de Falsos Positivos)
        if f05 > best_score:
            best_score = f05
            best_pipeline = pipeline
            best_model_name = name

    print(f"\nMEJOR MODELO SELECCIONADO PARA PRODUCCIÓN: {best_model_name} (F0.5-Score: {best_score:.4f})")
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"[OK] Pipeline empaquetado guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    run_training()