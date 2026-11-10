import os
import pandas as pd
import numpy as np
import pandera.pandas as pa
from pandera.errors import SchemaErrors

CleanedDataSchema = pa.DataFrameSchema({
    "Name": pa.Column(str, nullable=False),
    "Platform": pa.Column(str, nullable=False),
    "Year": pa.Column(int, pa.Check.in_range(1980, 2017), nullable=False),
    "Genre": pa.Column(str, nullable=False),
    "Publisher": pa.Column(str, nullable=True),
    "Publisher_Grouped": pa.Column(str, nullable=False),
    "Platform_Vendor": pa.Column(
        str, 
        pa.Check.isin(['Nintendo', 'Sony', 'Microsoft', 'PC', 'Other_Vendor']),
        nullable=False
    ),
    "is_hit": pa.Column(int, pa.Check.isin([0, 1]), nullable=False)
})


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica las 5 decisiones de preparación sobre el DataFrame crudo.
    """
    df = df.copy()

    # Decisión 1: Crear Target Binario (is_hit = Global_Sales >= 0.5M)
    df['is_hit'] = (df['Global_Sales'] >= 0.5).astype(int)

    # Decisión 2: Eliminar nulos en Year y castear a int
    df = df.dropna(subset=['Year']).copy()
    df['Year'] = df['Year'].astype(int)

    # Decisión 3: Agrupar editores minoritarios (Publisher_Grouped)
    top_publishers = df['Publisher'].value_counts().nlargest(20).index
    df['Publisher_Grouped'] = df['Publisher'].apply(
        lambda x: x if x in top_publishers else 'Other_Publisher'
    )

    # Decisión 4: Ingeniería de Features (Platform_Vendor)
    nintendo = ['Wii', 'NES', 'GB', 'DS', 'GBA', '3DS', 'N64', 'GC', 'WiiU', 'SNES']
    sony = ['PS2', 'PS3', 'PS', 'PS4', 'PSP', 'PSV']
    microsoft = ['X360', 'XB', 'XOne']

    def assign_vendor(platform: str) -> str:
        if platform in nintendo: return 'Nintendo'
        elif platform in sony: return 'Sony'
        elif platform in microsoft: return 'Microsoft'
        elif platform == 'PC': return 'PC'
        return 'Other_Vendor'

    df['Platform_Vendor'] = df['Platform'].apply(assign_vendor)

    # Decisión 5: Filtrado de anomalías temporales (> 2017)
    df = df[df['Year'] <= 2017].copy()

    # Descartar columnas con fuga de información (Data Leakage)
    drop_cols = ['Rank', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
    return df.drop(columns=drop_cols)


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida que el DataFrame cumpla con el esquema definido en Pandera.
    """
    try:
        validated_df = CleanedDataSchema.validate(df, lazy=True)
        print(" [OK] Validación de Esquema con Pandera completada exitosamente.")
        return validated_df
    except SchemaErrors as err:
        print(" [ERROR CRÍTICO] Violación de calidad de datos:")
        print(err.failure_cases)
        raise err


def run_data_pipeline(input_path: str = "data/raw/vgsales.csv", output_path: str = "data/processed/vgsales_cleaned.csv") -> pd.DataFrame:
    """
    Orquesta el pipeline completo de carga, auditoría, limpieza, validación y guardado.
    """
    print("=" * 60)
    print("EJECUTANDO PIPELINE DE PREPARACIÓN DE DATOS")
    print("=" * 60)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"No se encontró el archivo de datos crudos en: {input_path}")

    # 1. Carga y Auditoría inicial
    df_raw = pd.read_csv(input_path)
    print(f"- Filas iniciales: {df_raw.shape[0]} | Columnas iniciales: {df_raw.shape[1]}")
    print(f"- Duplicados exactos: {df_raw.duplicated().sum()}")

    # 2. Limpieza
    df_clean = clean_raw_data(df_raw)

    # 3. Validación (Quality Gate)
    df_validated = validate_schema(df_clean)

    # 4. Exportación
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_validated.to_csv(output_path, index=False)
    print(f"- Artefacto guardado exitosamente en: {output_path}")
    print("=" * 60)

    return df_validated


if __name__ == "__main__":
    run_data_pipeline()