## Modelo y Experimentación

### 1. Justificación de Negocio: Matriz de Costes Asimétrica
En la industria del videojuego independiente, el riesgo financiero no es simétrico:
* **Falso Positivo (Riesgo Crítico):** Predecir un *Hit* en un proyecto que acaba siendo un fracaso comercial conlleva la inversión de capital limitado y la quiebra del estudio.
* **Falso Negativo (Riesgo Aceptable):** Descartar un proyecto que podría haber sido un éxito representa un coste de oportunidad, pero preserva la liquidez de la empresa.

Por tanto, priorizamos la **Precisión** y la **Calibración del Umbral de Decisión** por encima del Recall puro.

### 2. Tarea y Métricas Elegidas
* **Tarea:** Clasificación Binaria Supervisada (Global_Sales >= 0.5M$).
* **Métricas Principales:**
  * **ROC-AUC & PR-AUC:** Para evaluar la separación entre clases independientemente del umbral.
  * **Precision @ Umbral Conservador ($\ge 0.65$):** Métrica operativa para la toma de decisiones financieras.
  * **$F_{0.5}$-Score:** Ponderación que prioriza la Precisión sobre la Sensibilidad.

### 3. Split sin Fuga de Información y Baseline
* Split 80% Train / 20% Test estratificado (`random_state=42`). Preprocesado encapsulado en un `Pipeline` único para evitar fugas.
* **Baseline Dummy (`most_frequent`):** Precision: 0.0000 | ROC-AUC: 0.5000.

### 4. Tuning de Umbral y Selección de Modelo
Tras evaluar Regresión Logística, Random Forest y LightGBM en MLflow, **LightGBM** demostró la mejor calibración de probabilidades. 

Al elevación del umbral de inferencia a **0.65**, el modelo incrementa la **Precisión del 57% al 74%**, asegurando que la mini-app solo devuelva una recomendación positiva cuando exista alta certeza comercial.