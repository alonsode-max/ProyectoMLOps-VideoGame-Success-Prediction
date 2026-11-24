# Imagen base ligera de Python
FROM python:3.11-slim

# Evitar escritura de bytecode y habilitar buffer continuo de logs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

# Instalar dependencias de sistema necesarias para LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente, artefactos y módulos del proyecto
COPY src/ ./src/

# Hardening de Seguridad: Crear y cambiar a un usuario no root
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Exponer el puerto del servicio
EXPOSE 8000

# Comando de arranque del servidor ASGI Uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]