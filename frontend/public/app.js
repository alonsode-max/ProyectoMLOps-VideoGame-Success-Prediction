const API_BASE_URL = 'http://localhost:8000';

// Elementos del DOM
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const predictForm = document.getElementById('predict-form');
const submitBtn = document.getElementById('submit-btn');

const errorBox = document.getElementById('error-box');
const errorMessage = document.getElementById('error-message');

const resultCard = document.getElementById('result-card');
const verdictTitle = document.getElementById('verdict-title');
const verdictBadge = document.getElementById('verdict-badge');
const probValue = document.getElementById('prob-value');
const thresholdValue = document.getElementById('threshold-value');
const probBar = document.getElementById('prob-bar');
const recommendationBox = document.getElementById('recommendation-box');
const jsonInspector = document.getElementById('json-inspector');

// 1. Verificar salud de la API (/health)
async function checkApiHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (!response.ok) throw new Error('API respondió con error');

        const data = await response.json();
        if (data.status === 'healthy' && data.model_loaded) {
            statusDot.className = 'w-2.5 h-2.5 rounded-full bg-emerald-400';
            statusText.textContent = `API Activa · Umbral ${(data.threshold * 100).toFixed(0)}%`;
        } else {
            statusDot.className = 'w-2.5 h-2.5 rounded-full bg-amber-400';
            statusText.textContent = 'API Online · Modelo no cargado';
        }
    } catch (err) {
        statusDot.className = 'w-2.5 h-2.5 rounded-full bg-rose-500';
        statusText.textContent = 'API Desconectada (Puerto 8000)';
    }
}

// 2. Enviar predicción (/predict)
predictForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Ocultar alertas previas
    errorBox.classList.add('hidden');
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span>Analizando viabilidad...</span>';

    const platform = document.getElementById('platform').value;
    const genre = document.getElementById('genre').value;
    const year = parseInt(document.getElementById('year').value, 10);
    const customPub = document.getElementById('publisher-custom').value.trim();
    const selectedPub = document.getElementById('publisher-select').value;

    const payload = {
        Platform: platform,
        Year: year,
        Genre: genre,
        Publisher: customPub.length > 0 ? customPub : selectedPub
    };

    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!response.ok) {
            const detail = data.detail ? JSON.stringify(data.detail) : 'Error de validación en los datos.';
            throw new Error(`HTTP ${response.status}: ${detail}`);
        }

        renderPrediction(payload, data);
    } catch (err) {
        resultCard.classList.add('hidden');
        errorBox.classList.remove('hidden');

        if (err.message.includes('Failed to fetch')) {
            errorMessage.textContent = `No se pudo conectar con la API en ${API_BASE_URL}. Verifica que el contenedor Docker o Uvicorn esté encendido.`;
        } else {
            errorMessage.textContent = err.message;
        }
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<span>Evaluar Viabilidad del Proyecto</span>';
    }
});

// 3. Renderizar resultados en la UI
function renderPrediction(requestPayload, result) {
    const probPct = (result.probability * 100).toFixed(1);
    const threshPct = (result.threshold * 100).toFixed(0);
    const isViable = result.is_hit === 1;

    probValue.textContent = `${probPct}%`;
    thresholdValue.textContent = `${threshPct}%`;
    probBar.style.width = `${Math.min(Math.max(result.probability * 100, 0), 100)}%`;

    if (isViable) {
        verdictTitle.textContent = 'Proyecto Viable';
        verdictTitle.className = 'text-2xl font-bold mt-0.5 text-emerald-400';
        verdictBadge.textContent = 'INVERSIÓN RECOMENDADA';
        verdictBadge.className = 'px-4 py-1.5 rounded-full text-xs font-bold bg-emerald-950 text-emerald-300 border border-emerald-700';
        probBar.className = 'h-full transition-all duration-500 rounded-full bg-emerald-500';
        recommendationBox.className = 'p-4 rounded-xl text-sm font-medium border bg-emerald-950/50 border-emerald-800 text-emerald-200';
    } else {
        verdictTitle.textContent = 'Riesgo Financiero Alto';
        verdictTitle.className = 'text-2xl font-bold mt-0.5 text-amber-400';
        verdictBadge.textContent = 'NO RECOMENDADO';
        verdictBadge.className = 'px-4 py-1.5 rounded-full text-xs font-bold bg-amber-950 text-amber-300 border border-amber-700';
        probBar.className = 'h-full transition-all duration-500 rounded-full bg-amber-500';
        recommendationBox.className = 'p-4 rounded-xl text-sm font-medium border bg-amber-950/50 border-amber-800 text-amber-200';
    }

    recommendationBox.textContent = result.recommendation;
    jsonInspector.textContent = JSON.stringify({ request: requestPayload, response: result }, null, 2);

    resultCard.classList.remove('hidden');
}

// Ejecutar healthcheck al cargar la página
checkApiHealth();