'''
    API de Predicción de Radón - Versión Producción
================================================
Sistema completo con:
- Gestión de histórico para cálculo de lags
- Caché en memoria para datos meteorológicos
- Soporte para múltiples dispositivos Arduino
- Limpieza automática de datos antiguos

Autor: Estrella Sánchez Montaño
Fecha: Febrero 2026
'''

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import numpy as np
import joblib
import sqlite3
import httpx
import asyncio
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

CONFIG = {
    "modelo_path": "modelo_radon_1h.pkl",
    "db_path": "radon_historico.db",
    "wunderground_api_key": "TU_API_KEY_AQUI",
    "wunderground_station": "EXTREMADURA123",
    "cache_meteo_minutos": 10,
    "historico_horas_mantener": 12,
    "intervalo_limpieza_minutos": 60,
}

# =============================================================================
# CACHÉ EN MEMORIA (alternativa simple a Redis)
# =============================================================================

class CacheMeteo:
    '''
    Caché en memoria para datos meteorológicos.
    Evita llamadas repetidas a Wunderground.    
    Para escalar, reemplazar por Redis.
    '''
    def __init__(self, ttl_minutos: int = 10):
        self.datos: Dict = {}
        self.timestamp: Optional[datetime] = None
        self.ttl = timedelta(minutes=ttl_minutos)
    
    def get(self) -> Optional[Dict]:
        '''Obtener datos si no han expirado'''
        if self.timestamp is None:
            return None
        if datetime.now() - self.timestamp > self.ttl:
            logger.info("Caché meteo expirada")
            return None
        logger.info("Usando datos meteoteorológicos desde caché")
        return self.datos
    
    def set(self, datos: Dict):
        '''Guardar datos con timestamp actual'''
        self.datos = datos
        self.timestamp = datetime.now()
        logger.info(f"Caché meteo actualizada: {datos}")
    
    def invalidar(self):
        '''Fuerza recarga en próxima consulta'''
        self.timestamp = None

# Instancia global de caché
cache_meteo = CacheMeteo(ttl_minutos=CONFIG["cache_meteo_minutos"])

# =============================================================================
# BASE DE DATOS SQLITE - HISTÓRICO DE DATOS DEL ARDUINO
# =============================================================================

def inicializar_db():
    '''Crear las tablas si no existen'''
    conn = sqlite3.connect(CONFIG["db_path"])
    
    # Tabla para lecturas de Arduino
    conn.execute('''
        CREATE TABLE IF NOT EXISTS lecturas_arduino (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id VARCHAR(50) NOT NULL,
            timestamp DATETIME NOT NULL,
            humedad REAL NOT NULL,
            temperatura REAL NOT NULL,
            co2 REAL
        )
    ''')
    
    # Tabla para lecturas meteorológicas
    conn.execute('''
        CREATE TABLE IF NOT EXISTS lecturas_meteo (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            estacion_id VARCHAR(50) NOT NULL,
            timestamp DATETIME NOT NULL,
            presion REAL NOT NULL,
            temperatura REAL,
            humedad REAL
        )
    ''')
    
    # Índices para consultas rápidas
    conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_arduino_device_time 
        ON lecturas_arduino(device_id, timestamp)
    ''')
    conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_meteo_time 
        ON lecturas_meteo(timestamp)
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Base de datos inicializada")

def guardar_lectura_arduino(device_id: str, humedad: float, temperatura: float, co2: float = None):
    '''Guardar la lectura del Arduino en histórico'''
    conn = sqlite3.connect(CONFIG["db_path"])
    conn.execute('''
        INSERT INTO lecturas_arduino (device_id, timestamp, humedad, temperatura, co2)
        VALUES (?, ?, ?, ?, ?)
    ''', (device_id, datetime.now(), humedad, temperatura, co2))
    conn.commit()
    conn.close()

def guardar_lectura_meteo(estacion_id: str, presion: float, temperatura: float = None, humedad: float = None):
    '''Guardar la lectura meteorológica en histórico'''
    conn = sqlite3.connect(CONFIG["db_path"])
    conn.execute('''
        INSERT INTO lecturas_meteo (estacion_id, timestamp, presion, temperatura, humedad)
        VALUES (?, ?, ?, ?, ?)
    ''', (estacion_id, datetime.now(), presion, temperatura, humedad))
    conn.commit()
    conn.close()

def obtener_humedad_lag(device_id: str, horas: int) -> Optional[float]:
    '''Obtener humedad de hace N horas para un dispositivo'''
    conn = sqlite3.connect(CONFIG["db_path"])
    tiempo_objetivo = datetime.now() - timedelta(hours=horas)
    
    resultado = conn.execute('''
        SELECT humedad FROM lecturas_arduino 
        WHERE device_id = ? AND timestamp <= ? 
        ORDER BY timestamp DESC LIMIT 1
    ''', (device_id, tiempo_objetivo)).fetchone()
    
    conn.close()
    return resultado[0] if resultado else None

def obtener_presion_lag(horas: int) -> Optional[float]:
    '''Obtener presión de hace N horas'''
    conn = sqlite3.connect(CONFIG["db_path"])
    tiempo_objetivo = datetime.now() - timedelta(hours=horas)
    
    resultado = conn.execute('''
        SELECT presion FROM lecturas_meteo 
        WHERE timestamp <= ? 
        ORDER BY timestamp DESC LIMIT 1
    ''', (tiempo_objetivo,)).fetchone()
    
    conn.close()
    return resultado[0] if resultado else None

def obtener_temp_media(device_id: str, horas: int) -> Optional[float]:
    '''Calcular media de temperatura interior de las últimas N horas'''
    conn = sqlite3.connect(CONFIG["db_path"])
    tiempo_inicio = datetime.now() - timedelta(hours=horas)
    
    resultado = conn.execute('''
        SELECT AVG(temperatura) FROM lecturas_arduino 
        WHERE device_id = ? AND timestamp >= ?
    ''', (device_id, tiempo_inicio)).fetchone()
    
    conn.close()
    return resultado[0] if resultado else None

def limpiar_historico_antiguo():
    '''Eliminar registros más antiguos que el límite configurado'''
    conn = sqlite3.connect(CONFIG["db_path"])
    limite = datetime.now() - timedelta(hours=CONFIG["historico_horas_mantener"])
    
    cursor = conn.execute("DELETE FROM lecturas_arduino WHERE timestamp < ?", (limite,))
    eliminados_arduino = cursor.rowcount
    
    cursor = conn.execute("DELETE FROM lecturas_meteo WHERE timestamp < ?", (limite,))
    eliminados_meteo = cursor.rowcount
    
    conn.commit()
    conn.close()
    
    logger.info(f"Limpieza: eliminados {eliminados_arduino} registros Arduino, {eliminados_meteo} registros meteo")


# =============================================================================
# CONSULTA API WUNDERGROUND
# =============================================================================

async def consultar_wunderground() -> Dict:
    '''
    Consultar API de Weather Underground.
    Documentación: https://docs.google.com/document/d/1eKCnKXI9xnoMGRRzOL1xPCBihNV2rOet08qpE_gArAY
    '''
    url = "https://api.weather.com/v2/pws/observations/current"
    params = {
        "stationId": CONFIG["wunderground_station"],
        "format": "json",
        "units": "m",  # métrico
        "apiKey": CONFIG["wunderground_api_key"]
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            obs = data["observations"][0]
            return {
                "presion": obs["metric"]["pressure"],
                "temperatura": obs["metric"]["temp"],
                "humedad": obs["humidity"],
                "timestamp": obs["obsTimeLocal"]
            }
    except httpx.TimeoutException:
        logger.error("Timeout consultando Wunderground")
        raise HTTPException(status_code=503, detail="Servicio meteorológico no disponible")
    except Exception as e:
        logger.error(f"Error consultando Wunderground: {e}")
        raise HTTPException(status_code=503, detail=f"Error meteorológico: {str(e)}")

async def obtener_meteo_actual() -> Dict:
    '''
    Obtener datos meteorológicos usando caché.
    Solo consulta Wunderground si la caché ha expirado.
    '''
    # Intentar obtener dats de caché
    datos = cache_meteo.get()
    if datos:
        return datos
    
    # Si no hay caché válida, consultar API
    datos = await consultar_wunderground()
    
    # Guardar en caché y en histórico
    cache_meteo.set(datos)
    guardar_lectura_meteo(
        estacion_id=CONFIG["wunderground_station"],
        presion=datos["presion"],
        temperatura=datos["temperatura"],
        humedad=datos["humedad"]
    )
    
    return datos

# =============================================================================
# MODELOS 
# =============================================================================

class DatosArduino(BaseModel):
    '''Datos enviados por el Arduino'''
    device_id: str = Field(..., description="Identificador único del Arduino", example="arduino_garaje")
    humedad: float = Field(..., ge=0, le=100, description="Humedad relativa (%)", example=65.5)
    temperatura: float = Field(..., ge=-10, le=50, description="Temperatura interior (°C)", example=22.3)
    co2: Optional[float] = Field(None, ge=300, le=5000, description="CO2 (ppm)", example=450)

class PrediccionResponse(BaseModel):
    '''Respuesta de predicción'''
    device_id: str
    prediccion_bq: float
    intervalo_confianza: Dict[str, float]
    nivel_alerta: str
    mensaje: str
    accion_recomendada: str
    timestamp: datetime
    meteo_source: str  # "cache" o "api"

class HealthResponse(BaseModel):
    '''Estado del servicio'''
    status: str
    modelo_cargado: bool
    historico: Dict
    cache_meteo_activa: bool
    timestamp: datetime

class ErrorHistorico(BaseModel):
    '''Error por histórico insuficiente'''
    error: str
    device_id: str
    horas_disponibles: float
    horas_requeridas: int = 10
    mensaje: str

# =============================================================================
# APLICACIÓN FASTAPI
# =============================================================================

# Cargar modelo
try:
    modelo = joblib.load(CONFIG["modelo_path"])
    logger.info("Modelo cargado correctamente")
except Exception as e:
    logger.error(f"Error cargando modelo: {e}")
    modelo = None

# Tarea de limpieza periódica
async def tarea_limpieza_periodica():
    '''Ejecutar limpieza de histórico cada hora'''
    while True:
        await asyncio.sleep(CONFIG["intervalo_limpieza_minutos"] * 60)
        limpiar_historico_antiguo()

@asynccontextmanager
async def lifespan(app: FastAPI):
    '''Inicialización y limpieza de la aplicación'''
    # Startup
    inicializar_db()
    asyncio.create_task(tarea_limpieza_periodica())
    logger.info("API iniciada")
    yield
    # Shutdown
    logger.info("API detenida")

app = FastAPI(
    title="API Predicción de Radón",
    description='''
    Sistema de predicción de concentración de radón en interiores.
    
    Características
    - Predicción con 1 hora de anticipación
    - Soporte para múltiples dispositivos Arduino
    - Integración con Weather Underground para datos meteorológicos
    - Sistema de alertas (Normal, Preventiva, Crítica)
    
    Arquitectura
    - Caché en memoria para datos meteorológicos (TTL: 10 min)
    - Histórico SQLite para cálculo de lags (retención: 12 horas)
    - Modelo LightGBM entrenado con datos de Arduino A2
    ''',
    version="2.0.0",
    lifespan=lifespan
)

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    '''Verificar el estado del servicio'''
    return HealthResponse(
        status="healthy" if modelo else "degraded",
        modelo_cargado=modelo is not None,
        historico=obtener_estadisticas_historico(),
        cache_meteo_activa=cache_meteo.timestamp is not None,
        timestamp=datetime.now()
    )

@app.post("/predict", response_model=PrediccionResponse)
async def predecir(datos: DatosArduino):
    '''
    Generar predicción de radón para dentro de 1 hora.
    
    El Arduino envía sus datos y la API:
    1. Guarda la lectura en el histórico
    2. Obtiene datos meteorológicos (de caché o Wunderground)
    3. Calcula los lags necesarios desde el histórico
    4. Genera la predicción
    '''
    if not modelo:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    # Guardar lectura actual
    guardar_lectura_arduino(datos.device_id, datos.humedad, datos.temperatura, datos.co2)
    
    # Obtener datos meteorológicos
    meteo_from_cache = cache_meteo.get() is not None
    meteo = await obtener_meteo_actual()
    
    # Calcular lags y medias desde histórico
    humedad_lag_4h = obtener_humedad_lag(datos.device_id, 4)
    presion_lag_6h = obtener_presion_lag(6)
    temp_media_10h = obtener_temp_media(datos.device_id, 10)
    
    # Verificar histórico suficiente
    if None in [humedad_lag_4h, presion_lag_6h, temp_media_10h]:
        # Calcular horas disponibles
        conn = sqlite3.connect(CONFIG["db_path"])
        primera = conn.execute('''
            SELECT MIN(timestamp) FROM lecturas_arduino WHERE device_id = ?
        ''', (datos.device_id,)).fetchone()[0]
        conn.close()
        
        if primera:
            horas_disponibles = (datetime.now() - datetime.fromisoformat(primera)).total_seconds() / 3600
        else:
            horas_disponibles = 0
        
        raise HTTPException(
            status_code=425,  # Código Too Early
            detail={
                "error": "Histórico insuficiente",
                "device_id": datos.device_id,
                "horas_disponibles": round(horas_disponibles, 1),
                "horas_requeridas": 10,
                "mensaje": f"El sistema necesita 10 horas de histórico para calcular los lags. Actualmente hay {round(horas_disponibles, 1)} horas. La primera predicción estará disponible en {round(10 - horas_disponibles, 1)} horas."
            }
        )
    
    # Calcular features temporales
    hora = datetime.now().hour
    hora_sin = np.sin(2 * np.pi * hora / 24)
    hora_cos = np.cos(2 * np.pi * hora / 24)
    
    # Preparar vector de features (el orden es crítico)
    features = [
        datos.humedad,          # SHT85_Humedad_r
        meteo["presion"],       # Presion_meteo
        datos.temperatura,      # SHT85_Temp
        humedad_lag_4h,         # Humedad_Lag_4h
        presion_lag_6h,         # Presion_Lag_6h
        temp_media_10h,         # Temp_Int_Media_10h
        hora_sin,               # Hora_Sin
        hora_cos                # Hora_Cos
    ]
    
    # Predecir
    prediccion = float(modelo.predict([features])[0])
    
    # Calcular intervalo de confianza (basado en el error del modelo)
    STD_ERROR = 37.7  # Desviación estándar del error del modelo
    intervalo = {
        "inferior": round(prediccion - STD_ERROR, 1),
        "superior": round(prediccion + STD_ERROR, 1)
    }
    
    # Determinar nivel de alerta
    # Umbral preventivo = 300 - STD_ERROR ≈ 260 Bq/m³
    if prediccion > 300:
        nivel = "CRITICA"
        mensaje = "ALERTA CRÍTICA: Se prevé superar el umbral legal de 300 Bq/m³"
        accion = "Ventilar inmediatamente"
    elif prediccion > 260:
        nivel = "PREVENTIVA"
        mensaje = "ALERTA PREVENTIVA: Niveles cercanos al umbral"
        accion = "Considerar ventilación"
    else:
        nivel = "NORMAL"
        mensaje = "Niveles dentro del rango seguro"
        accion = "Ninguna acción requerida"
    
    return PrediccionResponse(
        device_id=datos.device_id,
        prediccion_bq=round(prediccion, 1),
        intervalo_confianza=intervalo,
        nivel_alerta=nivel,
        mensaje=mensaje,
        accion_recomendada=accion,
        timestamp=datetime.now(),
        meteo_source="cache" if meteo_from_cache else "api"
    )

@app.post("/predict/batch")
async def predecir_batch(datos_lista: List[DatosArduino]):
    '''Predicción para múltiples dispositivos en una sola llamada'''
    if len(datos_lista) > 100:
        raise HTTPException(status_code=400, detail="Máximo 100 dispositivos por lote")
    
    resultados = []
    for datos in datos_lista:
        try:
            resultado = await predecir(datos)
            resultados.append(resultado)
        except HTTPException as e:
            resultados.append({"device_id": datos.device_id, "error": e.detail})
    
    return {"predicciones": resultados, "total": len(resultados)}

@app.delete("/cache/meteo")
async def invalidar_cache_meteo():
    '''Fuerza recarga de datos meteorológicos'''
    cache_meteo.invalidar()
    return {"mensaje": "Caché meteorológica invalidada"}

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
