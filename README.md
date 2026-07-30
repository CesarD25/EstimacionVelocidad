# Estimación de velocidad con YOLO y ByteTrack

Sistema de procesamiento de video que detecta objetos con un modelo YOLO
preentrenado, mantiene sus trayectorias mediante ByteTrack y estima su velocidad
al cruzar dos líneas verticales A y B.

El programa:

1. Normaliza la resolución del video.
2. Detecta personas, vehículos, bicicletas y pelotas con YOLOv8.
3. Descarta objetos estáticos mediante un filtro temporal de movimiento.
4. Asigna IDs visibles consecutivos únicamente a objetos en movimiento.
5. Detecta los cruces de las líneas A y B.
6. Calcula tiempo, velocidad en m/s y velocidad en km/h.
7. Genera un video anotado y archivos CSV y Excel.

## Requisitos

- Windows 10/11, Linux o macOS.
- Python 3.10 o superior.
- Aproximadamente 2 GB de espacio disponible para Python, PyTorch y las
  dependencias.
- Conexión a Internet durante la primera instalación.
- Pantalla con interfaz gráfica para realizar la calibración y visualizar el
  procesamiento.

El procesamiento puede ejecutarse con CPU. Una GPU compatible con PyTorch/CUDA
es opcional y mejora la velocidad de inferencia.

## Instalación

### 1. Abrir una terminal en el proyecto

En PowerShell:

```powershell
cd C:\Users\HP\Videos\EstimacionVelocidad
```

### 2. Crear el entorno virtual

```powershell
python -m venv .venv
```

### 3. Activar el entorno

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Símbolo del sistema:

```bat
.\.venv\Scripts\activate.bat
```

Linux o macOS:

```bash
source .venv/bin/activate
```

Si PowerShell bloquea la activación, también se puede usar directamente el
ejecutable del entorno, sin activarlo:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Instalar dependencias

Con el entorno activado:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

El archivo `yolov8n.pt` contiene los pesos preentrenados. Si no estuviera
presente, Ultralytics intentará descargarlo automáticamente durante la primera
ejecución.

## Ejecución

Con el entorno activado:

```powershell
python main.py
```

Sin activar el entorno:

```powershell
.\.venv\Scripts\python.exe main.py
```

No se utilizan argumentos ni banderas de línea de comandos. El programa muestra
un menú interactivo:

```text
1) Calibrar un video
2) Ejecutar un video
```

Después muestra los videos encontrados en `videos_entrada/`. También permite
escribir manualmente la ruta de otro video.

## Calibración

Cada video o posición de cámara debe calibrarse antes de procesarse:

1. Ejecute `main.py`.
2. Seleccione `Calibrar un video`.
3. Elija el video.
4. Marque cuatro esquinas de una superficie rectangular cuya medida real
   conozca, en este orden:
   superior izquierda, superior derecha, inferior derecha e inferior izquierda.
5. Presione `Enter` e introduzca el ancho y el largo reales en metros.
6. Haga un clic en la posición de la línea vertical A.
7. Haga otro clic en la posición de la línea vertical B.
8. Presione `Enter`.
9. Marque dos esquinas opuestas de la región de interés o ROI.
10. Presione `Enter` para guardar.

En las ventanas de calibración:

- `Enter` confirma la selección.
- `R` o `Esc` reinicia los puntos de la etapa actual.

La calibración se guarda como:

```text
calibraciones/calibracion_<nombre_del_video>.json
```

La homografía de cuatro puntos transforma posiciones de imagen a coordenadas
reales. Por esta razón las dimensiones introducidas deben ser medidas reales y
no valores aproximados.

## Procesamiento de un video

1. Ejecute `main.py`.
2. Seleccione `Ejecutar un video`.
3. Elija un video que tenga una calibración asociada.
4. Presione `Q` en la ventana del video si desea detenerlo antes de terminar.

Las resoluciones se normalizan automáticamente:

- Video horizontal: `960 × 540`.
- Video vertical o tipo reel: `540 × 960`.

La misma resolución se usa para calibrar, detectar, visualizar y exportar. Las
calibraciones antiguas se ajustan automáticamente si fueron creadas con otra
resolución.

## Detección de movimiento e identificadores

ByteTrack conserva internamente la trayectoria de las detecciones de YOLO, pero
el sistema no muestra inmediatamente esos identificadores. Primero verifica el
desplazamiento del objeto durante varios fotogramas.

Un ID visible se asigna solamente cuando el desplazamiento supera el umbral
configurado. Los objetos estáticos:

- no muestran caja ni ID;
- no generan eventos de cruce;
- no producen mediciones de velocidad.

Los IDs visibles son consecutivos: `1`, `2`, `3`, etc.

## Archivos de salida

Por cada video procesado se generan:

```text
videos_salida/<video>_resultado.mp4
resultados/<video>_mediciones.csv
resultados/<video>_mediciones.xlsx
```

El CSV y el Excel pueden contener:

- `track_id`: identificador visible del objeto en movimiento;
- `clase`: clase detectada por YOLO;
- `direccion`: `A->B` o `B->A`;
- `frame_a` y `frame_b`: fotogramas de cruce;
- `distancia_m`: distancia real entre las líneas;
- `tiempo_s`: tiempo transcurrido;
- `velocidad_m_s`;
- `velocidad_km_h`;
- velocidad de referencia y errores, cuando exista una referencia real.

## Configuración

Los parámetros editables están en `config.yaml`.

### Salidas

```yaml
salida:
  directorio_videos: videos_salida
  directorio_resultados: resultados
```

### YOLO y ByteTrack

```yaml
deteccion:
  modelo: yolov8n.pt
  confianza: 0.25
  iou: 0.45
  clases: [0, 1, 2, 3, 5, 7, 32]
  dispositivo: auto
  tracker: bytetrack.yaml
```

Clases COCO configuradas:

| ID | Clase |
|---:|---|
| 0 | Persona |
| 1 | Bicicleta |
| 2 | Carro |
| 3 | Motocicleta |
| 5 | Bus |
| 7 | Camión |
| 32 | Pelota deportiva |

`confianza` controla el mínimo de confianza de YOLO. Un valor mayor reduce
detecciones débiles, pero puede omitir objetos difíciles. `iou` controla la
supresión de cajas superpuestas.

### Filtro de movimiento

```yaml
movimiento:
  ventana_frames: 6
  minimo_frames: 3
  desplazamiento_minimo_px: 12
```

- `ventana_frames`: cantidad máxima de posiciones recientes analizadas.
- `minimo_frames`: observaciones necesarias antes de confirmar movimiento.
- `desplazamiento_minimo_px`: desplazamiento mínimo entre la primera y última
  posición de la ventana.

Si aparecen IDs sobre objetos estáticos, aumente
`desplazamiento_minimo_px`. Si no se identifican objetos lentos, redúzcalo.

## Estructura del proyecto

```text
EstimacionVelocidad/
├── calibraciones/
├── resultados/
├── src/
├── videos_entrada/
│   └── metadata/
├── videos_salida/
├── .gitattributes
├── config.yaml
├── main.py
├── README.md
├── requirements.txt
├── requisitos_proyecto.md
└── yolov8n.pt
```

### Carpetas

| Carpeta | Contenido |
|---|---|
| `calibraciones/` | Homografías, líneas A/B y ROI guardadas para cada video. |
| `resultados/` | Mediciones generadas en CSV y Excel. |
| `src/` | Módulos internos del sistema. |
| `videos_entrada/` | Videos disponibles en el menú interactivo. |
| `videos_entrada/metadata/` | Información y valores reales de referencia de cada video. |
| `videos_salida/` | Videos MP4 anotados generados durante el procesamiento. |
| `.venv/` | Entorno virtual local de Python; puede recrearse con `python -m venv .venv`. |
| `.git/` | Historial y configuración interna de Git. |

### Archivos principales

| Archivo | Función |
|---|---|
| `main.py` | Menú, procesamiento principal, filtro de movimiento, cruces y coordinación de módulos. |
| `config.yaml` | Parámetros de YOLO, ByteTrack, movimiento, visualización y salidas. |
| `requirements.txt` | Dependencias que deben instalarse con `pip`. |
| `yolov8n.pt` | Pesos del modelo YOLOv8 Nano preentrenado. |
| `README.md` | Manual de instalación, ejecución y estructura. |
| `requisitos_proyecto.md` | Resumen del estado de los requisitos funcionales. |
| `.gitattributes` | Reglas de normalización de archivos para Git. |

### Módulos de `src/`

| Archivo | Responsabilidad |
|---|---|
| `src/calibracion_interactiva.py` | Selección de cuatro puntos, líneas verticales A/B, ROI y creación de la homografía. |
| `src/captura.py` | Apertura del video, lectura de propiedades y normalización de resolución. |
| `src/configuracion.py` | Lectura de `config.yaml`. |
| `src/cruces.py` | Determina cuándo una trayectoria cruza A o B. |
| `src/detector_yolo.py` | Carga YOLO y ejecuta detección con ByteTrack. |
| `src/evaluacion.py` | Calcula error absoluto y porcentual frente a una velocidad real. |
| `src/exportacion.py` | Guarda mediciones en CSV y Excel. |
| `src/lineas.py` | Dibuja A/B y transforma puntos mediante homografía. |
| `src/roi.py` | Dibuja la región de interés. |
| `src/velocidad.py` | Calcula tiempo y velocidad entre cruces. |
| `src/visualizacion.py` | Dibuja cajas, centros, clases e identificadores. |
| `src/__init__.py` | Declara `src` como paquete de Python. |

## Metadatos y evaluación

Cada archivo de `videos_entrada/metadata/` puede incluir:

```yaml
Velocidad_de_referencia_m_s: 2.5
```

Cuando ese valor existe, el CSV y el Excel incorporan automáticamente:

- velocidad de referencia;
- error absoluto en m/s;
- error porcentual.

La referencia debe provenir de una medición independiente y fiable. Sin ella,
el sistema puede estimar velocidad, pero no calcular un error experimental
válido.

## Problemas frecuentes

### No existe calibración para el video

Ejecute nuevamente `main.py`, seleccione `Calibrar un video` y calibre ese
archivo antes de procesarlo.

### No aparece ningún ID

- Compruebe que el objeto pertenezca a una clase configurada.
- Reduzca `deteccion.movimiento.desplazamiento_minimo_px`.
- Reduzca ligeramente `deteccion.confianza`.
- Verifique que el objeto esté dentro de la ROI.

### Aparecen IDs en objetos aparentemente estáticos

Aumente `deteccion.movimiento.desplazamiento_minimo_px` o
`deteccion.movimiento.minimo_frames`.

### El modelo se ejecuta lentamente

- Use `yolov8n.pt`, que es la variante ligera.
- Cierre aplicaciones que consuman CPU o GPU.
- Configure un dispositivo CUDA compatible en `deteccion.dispositivo` si tiene
  PyTorch con soporte CUDA.

### Las velocidades no son realistas

- Repita la calibración con medidas reales precisas.
- Marque los cuatro puntos en el orden indicado.
- Coloque A y B sobre el mismo plano utilizado para la homografía.
- Evite videos con movimiento de cámara después de calibrar.
