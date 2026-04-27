# Video Analyzer — Analizador de Lenguaje No Verbal

Herramienta de análisis automático de lenguaje no verbal en vídeo. Dado un archivo MP4, detecta gestos corporales y faciales fotograma a fotograma usando los modelos de estimación de pose, cara y manos de [MediaPipe](https://developers.google.com/mediapipe).

## ¿Qué detecta?

El sistema reconoce los siguientes gestos:

| Gesto | Descripción |
|---|---|
| **Brazos cruzados** | Las muñecas se cruzan horizontalmente por delante del torso |
| **Brazos abiertos** | Los brazos se extienden a los lados más allá del ancho de hombros |
| **Brazos levantados** | Uno o ambos brazos suben por encima del nivel de los hombros |
| **Cejas levantadas** | Elevación de cejas respecto a la línea base calibrada al inicio del vídeo |
| **Tocar la cara** | Una mano se aproxima a la región facial |
| **Frecuencia de parpadeo** | Tasa de parpadeos por minuto mediante el Eye Aspect Ratio (EAR) |

Todos los umbrales están normalizados por la geometría del propio sujeto (distancia entre hombros, altura del torso, etc.), por lo que el sistema funciona con independencia de la distancia a la cámara.

## Cómo funciona

### Pipeline general

```
Vídeo MP4
    │
    ▼
VideoProcessor ──► MediaPipe (pose + face + hands)
    │                   └─ 33 puntos de pose
    │                   └─ 478 landmarks faciales
    │                   └─ 21 landmarks por mano
    │
    ▼
Stability Gate ──► descarta fotogramas con tracking inestable
    │
    ▼
GestureManager ──► evalúa cada gesto con ventana deslizante
    │                   └─ confirmación: ≥70 % de fotogramas positivos
    │                   └─ cooldown: evita detecciones repetidas
    │
    ├─► Modo normal: guarda un clip de ~20 fotogramas por detección
    └─► Modo debug: genera vídeo completo con anotaciones visuales
```

### Confirmación con ventana deslizante

Para evitar falsos positivos por fotogramas ruidosos, ningún gesto se confirma en un único fotograma. Cada detector mantiene una ventana deslizante de N fotogramas y dispara únicamente cuando la fracción de fotogramas positivos dentro de esa ventana supera un umbral (por defecto el 70 %). Tras cada detección, el detector entra en cooldown y no vuelve a disparar hasta que transcurre el número de fotogramas configurado.

### Puerta de estabilidad

Antes de enviar los landmarks a los detectores, el procesador comprueba que la estimación de pose sea estable: la velocidad de desplazamiento de hombros y caderas debe permanecer por debajo de un umbral durante al menos 5 fotogramas consecutivos. Mientras el tracking sea inestable (por ejemplo, en los primeros fotogramas o tras oclusiones), las ventanas de confirmación se resetean para evitar acumulaciones espurias.

### Inferencia facial mejorada

El modelo de face landmarker tiene dificultades con caras pequeñas (sujeto lejos de la cámara). El sistema resuelve esto usando los landmarks de pose para localizar la cabeza, recortando esa región y pasando solo el recorte ampliado al modelo facial. Los resultados se reescalan al sistema de coordenadas del fotograma original.

## Modos de uso

### Modo normal (por defecto)

Analiza el vídeo y guarda un clip corto (fotogramas antes y después del trigger) por cada gesto detectado:

```bash
python main.py ruta/al/video.mp4
```

Los clips se guardan en:
```
output/
└── nombre_video/
    ├── crossed_arms_frame123.mp4
    ├── raised_eyebrows_frame456.mp4
    └── ...
```

### Modo debug

Genera un vídeo anotado con landmarks, métricas en tiempo real y el estado de confirmación de cada gesto:

```bash
python main.py ruta/al/video.mp4 --debug
```

El resultado es un único vídeo en `output/nombre_video_annotated.mp4`. Incluye una barra lateral con gráficas temporales de las métricas clave, insets de la zona facial y barras de progreso de confirmación por gesto. Es el modo recomendado para ajustar umbrales.

<!-- Captura del modo debug mostrando la barra lateral con métricas y los landmarks superpuestos -->
![Debug overlay](docs/debug_overlay.png)

### Opciones de línea de comandos

```
python main.py <video_path> [--output-dir OUTPUT] [--config CONFIG] [--debug]

Argumentos:
  video_path          Ruta al archivo .mp4 de entrada
  --output-dir        Directorio de salida (default: output/)
  --config            Archivo de configuración (default: config/thresholds.json)
  --debug             Genera vídeo anotado en lugar de clips individuales
```

## Instalación

**Requisitos:** Python 3.10+, conda (recomendado)

```bash
# Clonar el repositorio
git clone <url-del-repo>
cd video_analyzer

# Crear entorno y activar
conda create -n video_analyzer python=3.10
conda activate video_analyzer

# Instalar dependencias
pip install mediapipe opencv-python numpy
```

Los modelos de MediaPipe deben descargarse y colocarse en la carpeta `models/`:

| Archivo | Modelo |
|---|---|
| `pose_landmarker_heavy.task` | [Pose Landmarker Heavy](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) |
| `face_landmarker.task` | [Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) |
| `hand_landmarker.task` | [Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) |

## Configuración de umbrales

Todos los parámetros de detección se editan en `config/thresholds.json`. Cada gesto tiene su propia sección con sus umbrales específicos más los tres parámetros comunes:

```jsonc
{
  "crossed_arms": {
    "wrist_cross_ratio": 0.10,        // solapamiento mínimo de muñecas (fracción del ancho de hombros)
    "confirmation_window": 10,        // fotogramas en la ventana deslizante
    "confirmation_ratio": 0.70,       // fracción positiva requerida para confirmar
    "cooldown_frames": 50             // fotogramas de espera tras una detección
  }
}
```

El bloque `stability` controla la puerta de estabilidad global y `clip` define cuántos fotogramas antes y después del trigger se incluyen en cada clip.

## Añadir un gesto nuevo

El sistema carga automáticamente cualquier detector que se coloque en la carpeta `gestures/`. Para añadir uno:

1. Crear `gestures/mi_gesto.py` con una clase que herede de `BaseGesture`
2. Implementar el método `detect(landmarks) -> bool`
3. Añadir la clave `"mi_gesto"` en `config/thresholds.json` con sus umbrales

```python
from gestures.base_gesture import BaseGesture

class MiGesto(BaseGesture):
    def __init__(self, thresholds: dict):
        super().__init__("mi_gesto", thresholds)

    def detect(self, landmarks: dict) -> bool:
        pose = landmarks.get("pose")
        if pose is None:
            return False
        # ... lógica de detección ...
        return True
```

El `GestureManager` lo descubrirá e inyectará automáticamente en el siguiente arranque.

## Estructura del proyecto

```
video_analyzer/
├── main.py                  # Punto de entrada y CLI
├── config/
│   └── thresholds.json      # Umbrales de todos los gestos
├── core/
│   ├── video_processor.py   # Loop principal, inferencia MediaPipe
│   ├── gesture_manager.py   # Carga y orquesta los detectores
│   ├── clip_saver.py        # Guarda clips en modo normal
│   └── debug_overlay.py     # Anotaciones y barra lateral en modo debug
├── gestures/
│   ├── base_gesture.py      # Clase base con ventana deslizante y cooldown
│   ├── crossed_arms.py
│   ├── open_arms.py
│   ├── raised_arms.py
│   ├── raised_eyebrows.py   # Incluye calibración automática al inicio
│   ├── blink_frequency.py   # EAR + BPM rolling
│   └── touch_face.py
└── models/                  # Modelos .task de MediaPipe (no incluidos en el repo)
```
