# 🏃‍♂️ Sistema de Reconocimiento de Actividades Humanas

Sistema de IA para detectar actividades humanas en tiempo real usando MediaPipe y Machine Learning.

## 🎯 Características

- **11 actividades detectadas**: Caminar, girar, sentarse, ponerse de pie, sentadillas, inclinaciones, etc.
- **98.2% de precisión** en datos reales
- **Interfaz web moderna** con cámara en tiempo real y análisis de videos
- **Modelo entrenado** con 17,124 muestras reales

## 🚀 Instalación y Uso

### Para Windows:
```cmd
# 1. Instalar dependencias (solo la primera vez)
install_windows.bat

# 2. Ejecutar la aplicación
run_windows.bat
```

**🌐 La aplicación se abrirá automáticamente en:** http://localhost:5000

### Para Linux/macOS:
```bash
# 1. Dar permisos de ejecución (solo la primera vez)
chmod +x install_unix.sh run_unix.sh

# 2. Instalar dependencias (solo la primera vez)
./install_unix.sh

# 3. Ejecutar la aplicación
./run_unix.sh
```

### Requisitos Previos:
- **Python 3.8+** instalado en tu sistema
- **Cámara web** conectada (para función en tiempo real)
- **Conexión a internet** (para descargar dependencias)

**🌐 La aplicación estará disponible en:** http://localhost:5000

## 📁 Estructura del Proyecto

```
App_ProyectoFinal/
├── app.py                        # 🚀 Aplicación principal
├── create_real_model.py          # 🤖 Crear modelo desde datos
├── eda_proyecto_final.py         # 📊 Análisis exploratorio
├── models/                       # 🧠 Modelos entrenados
│   ├── activity_model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
├── src/
│   ├── core/
│   │   ├── pose_processor.py     # MediaPipe
│   │   └── activity_predictor.py # Predicción
│   ├── interface/
│   │   └── gradio_app.py         # Interfaz Web
│   └── utils/
│       └── feature_extractor_real.py # Extracción características
├── data/
│   ├── all_video_landmarks_mediapipe.json  # Datos landmarks
│   └── VIDEOS FINAL TALLER LABELING.json  # Etiquetas
└── venv/                         # Entorno virtual
```

## 🎭 Actividades Detectadas

1. **Caminar acercándose**
2. **Caminar alejándose (espaldas)**
3. **Giro 180° derecha**
4. **Giro 180° izquierda**
5. **Inclinarse derecha**
6. **Inclinarse izquierda**
7. **Parado sin movimiento**
8. **Ponerse de pie**
9. **Sentadillas**
10. **Sentado sin movimiento**
11. **Sentarse**

## 🛠️ Tecnologías

- **MediaPipe**: Detección de pose
- **scikit-learn**: Machine Learning  
- **Flask**: Interfaz web moderna
- **Python 3.12**

## 📊 Rendimiento del Modelo

- **Precisión**: 98.2%
- **Datos de entrenamiento**: 17,124 muestras
- **Características**: 31 por frame
- **Algoritmo**: Random Forest

## 🎮 Funcionalidades

### 📹 Cámara en Tiempo Real
- **Detección instantánea** de actividades
- **Visualización de landmarks** de pose en tiempo real
- **Confianza actualizada** cada 500ms
- **Interfaz web moderna** y responsive

### 🎮 Cómo Usar la Aplicación
1. Ejecuta `run_windows.bat`
2. Abre tu navegador en `http://localhost:5000`
3. Haz clic en "📹 Iniciar Cámara"
4. ¡Realiza movimientos y observa la detección en tiempo real!

## 🔧 Desarrollo

Para re-entrenar el modelo:
```bash
python create_real_model.py
```
