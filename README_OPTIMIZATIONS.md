# 🏃‍♂️ Sistema de Reconocimiento de Actividades Humanas - OPTIMIZADO

## 📊 Mejoras Implementadas

### 🎯 Problemas Solucionados

1. **Incompatibilidad Entrenamiento-Producción**: El modelo anterior fue entrenado con ventanas de 30 frames pero la aplicación usaba predicción frame por frame
2. **Características Subóptimas**: Se reemplazaron las 31 características básicas por 16 características cinemáticas avanzadas
3. **Desbalance de Clases**: Implementación de SMOTE para equilibrar las clases minoritarias
4. **Modelo Base**: Actualización a XGBoost con hiperparámetros optimizados

### ⚡ Mejoras de Rendimiento Esperadas

- **Consistencia Temporal**: Uso de ventanas de 30 frames como en el entrenamiento
- **Características Cinemáticas**: 4 ángulos articulares + 1 inclinación de tronco + 11 velocidades lineales
- **Suavizado Temporal**: Buffer de predicciones para evitar cambios bruscos
- **Detección Mejorada**: Filtros anti-sesgo y análisis de incertidumbre

## 🚀 Instalación y Uso

### 1. Instalar Dependencias Nuevas

```bash
# Activar entorno virtual
.\venv_windows\Scripts\activate

# Instalar nuevas dependencias
pip install pandas xgboost imbalanced-learn seaborn matplotlib
```

### 2. Reentrenar el Modelo (Recomendado)

```bash
# Ejecutar reentrenamiento optimizado
python retrain_optimized_model.py
```

Este proceso:
- Carga tus datos de entrenamiento reales
- Crea ventanas temporales de 30 frames
- Extrae 16 características cinemáticas por frame
- Aplica SMOTE para balancear clases
- Entrena XGBoost con búsqueda de hiperparámetros
- Genera matriz de confusión y métricas detalladas

### 3. Probar el Modelo Optimizado

```bash
# Prueba con datos simulados
python test_optimized_model.py
```

### 4. Ejecutar la Aplicación

```bash
# Método 1: Script automatizado
.\run_windows.bat

# Método 2: Comando directo
python app.py
```

## 🔧 Arquitectura del Nuevo Sistema

### Flujo de Predicción Optimizado

1. **Captura de Frame** → MediaPipe detecta 33 puntos corporales
2. **Extracción de Características** → Se generan 16 características cinemáticas
3. **Buffer Temporal** → Se mantiene ventana de 30 frames (1 segundo)
4. **Predicción** → XGBoost predice sobre la ventana completa (480 características)
5. **Post-procesamiento** → Suavizado temporal y filtros anti-sesgo

### Características Cinemáticas (16 total)

#### Ángulos Articulares (4):
- `right_knee_angle`: Ángulo cadera-rodilla-tobillo derecho
- `left_knee_angle`: Ángulo cadera-rodilla-tobillo izquierdo  
- `right_hip_angle`: Ángulo hombro-cadera-rodilla derecho
- `left_hip_angle`: Ángulo hombro-cadera-rodilla izquierdo

#### Inclinación Corporal (1):
- `trunk_inclination`: Ángulo del tronco respecto a la vertical

#### Velocidades Lineales (11):
- `vel_nose`: Velocidad de movimiento de la cabeza
- `vel_left_shoulder`, `vel_right_shoulder`: Velocidades de hombros
- `vel_left_hip`, `vel_right_hip`: Velocidades de caderas
- `vel_left_knee`, `vel_right_knee`: Velocidades de rodillas
- `vel_left_ankle`, `vel_right_ankle`: Velocidades de tobillos
- `vel_left_wrist`, `vel_right_wrist`: Velocidades de muñecas

## 📈 Rendimiento Esperado

### Métricas del EDA (Referencia)
- **Accuracy General**: ~69% (vs 51% del modelo anterior)
- **Clases Fuertes**: `squats`, `approach`, `walk_away` (>90% F1-score)
- **Clases Débiles**: `incline_left/right`, `turn` (~40-50% F1-score)

### Mejoras de Producción
- **Inicialización**: 30 frames (1 segundo) para llenar buffer
- **Latencia**: Predicción en tiempo real después de inicialización
- **Estabilidad**: Suavizado temporal reduce cambios bruscos
- **Robustez**: Filtros anti-sesgo mejoran predicciones

## 🛠️ Estructura de Archivos

```
📁 App_ProyectoFinal/
├── 🚀 retrain_optimized_model.py    # Reentrenamiento optimizado
├── 🧪 test_optimized_model.py       # Pruebas del nuevo modelo
├── 📱 app.py                        # Aplicación Flask con UI mejorada
├── 📁 src/
│   ├── 📁 core/
│   │   ├── activity_predictor.py    # Predictor con ventanas temporales
│   │   └── pose_processor.py        # Procesamiento de poses
│   └── 📁 utils/
│       ├── kinematic_features.py    # Extractor de características cinemáticas
│       └── feature_extractor_real.py # Extractor legacy (backup)
├── 📁 models/                       # Modelos entrenados
│   ├── activity_model.pkl           # XGBoost optimizado
│   ├── scaler.pkl                   # Normalizador
│   ├── label_encoder.pkl            # Codificador de etiquetas
│   └── model_info.json              # Metadatos del modelo
└── 📋 requirements.txt               # Dependencias actualizadas
```

## 🎯 Actividades Reconocibles

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

## 🚨 Notas Importantes

### Primer Uso
- Al activar la cámara, el sistema mostrará "Inicializando (X frames restantes)" hasta llenar el buffer de 30 frames
- Las primeras predicciones pueden ser menos estables mientras se calibra el suavizado temporal

### Rendimiento Esperado
- **Mejores resultados**: Actividades dinámicas con patrones claros (sentadillas, caminar)
- **Desafíos persistentes**: Diferenciación entre inclinaciones izquierda/derecha
- **Mejora general**: Mayor consistencia y menos "saltos" entre predicciones

### Troubleshooting
- Si el modelo no carga: Ejecutar primero `retrain_optimized_model.py`
- Si las predicciones son erráticas: Verificar iluminación y que la persona esté completamente visible
- Si la inicialización es lenta: Es normal, el sistema necesita 30 frames para funcionar óptimamente

## 🔄 Próximas Mejoras

1. **Calibración Automática**: Ajuste dinámico de umbrales según el usuario
2. **Detección de Transiciones**: Mejor manejo de cambios entre actividades
3. **Métricas en Tiempo Real**: Dashboard con estadísticas de confianza
4. **Entrenamiento Continuo**: Actualización del modelo con nuevos datos

---

**¡El sistema está listo para ofrecer una experiencia de reconocimiento de actividades significativamente mejorada! 🎉**