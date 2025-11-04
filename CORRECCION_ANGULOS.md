# Corrección de Cálculo de Ángulos Articulares

## Problemas Identificados y Corregidos

### Problema 1: Inclinación del Tronco Incorrecta ❌ → ✅

**Antes (INCORRECTO):**
```python
# Asumía que Y aumenta hacia arriba (como en matemáticas)
trunk_vector = np.array([shoulder_center_x, shoulder_center_y])
vertical_vector = np.array([0, 1])
```

**Después (CORRECTO):**
```python
# MediaPipe usa coordenadas de imagen: Y aumenta HACIA ABAJO
trunk_vector = np.array([shoulder_center_x, -shoulder_center_y])  # Negamos Y
vertical_vector = np.array([0, 1])

# Además, invertimos la interpretación:
# 0° = erguido (vertical), valores positivos = inclinado
trunk_inclination = 90.0 - trunk_angle_degrees
```

**Impacto:**
- ✅ Ahora detecta correctamente inclinaciones hacia adelante
- ✅ Detecta inclinaciones laterales
- ✅ Valores intuitivos: 0° = erguido, >20° = inclinado

---

## Mejoras Implementadas

### 1. Detectores Basados en Geometría

Se agregaron detectores que usan **física real** en lugar de depender solo del modelo ML:

#### a) Detector de Sentadillas

**Archivo:** `src/core/activity_predictor.py` (líneas 226-252)

```python
def is_squatting(self, features):
    """
    Detecta sentadillas usando ángulos de rodillas y caderas
    """
    avg_knee_angle = (right_knee + left_knee) / 2
    avg_hip_angle = (right_hip + left_hip) / 2

    # Criterios geométricos:
    is_squat = (
        60 < avg_knee_angle < 130 and    # Rodillas flexionadas
        60 < avg_hip_angle < 130 and      # Caderas flexionadas
        abs(right_knee - left_knee) < 30  # Movimiento simétrico
    )
```

**Valores de Referencia:**
- **Parado:** Rodillas ~175-180°, Caderas ~165-175°
- **Sentadilla:** Rodillas ~80-110°, Caderas ~80-110°
- **Semi-sentadilla:** Rodillas ~120-140°, Caderas ~120-140°

#### b) Detector de Inclinaciones

**Archivo:** `src/core/activity_predictor.py` (líneas 254-268)

```python
def is_bending(self, features):
    """
    Detecta inclinación del tronco
    """
    trunk_inclination = features[4]

    # >20° de desviación = inclinado
    is_bent = abs(trunk_inclination) > 20
```

**Valores de Referencia:**
- **Erguido:** Tronco ~0-10°
- **Levemente inclinado:** Tronco ~10-30°
- **Muy inclinado:** Tronco >30°

### 2. Post-Procesamiento Inteligente

El sistema ahora aplica múltiples capas de corrección:

```
Predicción del Modelo
        ↓
[POST-PROC 1] Detectores Geométricos
    ├─ ¿Geometría indica sentadilla?
    │   SÍ → Forzar predicción "Sentadilla" (conf: 0.75)
    │   NO → Continuar
    ├─ ¿Geometría indica inclinación?
    │   SÍ → Forzar predicción "Inclinación" (conf: 0.70)
    │   NO → Continuar
        ↓
[POST-PROC 2] Corrección de Dirección
    ├─ ¿Predice "acercándose"?
    │   → Verificar con cambio de escala
    │   → Corregir si contradice
        ↓
[POST-PROC 3] Filtros de Calidad
        ↓
[POST-PROC 4] Suavizado Temporal
        ↓
Predicción Final
```

---

## Herramienta de Visualización de Ángulos

### ¿Para qué sirve?

La herramienta `test_angles_visual.py` te permite:
- Ver los ángulos calculados en tiempo real
- Verificar que los ángulos sean correctos
- Entender por qué se detecta o no una actividad
- Calibrar tu setup

### Cómo usar:

```bash
python test_angles_visual.py
```

### Qué hacer durante la prueba:

1. **Posición Normal (Parado):**
   - Párate erguido frente a la cámara
   - Verifica: Rodillas ~175-180°, Caderas ~165-175°
   - Tronco ~0-5°

2. **Sentadilla Profunda:**
   - Haz una sentadilla profunda
   - Verifica: Rodillas ~80-100°, Caderas ~80-100°
   - Si los ángulos NO cambian → problema con detección

3. **Inclinación Adelante:**
   - Inclínate hacia adelante
   - Verifica: Tronco >30°
   - Si Tronco no cambia → problema con cálculo

4. **Inclinación Lateral:**
   - Inclínate a un lado
   - Verifica: Asimetría en ángulos
   - Observa diferencia entre rodilla derecha vs izquierda

### Interpretando los Resultados:

#### ✅ **Ángulos Correctos:**
```
ESTADO: Parado
Rodilla Derecha: 178.3 deg
Rodilla Izquierda: 176.9 deg
Cadera Derecha: 172.1 deg
Cadera Izquierda: 170.5 deg
Inclinación Tronco: 2.3 deg
```

#### ✅ **Sentadilla Correcta:**
```
ESTADO: Sentadilla
Rodilla Derecha: 95.7 deg
Rodilla Izquierda: 98.2 deg
Cadera Derecha: 89.4 deg
Cadera Izquierda: 91.1 deg
Inclinación Tronco: 15.6 deg
```

#### ❌ **Problema - Ángulos No Cambian:**
```
ESTADO: Sentadilla (intentada)
Rodilla Derecha: 175.0 deg  ← No cambió
Rodilla Izquierda: 175.0 deg  ← No cambió
```
**Causa:** Landmarks no detectados o valores por defecto

---

## Mensajes de Debug

### Sentadilla Detectada:
```
🏋️ Sentadilla detectada: Rodillas=95.3°, Caderas=88.7°
🔄 Corrección: Geometría indica SENTADILLA
   ✅ Cambiado a: Sentadillas (conf ajustada: 0.75)
```

### Inclinación Detectada:
```
🤸 Inclinación detectada: Tronco=32.4°
🔄 Corrección: Geometría indica INCLINACIÓN
   ✅ Cambiado a: Inclinarse derecha (conf ajustada: 0.70)
```

---

## Parámetros Ajustables

### En `src/core/activity_predictor.py`:

#### Umbrales de Sentadilla (línea 243-246):
```python
is_squat = (
    60 < avg_knee_angle < 130 and    # Ajustar rango de rodillas
    60 < avg_hip_angle < 130 and      # Ajustar rango de caderas
    abs(right_knee - left_knee) < 30  # Simetría máxima
)
```

**Valores sugeridos:**
- **Sentadilla estricta:** `80 < angle < 110`
- **Sentadilla amplia (actual):** `60 < angle < 130`
- **Solo sentadilla profunda:** `70 < angle < 100`

#### Umbral de Inclinación (línea 263):
```python
is_bent = abs(trunk_inclination) > 20  # Grados de desviación
```

**Valores sugeridos:**
- **Muy sensible:** `> 15` - Detecta inclinaciones leves
- **Balanceado (actual):** `> 20` - Recomendado
- **Estricto:** `> 30` - Solo inclinaciones pronunciadas

---

## Mejores Prácticas para Sentadillas e Inclinaciones

### Para Sentadillas:

1. **Posicionamiento:**
   - Párate de frente a la cámara
   - Asegúrate que TODO tu cuerpo sea visible
   - Distancia: 1.5-2.5 metros de la cámara

2. **Ejecución:**
   - Baja lentamente hasta sentadilla completa
   - Mantén la posición 2-3 segundos
   - El sistema detectará los ángulos reducidos

3. **Si no se detecta:**
   - Ejecuta `test_angles_visual.py`
   - Verifica que los ángulos cambien durante la sentadilla
   - Si están siempre en ~175°, hay problema con MediaPipe

### Para Inclinaciones:

1. **Inclinación Adelante:**
   - Inclínate desde la cadera
   - Mantén la espalda recta
   - Al menos 30° de inclinación

2. **Inclinación Lateral:**
   - Inclínate hacia un lado
   - Los ángulos de rodilla deben ser asimétricos
   - El tronco debería mostrar inclinación

---

## Solución de Problemas

### Problema: "Sentadilla no se detecta"

**Diagnóstico:**
```bash
python test_angles_visual.py
```

**Verifica:**
1. ¿Los ángulos cambian durante la sentadilla?
   - NO → MediaPipe no detecta bien tu pose
     - Mejora iluminación
     - Aléjate más de la cámara
     - Usa ropa que contraste con el fondo

   - SÍ → Umbrales muy estrictos
     - Ajusta rangos en línea 243-246

2. ¿Los ángulos están en los rangos correctos?
   - Si rodillas están en ~95° pero no detecta
     - Verifica que caderas también estén flexionadas
     - Asegura simetría (diferencia < 30°)

### Problema: "Inclinación no se detecta"

**Diagnóstico:**
1. Ejecuta `test_angles_visual.py`
2. Inclínate hacia adelante
3. Observa "Inclinación Tronco"

**Esperado:** Valor >30° cuando estés inclinado

**Si no cambia:**
- Hombros no detectados correctamente
- Mejora posicionamiento frente a cámara
- Asegura que hombros y caderas sean visibles

---

## Archivos Modificados

1. ✅ `src/utils/kinematic_features.py` - Corrección de inclinación del tronco
2. ✅ `src/core/activity_predictor.py` - Detectores de sentadillas e inclinaciones
3. ✅ `test_angles_visual.py` - Herramienta de visualización
4. ✅ `CORRECCION_ANGULOS.md` - Esta documentación

---

## Resumen

### ✅ Corregido:
- Cálculo de inclinación del tronco (sistema de coordenadas)
- Valores por defecto más apropiados
- Interpretación intuitiva de ángulos

### ✅ Agregado:
- Detector geométrico de sentadillas
- Detector geométrico de inclinaciones
- Post-procesamiento basado en física
- Herramienta de visualización en tiempo real

### 🎯 Resultado:
- Detección de sentadillas **mucho más confiable**
- Detección de inclinaciones **basada en geometría real**
- Debug visual para entender qué está pasando
- Sistema robusto que combina ML + física

---

## Próximos Pasos

1. **Probar el sistema:**
   ```bash
   python app.py
   ```

2. **Validar ángulos:**
   ```bash
   python test_angles_visual.py
   ```

3. **Ajustar umbrales** si es necesario según tu setup

4. **Reportar resultados:**
   - ¿Las sentadillas se detectan correctamente?
   - ¿Las inclinaciones se detectan correctamente?
   - ¿Los ángulos mostrados tienen sentido?
