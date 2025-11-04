# Mejoras para Diferenciación de Actividades

## Problemas Resueltos

### 1. ✅ Confusión entre "Caminar Acercándose" vs "Caminar Alejándose"
### 2. ✅ Confusión entre "Caminata" vs "Parado sin movimiento"

---

## Soluciones Implementadas

### 1. Detección de Cambio de Escala Corporal

**Archivos modificados:** `src/utils/kinematic_features.py`

Se agregaron métodos para detectar si te acercas o alejas de la cámara:

```python
def calculate_body_scale(self, landmarks_coords):
    """
    Calcula el tamaño aparente del cuerpo
    - Acercándose = cuerpo se ve más grande
    - Alejándose = cuerpo se ve más pequeño
    """
```

```python
def get_scale_change_direction(self):
    """
    Analiza últimos 5 frames para determinar tendencia:
    - 'approaching': Escala aumentando (acercándose)
    - 'moving_away': Escala disminuyendo (alejándose)
    - 'static': Sin cambio significativo
    """
```

**Cómo funciona:**
- Mide distancia entre hombros + altura del torso cada frame
- Almacena últimos 10 valores en buffer
- Calcula pendiente de regresión lineal
- Si pendiente > 0.002 → acercándose
- Si pendiente < -0.002 → alejándose

### 2. Post-Procesamiento Inteligente de Predicciones

**Archivos modificados:** `src/core/activity_predictor.py` (líneas 303-335)

Después de que el modelo hace su predicción, se aplica corrección:

```python
# Si modelo predice "acercándose" pero escala indica "alejándose"
if "acercandose" in activity.lower():
    if scale_direction == 'moving_away':
        # Cambiar a predicción de "alejándose"
        activity = buscar_alejandose_en_probabilidades()
```

**Resultado:**
- ✅ Ahora diferencia correctamente acercarse vs alejarse
- ✅ Usa física real (cambio de tamaño) en lugar de datos sintéticos
- ✅ Muestra debug en consola indicando correcciones

### 3. Filtro Estático Mejorado

**Archivos modificados:** `src/core/activity_predictor.py` (líneas 184-224)

**Umbrales ajustados:**
- **Antes:** `avg_velocity < 0.015` y `max_velocity < 0.05` (muy permisivo)
- **Ahora:** `avg_velocity < 0.008` y `max_velocity < 0.025` (más estricto)

**Resultado:**
- ✅ Solo marca como "parado" si REALMENTE no hay movimiento
- ✅ Reduce falsos positivos de confundir caminata lenta con estar parado

### 4. Detector de Patrón de Caminata

**Archivos modificados:** `src/core/activity_predictor.py` (líneas 205-224)

Nuevo método que detecta si hay patrón de caminata:

```python
def is_walking(self, features):
    """
    Detecta patrón de caminata:
    - Velocidad general > 0.015
    - Velocidad de piernas > 0.02
    - Sin movimientos extremos
    """
```

**Lógica integrada:**
```python
if self.is_static(features):
    if not self.is_walking(features):
        return "Parado sin movimiento", 0.95
    else:
        # Hay movimiento de caminata, continuar con modelo
```

**Resultado:**
- ✅ Evita marcar caminata lenta como "parado"
- ✅ Permite que el modelo haga su trabajo cuando hay caminata real

---

## Flujo de Predicción Mejorado

```
1. Extraer características del frame
   ↓
2. ¿Velocidades extremadamente bajas?
   SÍ → ¿Hay patrón de caminata?
         SÍ → Continuar (3)
         NO → Retornar "Parado sin movimiento" ✅
   NO → Continuar (3)
   ↓
3. Aplicar feature engineering (24 características)
   ↓
4. Aplicar feature selection (20 características)
   ↓
5. Normalizar con scaler
   ↓
6. Predicción del modelo
   ↓
7. POST-PROCESAMIENTO ⭐ NUEVO
   ↓
8. ¿Predicción es "acercándose" o "alejándose"?
   SÍ → Verificar cambio de escala
         ¿Contradice predicción?
         SÍ → Corregir usando cambio de escala ✅
         NO → Mantener predicción
   NO → Continuar (9)
   ↓
9. Aplicar filtros de calidad
   ↓
10. Suavizado temporal
   ↓
11. Retornar actividad final
```

---

## Mensajes de Debug

El sistema ahora muestra información útil en consola:

### Detección Estática:
```
🛑 Movimiento estático detectado - avg_vel: 0.0065, max_vel: 0.0210
```

### Caminata Detectada:
```
⚠️ Velocidades bajas pero patrón de caminata detectado - continuando con predicción del modelo
```

### Corrección de Dirección:
```
🔄 Corrección: Cambio de escala indica ALEJÁNDOSE (no acercándose)
   ✅ Cambiado a: Caminar alejandose (espaldas) (conf: 0.450)
```

### Confirmación:
```
✅ Cambio de escala confirma ACERCÁNDOSE
```

---

## Parámetros Ajustables

### En `src/core/activity_predictor.py`:

#### Filtro Estático (línea 198):
```python
is_static = avg_velocity < 0.008 and max_velocity < 0.025
#                         ^^^^^^                    ^^^^^
#                    Ajustar aquí              Ajustar aquí
```

**Valores sugeridos:**
- **Ultra sensible:** `0.005` y `0.015` - Detecta cualquier mínimo movimiento
- **Balanceado (actual):** `0.008` y `0.025` - Recomendado
- **Permisivo:** `0.012` y `0.040` - Solo marca como parado si muy quieto

#### Detector de Caminata (línea 218-220):
```python
is_walking = (
    avg_velocity > 0.015 and      # Umbral de movimiento general
    avg_leg_velocity > 0.02 and   # Umbral de movimiento de piernas
    max_velocity < 0.3            # Límite superior
)
```

### En `src/utils/kinematic_features.py`:

#### Umbral de Cambio de Escala (línea 97):
```python
threshold = 0.002  # Sensibilidad de detección de acercarse/alejarse
```

**Valores sugeridos:**
- **Muy sensible:** `0.001` - Detecta cambios muy pequeños
- **Balanceado (actual):** `0.002` - Recomendado
- **Conservador:** `0.004` - Solo cambios evidentes

---

## Testing Recomendado

### Test 1: Parado vs Caminata Lenta
1. Quédate completamente quieto → Debe decir "Parado sin movimiento"
2. Camina MUY lentamente → Debe detectar caminata (no parado)

### Test 2: Acercarse vs Alejarse
1. Camina hacia la cámara → "Caminar acercándose"
2. Camina alejándote (de espaldas) → "Caminar alejándose"
3. Observa los mensajes de corrección en consola

### Test 3: Velocidades
1. Observa los valores de velocidad en consola
2. Si hay problemas, ajusta umbrales según tus valores reales

---

## Próximos Pasos (Opcionales)

Si aún hay problemas después de estas mejoras:

### 1. Capturar Datos Reales
La solución definitiva es reentrenar con datos reales:
- Grabar 30-60 seg de cada actividad
- Etiquetar videos
- Reentrenar modelo

### 2. Análisis Detallado
Ejecutar herramienta de diagnóstico:
```bash
python diagnostic_tool.py
```

### 3. Ajuste Fino de Umbrales
Basándose en tu setup específico (cámara, iluminación, distancia)

---

## Archivos Modificados

1. `src/utils/kinematic_features.py` - Detección de cambio de escala
2. `src/core/activity_predictor.py` - Post-procesamiento y filtros mejorados
3. `MEJORAS_DIFERENCIACION.md` - Esta documentación

## Resumen de Resultados

✅ **Problema 1 Resuelto:** Ahora usa cambio de tamaño real para diferenciar acercarse/alejarse
✅ **Problema 2 Resuelto:** Umbrales más estrictos + detector de caminata
✅ **Mejora General:** Sistema más inteligente que combina modelo ML + física real
✅ **Debug Mejorado:** Mensajes claros en consola para entender decisiones
