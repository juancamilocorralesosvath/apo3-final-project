# Problema de Sesgo en el Modelo HAR

## Problema Identificado

El modelo actual predice **"Caminar acercándose"** con alta confianza (75-85%) incluso cuando el usuario está completamente quieto.

### Causa Raíz

**El modelo fue entrenado con datos sintéticos poco realistas** que no representan movimientos humanos reales.

Ver `simplified_har_optimizer.py` líneas 94-95:
```python
y = np.random.choice(n_classes, n_samples,
                   p=[0.15, 0.15, 0.15, 0.10, 0.10, 0.07, 0.07, 0.07, 0.07, 0.07, 0.05])
```

Los datos sintéticos:
- Son generados aleatoriamente con distribución gaussiana
- No capturan patrones reales de movimiento humano
- Tienen distribución de clases desbalanceada
- No incluyen suficientes muestras de "estar quieto"

## Soluciones Implementadas

### 1. Filtro de Movimiento Estático (INMEDIATO)

Se agregó un detector de movimiento mínimo en `src/core/activity_predictor.py`:

```python
def is_static(self, features):
    velocities = features[5:16]
    avg_velocity = np.mean(np.abs(velocities))
    max_velocity = np.max(np.abs(velocities))

    # Umbrales para considerar movimiento estático
    return avg_velocity < 0.015 and max_velocity < 0.05
```

**Cómo funciona:**
- Analiza las velocidades de todos los puntos clave
- Si la velocidad promedio es < 0.015 y la máxima < 0.05, fuerza "Parado sin movimiento"
- Esto filtra el ruido de MediaPipe que se interpretaba como movimiento

### 2. Herramienta de Diagnóstico

Se creó `diagnostic_tool.py` para ayudarte a:
- Analizar las características que genera tu postura
- Ver velocidades en tiempo real
- Obtener umbrales personalizados para tu cámara/setup

**Cómo usar:**
```bash
python diagnostic_tool.py
```
1. Quédate completamente quieto por 5 segundos
2. Luego muévete un poco
3. El script te dará umbrales recomendados

## Soluciones a Largo Plazo

### Opción 1: Recolectar Datos Reales

**Lo más recomendado** - Grabar videos de ti mismo realizando cada actividad:

1. Crear script de captura de datos
2. Grabar 30-60 segundos de cada actividad:
   - Parado quieto (muy importante!)
   - Sentado quieto
   - Caminando hacia la cámara
   - Caminando alejándose
   - Haciendo sentadillas
   - Etc.
3. Etiquetar cada video
4. Reentrenar el modelo con datos reales

### Opción 2: Mejorar Datos Sintéticos

Generar datos sintéticos más realistas:
- Modelar patrones de movimiento humano reales
- Más muestras de estados estáticos
- Distribución balanceada de clases
- Agregar ruido realista de cámara

### Opción 3: Transfer Learning

Usar un modelo pre-entrenado en datos HAR públicos:
- Dataset UCI HAR
- Dataset PAMAP2
- Dataset Opportunity

## Ajuste de Umbrales

Si el filtro de movimiento estático es demasiado agresivo o permisivo, ajusta los umbrales en `src/core/activity_predictor.py`:

```python
# Línea 197
is_static = avg_velocity < 0.015 and max_velocity < 0.05
#                         ^^^^^^                    ^^^^^
#                      Ajusta aquí              Y aquí
```

**Valores sugeridos:**
- **Muy sensible** (detecta cualquier movimiento pequeño): `0.008` y `0.025`
- **Balanceado** (default): `0.015` y `0.050`
- **Permisivo** (solo movimientos claros): `0.025` y `0.080`

## Próximos Pasos Recomendados

1. ✅ **Probar el filtro estático** - Ya implementado
2. 🔍 **Ejecutar diagnostic_tool.py** - Para verificar umbrales
3. 📹 **Opción 1:** Recolectar datos reales y reentrenar (más efectivo)
4. 🔧 **Opción 2:** Ajustar umbrales según tu setup
5. 📊 **Opción 3:** Mejorar datos sintéticos

## Archivos Modificados

- `src/core/activity_predictor.py` - Agregado filtro estático
- `diagnostic_tool.py` - Nueva herramienta de diagnóstico
- `SESGO_MODELO.md` - Esta documentación

## Contacto/Ayuda

Si necesitas ayuda adicional:
1. Ejecuta `diagnostic_tool.py` y comparte los resultados
2. Considera recolectar datos reales para reentrenar
3. Ajusta los umbrales según tus necesidades específicas
