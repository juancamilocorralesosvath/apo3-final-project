#!/usr/bin/env python3
"""
Diagnostico especializado para detectar sentadillas
Captura valores reales de angulos durante el movimiento
"""
import sys
sys.path.insert(0, 'src')

import cv2
import numpy as np
from src.core.pose_processor import process_frame
from src.utils.kinematic_features import kinematic_extractor

print("=" * 70)
print("DIAGNOSTICO DE SENTADILLAS")
print("=" * 70)
print()
print("Este script te ayudara a calibrar la deteccion de sentadillas")
print()
print("INSTRUCCIONES:")
print("  1. Parate ERGUIDO frente a la camara por 3 segundos")
print("  2. Haz UNA SENTADILLA LENTA (baja en 2 seg, mantén 2 seg, sube 2 seg)")
print("  3. Vuelve a estar ERGUIDO por 2 segundos")
print("  4. Presiona 'q' cuando termines")
print()
print("El script capturara tus angulos y te dira los umbrales ideales")
print("=" * 70)
print()

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: No se pudo abrir la camara")
    sys.exit(1)

# Almacenar datos
standing_samples = []
squatting_samples = []
all_samples = []

frame_count = 0
current_phase = "standing"

while True:
    ret, frame = camera.read()
    if not ret:
        break

    frame_count += 1

    # Copia para dibujar
    display_frame = frame.copy()

    # Procesar frame
    result = process_frame(frame)

    if result and result['landmarks_coords']:
        features = kinematic_extractor.extract_features(result['landmarks_coords'])

        # Angulos
        right_knee = features[0]
        left_knee = features[1]
        right_hip = features[2]
        left_hip = features[3]
        trunk = features[4]

        avg_knee = (right_knee + left_knee) / 2
        avg_hip = (right_hip + left_hip) / 2

        # Guardar muestra
        sample = {
            'frame': frame_count,
            'right_knee': right_knee,
            'left_knee': left_knee,
            'right_hip': right_hip,
            'left_hip': left_hip,
            'avg_knee': avg_knee,
            'avg_hip': avg_hip,
            'trunk': trunk
        }
        all_samples.append(sample)

        # Determinar fase automaticamente basado en angulos
        # Si rodillas < 140, probablemente esta en sentadilla
        if avg_knee < 140:
            current_phase = "squatting"
            squatting_samples.append(sample)
        else:
            if frame_count > 90:  # Despues de 3 segundos
                current_phase = "standing"
            standing_samples.append(sample)

        # UI
        h, w = display_frame.shape[:2]

        # Panel de info
        cv2.rectangle(display_frame, (10, 10), (500, 220), (0, 0, 0), -1)

        y = 35
        cv2.putText(display_frame, f"Frame: {frame_count}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 30

        # Fase actual con color
        phase_color = (0, 255, 0) if current_phase == "squatting" else (255, 255, 255)
        cv2.putText(display_frame, f"Fase: {current_phase.upper()}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, phase_color, 2)
        y += 35

        cv2.putText(display_frame, f"Rodilla Derecha: {right_knee:.1f} deg", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        y += 25

        cv2.putText(display_frame, f"Rodilla Izquierda: {left_knee:.1f} deg", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        y += 25

        cv2.putText(display_frame, f"Rodillas (Promedio): {avg_knee:.1f} deg", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += 30

        cv2.putText(display_frame, f"Caderas (Promedio): {avg_hip:.1f} deg", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += 25

        cv2.putText(display_frame, f"Tronco: {trunk:.1f} deg", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # Indicador de captura
        if current_phase == "squatting":
            cv2.rectangle(display_frame, (w-150, 10), (w-10, 60), (0, 255, 0), -1)
            cv2.putText(display_frame, "CAPTURANDO", (w-140, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.imshow('Diagnostico de Sentadillas', display_frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

# Analisis de resultados
print("\n" + "=" * 70)
print("ANALISIS DE RESULTADOS")
print("=" * 70)

if standing_samples and squatting_samples:
    # Estadisticas de estar parado
    standing_knees = [s['avg_knee'] for s in standing_samples]
    standing_hips = [s['avg_hip'] for s in standing_samples]

    print("\n1. POSICION ERGUIDA (Parado):")
    print(f"   Muestras capturadas: {len(standing_samples)}")
    print(f"   Rodillas:")
    print(f"     - Promedio: {np.mean(standing_knees):.1f} deg")
    print(f"     - Rango: {np.min(standing_knees):.1f} - {np.max(standing_knees):.1f} deg")
    print(f"   Caderas:")
    print(f"     - Promedio: {np.mean(standing_hips):.1f} deg")
    print(f"     - Rango: {np.min(standing_hips):.1f} - {np.max(standing_hips):.1f} deg")

    # Estadisticas de sentadilla
    squat_knees = [s['avg_knee'] for s in squatting_samples]
    squat_hips = [s['avg_hip'] for s in squatting_samples]

    print("\n2. SENTADILLA:")
    print(f"   Muestras capturadas: {len(squatting_samples)}")
    print(f"   Rodillas:")
    print(f"     - Promedio: {np.mean(squat_knees):.1f} deg")
    print(f"     - Rango: {np.min(squat_knees):.1f} - {np.max(squat_knees):.1f} deg")
    print(f"   Caderas:")
    print(f"     - Promedio: {np.mean(squat_hips):.1f} deg")
    print(f"     - Rango: {np.min(squat_hips):.1f} - {np.max(squat_hips):.1f} deg")

    # Calcular umbrales sugeridos
    print("\n" + "=" * 70)
    print("UMBRALES RECOMENDADOS")
    print("=" * 70)

    # Usar el valor mas alto de sentadilla como limite superior
    # y un poco mas bajo como limite inferior
    knee_upper = np.max(squat_knees) + 15  # Margen de seguridad
    knee_lower = np.min(squat_knees) - 10

    hip_upper = np.max(squat_hips) + 15
    hip_lower = np.min(squat_hips) - 10

    print(f"\nPara TU anatomia y estilo de sentadilla:")
    print(f"\nDetector de sentadillas:")
    print(f"  Rodillas: {knee_lower:.0f} < angulo < {knee_upper:.0f}")
    print(f"  Caderas:  {hip_lower:.0f} < angulo < {hip_upper:.0f}")

    print(f"\nCodigo sugerido para activity_predictor.py (linea 243):")
    print(f"```python")
    print(f"is_squat = (")
    print(f"    {knee_lower:.0f} < avg_knee_angle < {knee_upper:.0f} and")
    print(f"    {hip_lower:.0f} < avg_hip_angle < {hip_upper:.0f} and")
    print(f"    abs(right_knee_angle - left_knee_angle) < 30")
    print(f")")
    print(f"```")

    # Verificar separacion
    print("\n" + "=" * 70)
    print("VALIDACION")
    print("=" * 70)

    overlap_knee = knee_upper > np.min(standing_knees)
    overlap_hip = hip_upper > np.min(standing_hips)

    if overlap_knee or overlap_hip:
        print("\n⚠️  ADVERTENCIA: Hay solapamiento entre posturas!")
        if overlap_knee:
            print(f"   - Rodillas: umbral {knee_upper:.0f} > minimo parado {np.min(standing_knees):.0f}")
        if overlap_hip:
            print(f"   - Caderas: umbral {hip_upper:.0f} > minimo parado {np.min(standing_hips):.0f}")
        print("\n   Recomendacion: Haz una sentadilla MAS PROFUNDA")
        print("   o ajusta los umbrales manualmente")
    else:
        print("\n✅ Excelente! Hay clara separacion entre posturas")
        print("   Los umbrales sugeridos deberian funcionar bien")

    # Grafico ASCII simple
    print("\n" + "=" * 70)
    print("DISTRIBUCION DE ANGULOS DE RODILLAS")
    print("=" * 70)
    print()

    min_angle = min(np.min(standing_knees), np.min(squat_knees)) - 10
    max_angle = max(np.max(standing_knees), np.max(squat_knees)) + 10

    print(f"{'Parado:':<12} ", end="")
    standing_range_start = int((np.min(standing_knees) - min_angle) / (max_angle - min_angle) * 50)
    standing_range_end = int((np.max(standing_knees) - min_angle) / (max_angle - min_angle) * 50)
    print(" " * standing_range_start + "█" * (standing_range_end - standing_range_start))

    print(f"{'Sentadilla:':<12} ", end="")
    squat_range_start = int((np.min(squat_knees) - min_angle) / (max_angle - min_angle) * 50)
    squat_range_end = int((np.max(squat_knees) - min_angle) / (max_angle - min_angle) * 50)
    print(" " * squat_range_start + "█" * (squat_range_end - squat_range_start))

    print()
    print(f"{min_angle:.0f}°" + " " * 40 + f"{max_angle:.0f}°")

else:
    print("\n⚠️  No se capturaron suficientes datos")
    print("   Asegurate de:")
    print("   - Estar parado por 3 segundos")
    print("   - Hacer una sentadilla lenta y mantenerla")
    print("   - Volver a estar parado")

print("\n" + "=" * 70)
print("Diagnostico completado!")
print("=" * 70)
