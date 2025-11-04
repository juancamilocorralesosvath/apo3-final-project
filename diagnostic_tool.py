#!/usr/bin/env python3
"""
Herramienta de diagnostico para analizar caracteristicas en tiempo real
"""
import sys
sys.path.insert(0, 'src')

import cv2
import numpy as np
from src.core.pose_processor import process_frame
from src.utils.kinematic_features import extract_kinematic_features

print("=" * 60)
print("HERRAMIENTA DE DIAGNOSTICO HAR")
print("=" * 60)
print("Esta herramienta te ayudara a entender las caracteristicas")
print("que se extraen de tu postura.")
print()
print("Instrucciones:")
print("  1. Quedate COMPLETAMENTE QUIETO por 5 segundos")
print("  2. Luego MUEVETE un poco")
print("  3. Presiona 'q' para salir")
print("=" * 60)
print()

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: No se pudo abrir la camara")
    sys.exit(1)

frame_count = 0
static_samples = []
movement_samples = []

print("Capturando datos...")
print("Fase 1: Quedate QUIETO (5 segundos)")

while True:
    ret, frame = camera.read()
    if not ret:
        break

    frame_count += 1

    # Procesar frame
    result = process_frame(frame)

    if result and result['landmarks_coords']:
        features = extract_kinematic_features(result['landmarks_coords'])

        if len(features) == 16:
            velocities = features[5:16]
            avg_vel = np.mean(np.abs(velocities))
            max_vel = np.max(np.abs(velocities))

            # Primeros 150 frames (5 seg) = quieto
            if frame_count <= 150:
                static_samples.append({
                    'avg_velocity': avg_vel,
                    'max_velocity': max_vel,
                    'features': features.copy()
                })
                status_text = f"QUIETO - Frame {frame_count}/150"
            else:
                movement_samples.append({
                    'avg_velocity': avg_vel,
                    'max_velocity': max_vel,
                    'features': features.copy()
                })
                status_text = f"MOVIMIENTO - Frame {frame_count - 150}"

            # Mostrar info en frame
            cv2.putText(frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Avg Vel: {avg_vel:.4f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Max Vel: {max_vel:.4f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Diagnostico HAR', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    # Terminar despues de 300 frames (10 segundos)
    if frame_count >= 300:
        break

camera.release()
cv2.destroyAllWindows()

# Analizar resultados
print("\n" + "=" * 60)
print("ANALISIS DE RESULTADOS")
print("=" * 60)

if static_samples:
    static_avg_vels = [s['avg_velocity'] for s in static_samples]
    static_max_vels = [s['max_velocity'] for s in static_samples]

    print("\nESTADO QUIETO (estatico):")
    print(f"  Muestras capturadas: {len(static_samples)}")
    print(f"  Velocidad promedio:")
    print(f"    - Media: {np.mean(static_avg_vels):.6f}")
    print(f"    - Min: {np.min(static_avg_vels):.6f}")
    print(f"    - Max: {np.max(static_avg_vels):.6f}")
    print(f"    - Std: {np.std(static_avg_vels):.6f}")
    print(f"  Velocidad maxima:")
    print(f"    - Media: {np.mean(static_max_vels):.6f}")
    print(f"    - Min: {np.min(static_max_vels):.6f}")
    print(f"    - Max: {np.max(static_max_vels):.6f}")

if movement_samples:
    move_avg_vels = [s['avg_velocity'] for s in movement_samples]
    move_max_vels = [s['max_velocity'] for s in movement_samples]

    print("\nESTADO CON MOVIMIENTO:")
    print(f"  Muestras capturadas: {len(movement_samples)}")
    print(f"  Velocidad promedio:")
    print(f"    - Media: {np.mean(move_avg_vels):.6f}")
    print(f"    - Min: {np.min(move_avg_vels):.6f}")
    print(f"    - Max: {np.max(move_avg_vels):.6f}")
    print(f"    - Std: {np.std(move_avg_vels):.6f}")
    print(f"  Velocidad maxima:")
    print(f"    - Media: {np.mean(move_max_vels):.6f}")
    print(f"    - Min: {np.min(move_max_vels):.6f}")
    print(f"    - Max: {np.max(move_max_vels):.6f}")

# Sugerir umbrales
if static_samples and movement_samples:
    print("\n" + "=" * 60)
    print("RECOMENDACIONES DE UMBRALES")
    print("=" * 60)

    # Calcular umbral sugerido (media de quieto + 3 desviaciones)
    static_avg_mean = np.mean(static_avg_vels)
    static_avg_std = np.std(static_avg_vels)
    static_max_mean = np.mean(static_max_vels)
    static_max_std = np.std(static_max_vels)

    suggested_avg_threshold = static_avg_mean + 3 * static_avg_std
    suggested_max_threshold = static_max_mean + 3 * static_max_std

    print(f"\nUmbrales sugeridos (basados en datos capturados):")
    print(f"  avg_velocity_threshold: {suggested_avg_threshold:.6f}")
    print(f"  max_velocity_threshold: {suggested_max_threshold:.6f}")

    print(f"\nUmbrales actuales en el codigo:")
    print(f"  avg_velocity_threshold: 0.015000")
    print(f"  max_velocity_threshold: 0.050000")

    # Verificar si los umbrales actuales son buenos
    if suggested_avg_threshold < 0.015 and suggested_max_threshold < 0.05:
        print("\n✅ Los umbrales actuales parecen apropiados")
    else:
        print("\n⚠️  Los umbrales actuales podrian necesitar ajuste")
        print(f"    Considera cambiar a:")
        print(f"    - avg_velocity < {suggested_avg_threshold:.6f}")
        print(f"    - max_velocity < {suggested_max_threshold:.6f}")

print("\n" + "=" * 60)
print("Diagnostico completado!")
print("=" * 60)
