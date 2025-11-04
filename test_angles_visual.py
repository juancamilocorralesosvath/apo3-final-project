#!/usr/bin/env python3
"""
Script para visualizar y validar los ángulos calculados
"""
import sys
sys.path.insert(0, 'src')

import cv2
import numpy as np
from src.core.pose_processor import process_frame
from src.utils.kinematic_features import kinematic_extractor

def draw_angle(image, p1, p2, p3, angle_value, color=(0, 255, 0)):
    """
    Dibuja un ángulo en la imagen
    p2 es el vértice del ángulo
    """
    h, w = image.shape[:2]

    # Convertir coordenadas normalizadas a píxeles
    pt1 = (int(p1[0] * w), int(p1[1] * h))
    pt2 = (int(p2[0] * w), int(p2[1] * h))
    pt3 = (int(p3[0] * w), int(p3[1] * h))

    # Dibujar líneas
    cv2.line(image, pt1, pt2, color, 2)
    cv2.line(image, pt2, pt3, color, 2)

    # Dibujar vértice
    cv2.circle(image, pt2, 5, (0, 0, 255), -1)

    # Mostrar valor del ángulo
    cv2.putText(image, f"{angle_value:.1f}deg",
                (pt2[0] + 10, pt2[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

print("=" * 60)
print("VISUALIZADOR DE ANGULOS")
print("=" * 60)
print("Este script te mostrara los angulos calculados en tiempo real")
print()
print("Prueba estas posiciones:")
print("  1. Parado normal - Rodillas ~180°, Caderas ~170°")
print("  2. Sentadilla profunda - Rodillas ~90°, Caderas ~90°")
print("  3. Inclinacion adelante - Tronco con angulo reducido")
print("  4. Inclinacion lateral - Observa asimetria")
print()
print("Presiona 'q' para salir")
print("=" * 60)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: No se pudo abrir la camara")
    sys.exit(1)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Hacer una copia para dibujar
    display_frame = frame.copy()

    # Procesar frame
    result = process_frame(frame)

    if result and result['landmarks_coords']:
        landmarks = result['landmarks_coords']

        # Extraer características para obtener ángulos
        features = kinematic_extractor.extract_features(landmarks)

        # Ángulos están en índices 0-4
        right_knee_angle = features[0]
        left_knee_angle = features[1]
        right_hip_angle = features[2]
        left_hip_angle = features[3]
        trunk_inclination = features[4]

        # Panel de información
        info_y = 30
        cv2.rectangle(display_frame, (10, 10), (400, 180), (0, 0, 0), -1)

        cv2.putText(display_frame, "ANGULOS ARTICULARES:", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 30

        cv2.putText(display_frame, f"Rodilla Derecha: {right_knee_angle:.1f} deg", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        info_y += 25

        cv2.putText(display_frame, f"Rodilla Izquierda: {left_knee_angle:.1f} deg", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        info_y += 25

        cv2.putText(display_frame, f"Cadera Derecha: {right_hip_angle:.1f} deg", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        info_y += 25

        cv2.putText(display_frame, f"Cadera Izquierda: {left_hip_angle:.1f} deg", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        info_y += 25

        cv2.putText(display_frame, f"Inclinacion Tronco: {trunk_inclination:.1f} deg", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # Dibujar ángulos en la imagen
        # Rodilla derecha (amarillo)
        if all(k in landmarks for k in ['right_hip', 'right_knee', 'right_ankle']):
            draw_angle(display_frame,
                      landmarks['right_hip'],
                      landmarks['right_knee'],
                      landmarks['right_ankle'],
                      right_knee_angle,
                      (0, 255, 255))

        # Cadera derecha (verde)
        if all(k in landmarks for k in ['right_shoulder', 'right_hip', 'right_knee']):
            draw_angle(display_frame,
                      landmarks['right_shoulder'],
                      landmarks['right_hip'],
                      landmarks['right_knee'],
                      right_hip_angle,
                      (0, 255, 0))

        # Guía de valores esperados
        guide_y = display_frame.shape[0] - 100
        cv2.rectangle(display_frame, (10, guide_y - 30), (500, display_frame.shape[0] - 10), (0, 0, 0), -1)

        cv2.putText(display_frame, "VALORES DE REFERENCIA:", (20, guide_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(display_frame, "Parado: Rodillas ~175-180, Caderas ~165-175", (20, guide_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(display_frame, "Sentadilla: Rodillas ~80-110, Caderas ~80-110", (20, guide_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(display_frame, "Inclinacion: Tronco < 45 grados", (20, guide_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    cv2.imshow('Visualizador de Angulos', display_frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

print("\nVisualizacion completada!")
