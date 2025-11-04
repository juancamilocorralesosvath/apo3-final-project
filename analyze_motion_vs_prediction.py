#!/usr/bin/env python3
"""
Herramienta de analisis: Compara movimiento real (OpenCV) vs Predicciones del modelo
Ayuda a identificar discrepancias y sugerir mejoras
"""
import sys
sys.path.insert(0, 'src')

import cv2
import numpy as np
from src.core.pose_processor import process_frame
from src.core.activity_predictor import ActivityPredictor
from src.utils.motion_detector import MotionDetector

print("=" * 80)
print("ANALISIS: MOVIMIENTO REAL vs PREDICCIONES DEL MODELO")
print("=" * 80)
print()
print("Esta herramienta compara:")
print("  1. Movimiento REAL detectado por OpenCV")
print("  2. Actividad PREDICHA por tu modelo ML")
print("  3. Valida si son consistentes")
print()
print("Prueba diferentes actividades y observa las discrepancias!")
print()
print("Controles:")
print("  - 'q': Salir")
print("  - 'd': Toggle vista de diferencia de frames")
print("  - 'f': Toggle vista de flujo optico")
print("=" * 80)
print()

# Inicializar componentes
predictor = ActivityPredictor()
motion_detector = MotionDetector()

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: No se pudo abrir la camara")
    sys.exit(1)

# Modos de visualizacion
show_diff = False
show_flow = False

# Estadisticas
stats = {
    'total_frames': 0,
    'consistent': 0,
    'inconsistent': 0,
    'inconsistency_types': {}
}

while True:
    ret, frame = camera.read()
    if not ret:
        break

    stats['total_frames'] += 1

    # Crear copia para dibujar
    display_frame = frame.copy()
    h, w = display_frame.shape[:2]

    # Procesar con MediaPipe
    result = process_frame(frame)

    predicted_activity = "Esperando..."
    confidence = 0.0
    motion_analysis = None
    validation_result = None

    if result and result['landmarks_coords']:
        landmarks = result['landmarks_coords']

        # PREDICCION DEL MODELO
        predicted_activity, confidence = predictor.predict_activity(landmarks)

        # ANALISIS DE MOVIMIENTO REAL
        motion_analysis = motion_detector.get_comprehensive_motion_analysis(frame, landmarks)

        # VALIDACION: Comparar prediccion vs movimiento real
        validation_result = motion_detector.validate_activity_prediction(
            predicted_activity,
            motion_analysis
        )

        # Actualizar estadisticas
        if validation_result['is_valid']:
            stats['consistent'] += 1
        else:
            stats['inconsistent'] += 1
            for inc in validation_result['inconsistencies']:
                inc_type = inc['type']
                stats['inconsistency_types'][inc_type] = stats['inconsistency_types'].get(inc_type, 0) + 1

    # === VISUALIZACION ===

    # Panel principal de informacion
    panel_height = 280
    cv2.rectangle(display_frame, (0, 0), (w, panel_height), (0, 0, 0), -1)

    y = 30

    # Titulo
    cv2.putText(display_frame, "ANALISIS DE CONSISTENCIA", (20, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y += 35

    # Seccion 1: Prediccion del modelo
    cv2.putText(display_frame, "1. PREDICCION DEL MODELO:", (20, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 2)
    y += 25

    cv2.putText(display_frame, f"   Actividad: {predicted_activity}", (20, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 20

    cv2.putText(display_frame, f"   Confianza: {confidence:.2f}", (20, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 30

    # Seccion 2: Movimiento real detectado
    if motion_analysis:
        motion_level = motion_analysis['motion_level']
        motion_colors = {
            'static': (150, 150, 150),
            'minimal': (255, 200, 100),
            'moderate': (100, 255, 100),
            'high': (100, 100, 255)
        }
        motion_color = motion_colors.get(motion_level, (255, 255, 255))

        cv2.putText(display_frame, "2. MOVIMIENTO REAL (OpenCV):", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 2)
        y += 25

        cv2.putText(display_frame, f"   Nivel: {motion_level.upper()}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, motion_color, 2)
        y += 20

        frame_diff = motion_analysis['frame_difference']
        cv2.putText(display_frame,
                   f"   Movimiento: {frame_diff['motion_percentage']:.1f}%", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y += 18

        optical_flow = motion_analysis['optical_flow']
        cv2.putText(display_frame,
                   f"   Flujo optico: {optical_flow['avg_flow_magnitude']:.2f}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y += 25

    # Seccion 3: Validacion
    if validation_result:
        y += 5
        is_valid = validation_result['is_valid']
        status_text = "CONSISTENTE" if is_valid else "INCONSISTENTE"
        status_color = (100, 255, 100) if is_valid else (100, 100, 255)

        cv2.putText(display_frame, "3. VALIDACION:", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 2)
        y += 25

        cv2.putText(display_frame, f"   Estado: {status_text}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        y += 25

        # Mostrar inconsistencias si las hay
        if not is_valid and validation_result['inconsistencies']:
            for i, inc in enumerate(validation_result['inconsistencies'][:2]):  # Max 2
                severity_colors = {
                    'high': (0, 0, 255),
                    'medium': (0, 165, 255),
                    'low': (0, 255, 255)
                }
                color = severity_colors.get(inc['severity'], (200, 200, 200))

                msg = inc['message']
                if len(msg) > 50:
                    msg = msg[:47] + "..."

                cv2.putText(display_frame, f"   ! {msg}", (20, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                y += 18

    # Panel de estadisticas
    stats_y = h - 120
    cv2.rectangle(display_frame, (0, stats_y), (w, h), (0, 0, 0), -1)

    consistency_rate = (stats['consistent'] / max(stats['total_frames'], 1)) * 100

    cv2.putText(display_frame, f"ESTADISTICAS:", (20, stats_y + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.putText(display_frame, f"Frames: {stats['total_frames']}", (20, stats_y + 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    cv2.putText(display_frame, f"Consistentes: {stats['consistent']} ({consistency_rate:.1f}%)",
               (20, stats_y + 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)

    cv2.putText(display_frame, f"Inconsistentes: {stats['inconsistent']}",
               (20, stats_y + 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)

    # Mostrar tipos de inconsistencias mas comunes
    if stats['inconsistency_types']:
        most_common = max(stats['inconsistency_types'].items(), key=lambda x: x[1])
        cv2.putText(display_frame, f"Mas comun: {most_common[0]} ({most_common[1]})",
                   (20, stats_y + 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 100), 1)

    # Vistas adicionales
    if show_diff and motion_analysis and 'diff_frame' in motion_analysis['frame_difference']:
        diff_vis = motion_analysis['frame_difference']['diff_frame']
        diff_vis = cv2.cvtColor(diff_vis, cv2.COLOR_GRAY2BGR)
        diff_vis = cv2.resize(diff_vis, (w//3, h//3))
        display_frame[0:h//3, w-w//3:w] = diff_vis

        cv2.putText(display_frame, "Diferencia Frames", (w-w//3+10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    if show_flow and motion_analysis:
        flow_vectors = motion_analysis['optical_flow'].get('flow_vectors', [])
        for vec in flow_vectors[:50]:  # Mostrar max 50 vectores
            prev = tuple(map(int, vec['prev']))
            next_pt = tuple(map(int, vec['next']))
            cv2.arrowedLine(display_frame, prev, next_pt, (0, 255, 0), 1, tipLength=0.3)

    # Instrucciones
    cv2.putText(display_frame, "'d': Toggle Diff | 'f': Toggle Flow | 'q': Salir",
               (w//2 - 200, h - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    cv2.imshow('Analisis: Movimiento vs Prediccion', display_frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('d'):
        show_diff = not show_diff
        print(f"Vista diferencia de frames: {'ON' if show_diff else 'OFF'}")
    elif key == ord('f'):
        show_flow = not show_flow
        print(f"Vista flujo optico: {'ON' if show_flow else 'OFF'}")

camera.release()
cv2.destroyAllWindows()

# Reporte final
print("\n" + "=" * 80)
print("REPORTE FINAL")
print("=" * 80)
print(f"\nFrames procesados: {stats['total_frames']}")
print(f"Predicciones consistentes: {stats['consistent']} ({consistency_rate:.1f}%)")
print(f"Predicciones inconsistentes: {stats['inconsistent']}")

if stats['inconsistency_types']:
    print("\nTipos de inconsistencias detectadas:")
    sorted_types = sorted(stats['inconsistency_types'].items(), key=lambda x: x[1], reverse=True)
    for inc_type, count in sorted_types:
        percentage = (count / stats['inconsistent']) * 100
        print(f"  - {inc_type}: {count} veces ({percentage:.1f}%)")

print("\n" + "=" * 80)
print("RECOMENDACIONES")
print("=" * 80)

if consistency_rate < 60:
    print("\n⚠️  BAJA CONSISTENCIA (<60%)")
    print("\nProblemas identificados:")

    if 'motion_mismatch' in stats['inconsistency_types']:
        print("\n1. ACTIVIDADES DINAMICAS vs ESTATICAS")
        print("   Problema: El modelo confunde movimiento con estatico")
        print("   Solucion:")
        print("     - Reentrenar con datos mas balanceados")
        print("     - Agregar filtros de movimiento mas estrictos")
        print("     - Usar deteccion de movimiento como feature adicional")

    if 'walking_validation' in stats['inconsistency_types']:
        print("\n2. DETECCION DE CAMINATA")
        print("   Problema: Predice caminata sin movimiento real suficiente")
        print("   Solucion:")
        print("     - Validar con flujo optico antes de confirmar caminata")
        print("     - Ajustar umbrales de velocidad en el modelo")
        print("     - Considerar datos de entrenamiento de caminata")

    if 'squat_validation' in stats['inconsistency_types']:
        print("\n3. DETECCION DE SENTADILLAS")
        print("   Problema: Predice sentadillas sin movimiento en piernas")
        print("   Solucion:")
        print("     - Ya implementado: Detector geometrico de sentadillas")
        print("     - Validar movimiento en lower body region")

elif consistency_rate < 80:
    print("\n✓ CONSISTENCIA ACEPTABLE (60-80%)")
    print("El modelo funciona razonablemente bien.")
    print("Considera las inconsistencias detectadas para mejoras incrementales.")

else:
    print("\n✅ EXCELENTE CONSISTENCIA (>80%)")
    print("El modelo esta alineado con el movimiento real!")

print("\n" + "=" * 80)
print("Analisis completado!")
print("=" * 80)
