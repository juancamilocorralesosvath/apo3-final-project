import numpy as np

def extract_features_from_landmarks(landmarks_coords):
    """
    Extrae exactamente las 31 características que espera tu modelo real
    Compatible con MediaPipe landmarks
    """
    features = []
    
    # Mapeo de nombres a índices MediaPipe
    mediapipe_mapping = {
        'nose': 0,
        'left_shoulder': 11, 'right_shoulder': 12,
        'left_elbow': 13, 'right_elbow': 14,
        'left_wrist': 15, 'right_wrist': 16,
        'left_hip': 23, 'right_hip': 24,
        'left_knee': 25, 'right_knee': 26,
        'left_ankle': 27, 'right_ankle': 28
    }
    
    # 26 coordenadas (13 puntos x 2 coordenadas)
    key_points = [
        'nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    for point in key_points:
        if point in landmarks_coords:
            x, y = landmarks_coords[point]
            features.extend([x, y])
        else:
            # Debug: notificar landmarks faltantes
            print(f"⚠️  Landmark faltante: {point}")
            features.extend([0.5, 0.5])  # Valores por defecto centrados
    
    # 5 características adicionales
    # 1. Inclinación de hombros
    if 'left_shoulder' in landmarks_coords and 'right_shoulder' in landmarks_coords:
        shoulder_tilt = landmarks_coords['left_shoulder'][1] - landmarks_coords['right_shoulder'][1]
        features.append(shoulder_tilt)
    else:
        features.append(0.0)
    
    # 2-3. Centro de cadera
    if 'left_hip' in landmarks_coords and 'right_hip' in landmarks_coords:
        hip_center_x = (landmarks_coords['left_hip'][0] + landmarks_coords['right_hip'][0]) / 2
        hip_center_y = (landmarks_coords['left_hip'][1] + landmarks_coords['right_hip'][1]) / 2
        features.extend([hip_center_x, hip_center_y])
    else:
        features.extend([0.5, 0.6])
    
    # 4-5. Ángulos aproximados (placeholder por ahora)
    features.extend([170.0, 175.0])  # hip_angle, knee_angle
    
    return np.array(features)

def mediapipe_to_coords(landmarks):
    """
    Convierte landmarks de MediaPipe al formato esperado por el extractor
    """
    if not landmarks:
        return {}
    
    coords = {}
    
    # Mapear índices MediaPipe a nombres
    if len(landmarks) >= 29:
        coords['nose'] = (landmarks[0].x, landmarks[0].y)
        coords['left_shoulder'] = (landmarks[11].x, landmarks[11].y)
        coords['right_shoulder'] = (landmarks[12].x, landmarks[12].y)
        coords['left_elbow'] = (landmarks[13].x, landmarks[13].y)
        coords['right_elbow'] = (landmarks[14].x, landmarks[14].y)
        coords['left_wrist'] = (landmarks[15].x, landmarks[15].y)
        coords['right_wrist'] = (landmarks[16].x, landmarks[16].y)
        coords['left_hip'] = (landmarks[23].x, landmarks[23].y)
        coords['right_hip'] = (landmarks[24].x, landmarks[24].y)
        coords['left_knee'] = (landmarks[25].x, landmarks[25].y)
        coords['right_knee'] = (landmarks[26].x, landmarks[26].y)
        coords['left_ankle'] = (landmarks[27].x, landmarks[27].y)
        coords['right_ankle'] = (landmarks[28].x, landmarks[28].y)
    
    return coords
