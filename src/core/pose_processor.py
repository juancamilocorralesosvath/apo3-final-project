import cv2
import mediapipe as mp
from typing import Optional, Dict

# Inicializar MediaPipe una sola vez
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def process_frame(frame) -> Optional[Dict]:
    """
    Procesa un frame de video y devuelve:
    - landmarks_obj: objeto de MediaPipe para dibujar
    - landmarks_coords: diccionario con coordenadas (x, y) de articulaciones clave
    - frame_rgb: imagen en RGB para visualización
    """
    if frame is None:
        return None

    # Convertir BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = pose.process(rgb)
    rgb.flags.writeable = True

    if not results.pose_landmarks:
        return None

    lm = results.pose_landmarks.landmark
    landmarks_dict = {
        'nose': (lm[mp_pose.PoseLandmark.NOSE].x, lm[mp_pose.PoseLandmark.NOSE].y),
        'left_shoulder': (lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y),
        'right_shoulder': (lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y),
        'left_elbow': (lm[mp_pose.PoseLandmark.LEFT_ELBOW].x, lm[mp_pose.PoseLandmark.LEFT_ELBOW].y),
        'right_elbow': (lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y),
        'left_wrist': (lm[mp_pose.PoseLandmark.LEFT_WRIST].x, lm[mp_pose.PoseLandmark.LEFT_WRIST].y),
        'right_wrist': (lm[mp_pose.PoseLandmark.RIGHT_WRIST].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST].y),
        'left_hip': (lm[mp_pose.PoseLandmark.LEFT_HIP].x, lm[mp_pose.PoseLandmark.LEFT_HIP].y),
        'right_hip': (lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y),
        'left_knee': (lm[mp_pose.PoseLandmark.LEFT_KNEE].x, lm[mp_pose.PoseLandmark.LEFT_KNEE].y),
        'right_knee': (lm[mp_pose.PoseLandmark.RIGHT_KNEE].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE].y),
        'left_ankle': (lm[mp_pose.PoseLandmark.LEFT_ANKLE].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE].y),
        'right_ankle': (lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y)
    }

    return {
        'landmarks_obj': results.pose_landmarks,
        'landmarks_coords': landmarks_dict,
        'frame_rgb': rgb
    }