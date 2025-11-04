import numpy as np
from collections import deque

class KinematicFeatureExtractor:
    """
    Extractor de características cinemáticas optimizado para producción
    Genera las 16 características que se usaron en el entrenamiento exitoso
    """

    def __init__(self, history_length=5):
        # Buffer para calcular velocidades (necesitamos frames anteriores)
        self.position_history = deque(maxlen=history_length)
        self.history_length = history_length

        # Buffer para calcular cambios de escala (detectar acercarse/alejarse)
        self.body_scale_history = deque(maxlen=10)

        # Nombres de las características en el orden correcto
        self.feature_names = [
            'right_knee_angle', 'left_knee_angle', 'right_hip_angle', 'left_hip_angle',
            'trunk_inclination', 'vel_nose', 'vel_left_shoulder', 'vel_right_shoulder',
            'vel_left_hip', 'vel_right_hip', 'vel_left_knee', 'vel_right_knee',
            'vel_left_ankle', 'vel_right_ankle', 'vel_left_wrist', 'vel_right_wrist'
        ]
    
    def calculate_angle(self, p1, p2, p3):
        """Calcula ángulo entre tres puntos"""
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)
        
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 180.0  # Ángulo por defecto
        
        cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        angle = np.arccos(cos_angle)
        return np.degrees(angle)
    
    def calculate_body_scale(self, landmarks_coords):
        """
        Calcula la escala del cuerpo (distancia entre puntos clave)
        para detectar si la persona se acerca o aleja de la cámara
        """
        try:
            if 'left_shoulder' in landmarks_coords and 'right_shoulder' in landmarks_coords:
                # Distancia entre hombros (indica escala del cuerpo)
                shoulder_dist = np.linalg.norm(
                    np.array(landmarks_coords['left_shoulder']) -
                    np.array(landmarks_coords['right_shoulder'])
                )

                # Altura del cuerpo aproximada (hombros a caderas)
                if 'left_hip' in landmarks_coords:
                    shoulder_center = (
                        (landmarks_coords['left_shoulder'][0] + landmarks_coords['right_shoulder'][0]) / 2,
                        (landmarks_coords['left_shoulder'][1] + landmarks_coords['right_shoulder'][1]) / 2
                    )
                    hip_center = (
                        (landmarks_coords['left_hip'][0] + landmarks_coords['right_hip'][0]) / 2,
                        (landmarks_coords['left_hip'][1] + landmarks_coords['right_hip'][1]) / 2
                    )
                    body_height = np.linalg.norm(np.array(shoulder_center) - np.array(hip_center))

                    # Combinar ancho y altura para una métrica de escala robusta
                    body_scale = (shoulder_dist + body_height) / 2
                else:
                    body_scale = shoulder_dist

                return body_scale
        except:
            pass

        return 0.5  # Valor por defecto

    def get_scale_change_direction(self):
        """
        Determina si el cuerpo está creciendo (acercándose) o reduciéndose (alejándose)
        Retorna: 'approaching', 'moving_away', 'static'
        """
        if len(self.body_scale_history) < 5:
            return 'static'

        recent_scales = list(self.body_scale_history)[-5:]

        # Calcular tendencia usando regresión lineal simple
        x = np.arange(len(recent_scales))
        y = np.array(recent_scales)

        # Pendiente de la línea de tendencia
        slope = np.polyfit(x, y, 1)[0]

        # Umbral de cambio significativo (ajustar según sea necesario)
        threshold = 0.002

        if slope > threshold:
            return 'approaching'  # Escala aumentando = acercándose
        elif slope < -threshold:
            return 'moving_away'  # Escala disminuyendo = alejándose
        else:
            return 'static'

    def normalize_coordinates(self, landmarks_coords):
        """Normaliza coordenadas respecto al centro de caderas"""
        if 'left_hip' not in landmarks_coords or 'right_hip' not in landmarks_coords:
            return landmarks_coords

        # Centro de caderas
        hip_center_x = (landmarks_coords['left_hip'][0] + landmarks_coords['right_hip'][0]) / 2
        hip_center_y = (landmarks_coords['left_hip'][1] + landmarks_coords['right_hip'][1]) / 2

        # Normalizar todos los puntos
        normalized = {}
        for point, (x, y) in landmarks_coords.items():
            normalized[point] = (x - hip_center_x, y - hip_center_y)

        return normalized
    
    def extract_features(self, landmarks_coords):
        """
        Extrae las 16 características cinemáticas principales
        """
        if not landmarks_coords:
            return np.zeros(16)

        try:
            # Calcular y almacenar escala del cuerpo (para detección de acercarse/alejarse)
            body_scale = self.calculate_body_scale(landmarks_coords)
            self.body_scale_history.append(body_scale)

            # Normalizar coordenadas
            norm_coords = self.normalize_coordinates(landmarks_coords)

            # Agregar al historial para cálculos de velocidad
            self.position_history.append(norm_coords)
            
            features = []
            
            # === ÁNGULOS ARTICULARES (4 características) ===
            
            # 1. Ángulo rodilla derecha (cadera-rodilla-tobillo)
            if all(point in norm_coords for point in ['right_hip', 'right_knee', 'right_ankle']):
                angle = self.calculate_angle(
                    norm_coords['right_hip'], 
                    norm_coords['right_knee'], 
                    norm_coords['right_ankle']
                )
                features.append(angle)
            else:
                features.append(175.0)  # Ángulo por defecto (pierna extendida)
            
            # 2. Ángulo rodilla izquierda
            if all(point in norm_coords for point in ['left_hip', 'left_knee', 'left_ankle']):
                angle = self.calculate_angle(
                    norm_coords['left_hip'], 
                    norm_coords['left_knee'], 
                    norm_coords['left_ankle']
                )
                features.append(angle)
            else:
                features.append(175.0)
            
            # 3. Ángulo cadera derecha (hombro-cadera-rodilla)
            if all(point in norm_coords for point in ['right_shoulder', 'right_hip', 'right_knee']):
                angle = self.calculate_angle(
                    norm_coords['right_shoulder'], 
                    norm_coords['right_hip'], 
                    norm_coords['right_knee']
                )
                features.append(angle)
            else:
                features.append(170.0)  # Ángulo por defecto
            
            # 4. Ángulo cadera izquierda
            if all(point in norm_coords for point in ['left_shoulder', 'left_hip', 'left_knee']):
                angle = self.calculate_angle(
                    norm_coords['left_shoulder'], 
                    norm_coords['left_hip'], 
                    norm_coords['left_knee']
                )
                features.append(angle)
            else:
                features.append(170.0)
            
            # === INCLINACIÓN DEL TRONCO (1 característica) ===

            # 5. Inclinación del tronco
            if all(point in norm_coords for point in ['left_shoulder', 'right_shoulder']):
                # Centro de hombros (ya normalizado, el centro de caderas es (0,0))
                shoulder_center_x = (norm_coords['left_shoulder'][0] + norm_coords['right_shoulder'][0]) / 2
                shoulder_center_y = (norm_coords['left_shoulder'][1] + norm_coords['right_shoulder'][1]) / 2

                # CORREGIDO: En MediaPipe, Y aumenta hacia abajo
                # Vector del tronco (desde caderas (0,0) a hombros)
                trunk_vector = np.array([shoulder_center_x, -shoulder_center_y])  # Negamos Y para correcto sentido
                vertical_vector = np.array([0, 1])  # Vector vertical hacia arriba

                norm_trunk = np.linalg.norm(trunk_vector)
                if norm_trunk > 0:
                    cos_angle = np.dot(trunk_vector, vertical_vector) / norm_trunk
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    trunk_angle_degrees = np.degrees(angle)

                    # Para inclinaciones, es mejor usar la desviación de la vertical
                    # 0° = completamente horizontal, 90° = completamente vertical (erguido)
                    # Invertimos para que sea más intuitivo
                    trunk_inclination = 90.0 - trunk_angle_degrees

                    features.append(trunk_inclination)
                else:
                    features.append(0.0)  # Sin inclinación (erguido)
            else:
                features.append(0.0)
            
            # === VELOCIDADES LINEALES (11 características) ===
            
            landmark_names = [
                'nose', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip',
                'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 
                'left_wrist', 'right_wrist'
            ]
            
            # Calcular velocidades si tenemos historial suficiente
            if len(self.position_history) >= 2:
                prev_coords = self.position_history[-2]
                curr_coords = self.position_history[-1]
                
                for landmark in landmark_names:
                    if landmark in prev_coords and landmark in curr_coords:
                        # Distancia euclidiana entre posiciones consecutivas
                        prev_pos = np.array(prev_coords[landmark])
                        curr_pos = np.array(curr_coords[landmark])
                        velocity = np.linalg.norm(curr_pos - prev_pos)
                        features.append(velocity)
                    else:
                        features.append(0.0)  # Velocidad cero si falta el landmark
            else:
                # Si no hay historial suficiente, todas las velocidades son cero
                features.extend([0.0] * 11)
            
            # Verificar que tenemos exactamente 16 características
            if len(features) != 16:
                print(f"⚠️  Error: Se generaron {len(features)} características, se esperaban 16")
                # Completar o truncar según sea necesario
                if len(features) < 16:
                    features.extend([0.0] * (16 - len(features)))
                else:
                    features = features[:16]
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error extrayendo características cinemáticas: {e}")
            return np.zeros(16)  # Vector de características por defecto

# Instancia global para mantener el estado entre llamadas
kinematic_extractor = KinematicFeatureExtractor()

def extract_kinematic_features(landmarks_coords):
    """
    Función de conveniencia para extraer características cinemáticas
    """
    return kinematic_extractor.extract_features(landmarks_coords)