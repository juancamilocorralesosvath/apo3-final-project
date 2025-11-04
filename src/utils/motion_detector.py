"""
Detector de movimiento usando tecnicas clasicas de Vision por Computadora
Complementa el modelo ML con validacion basada en movimiento real
"""
import cv2
import numpy as np
from collections import deque

class MotionDetector:
    """
    Detecta movimiento real en el video usando multiples tecnicas
    """

    def __init__(self, history_length=5):
        # Buffer de frames para analisis temporal
        self.frame_buffer = deque(maxlen=history_length)
        self.gray_buffer = deque(maxlen=history_length)

        # Parametros para deteccion de movimiento
        self.motion_threshold = 25  # Umbral de diferencia de pixeles
        self.min_contour_area = 500  # Area minima para considerar movimiento

        # Metricas de movimiento
        self.motion_history = deque(maxlen=30)  # ~1 segundo a 30fps

        # Flujo optico (Lucas-Kanade)
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # Puntos para rastrear con flujo optico
        self.prev_points = None
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )

    def preprocess_frame(self, frame):
        """
        Preprocesa frame para deteccion de movimiento
        """
        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Aplicar desenfoque gaussiano para reducir ruido
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        return gray

    def detect_frame_difference(self, frame):
        """
        Metodo 1: Deteccion por diferencia de frames
        Detecta cambios entre frames consecutivos
        """
        gray = self.preprocess_frame(frame)
        self.gray_buffer.append(gray)

        if len(self.gray_buffer) < 2:
            return {
                'motion_detected': False,
                'motion_percentage': 0.0,
                'motion_intensity': 0.0,
                'motion_regions': []
            }

        # Calcular diferencia entre frame actual y anterior
        frame_diff = cv2.absdiff(self.gray_buffer[-2], self.gray_buffer[-1])

        # Aplicar umbral
        _, thresh = cv2.threshold(frame_diff, self.motion_threshold, 255, cv2.THRESH_BINARY)

        # Dilatar para llenar huecos
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Encontrar contornos (regiones de movimiento)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtrar contornos por area
        motion_regions = []
        total_motion_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                motion_regions.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'center': (x + w//2, y + h//2)
                })
                total_motion_area += area

        # Calcular metricas
        frame_size = gray.shape[0] * gray.shape[1]
        motion_percentage = (total_motion_area / frame_size) * 100

        # Intensidad de movimiento (promedio de diferencias)
        motion_intensity = np.mean(frame_diff[thresh > 0]) if np.any(thresh > 0) else 0.0

        motion_detected = len(motion_regions) > 0

        return {
            'motion_detected': motion_detected,
            'motion_percentage': motion_percentage,
            'motion_intensity': motion_intensity,
            'motion_regions': motion_regions,
            'diff_frame': thresh  # Para visualizacion
        }

    def detect_optical_flow(self, frame):
        """
        Metodo 2: Deteccion por flujo optico (Lucas-Kanade)
        Rastrea movimiento de puntos caracteristicos
        """
        gray = self.preprocess_frame(frame)

        # Si no hay puntos previos, detectar nuevos
        if self.prev_points is None or len(self.prev_points) < 10:
            self.prev_points = cv2.goodFeaturesToTrack(
                gray,
                mask=None,
                **self.feature_params
            )
            self.prev_gray = gray
            return {
                'flow_detected': False,
                'avg_flow_magnitude': 0.0,
                'flow_vectors': []
            }

        # Calcular flujo optico
        next_points, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.prev_points,
            None,
            **self.lk_params
        )

        if next_points is None:
            self.prev_points = None
            return {
                'flow_detected': False,
                'avg_flow_magnitude': 0.0,
                'flow_vectors': []
            }

        # Seleccionar puntos buenos
        good_prev = self.prev_points[status == 1]
        good_next = next_points[status == 1]

        # Calcular vectores de flujo
        flow_vectors = []
        flow_magnitudes = []

        for prev_pt, next_pt in zip(good_prev, good_next):
            dx = next_pt[0] - prev_pt[0]
            dy = next_pt[1] - prev_pt[1]
            magnitude = np.sqrt(dx**2 + dy**2)

            flow_vectors.append({
                'prev': tuple(prev_pt),
                'next': tuple(next_pt),
                'dx': dx,
                'dy': dy,
                'magnitude': magnitude
            })
            flow_magnitudes.append(magnitude)

        # Metricas
        avg_flow_magnitude = np.mean(flow_magnitudes) if flow_magnitudes else 0.0
        max_flow_magnitude = np.max(flow_magnitudes) if flow_magnitudes else 0.0

        # Actualizar para proximo frame
        self.prev_gray = gray.copy()
        self.prev_points = good_next.reshape(-1, 1, 2)

        return {
            'flow_detected': avg_flow_magnitude > 1.0,  # Umbral de movimiento
            'avg_flow_magnitude': avg_flow_magnitude,
            'max_flow_magnitude': max_flow_magnitude,
            'flow_vectors': flow_vectors,
            'num_tracked_points': len(flow_vectors)
        }

    def analyze_motion_by_region(self, frame, landmarks_coords):
        """
        Metodo 3: Analisis de movimiento por regiones corporales
        Detecta que partes del cuerpo se estan moviendo
        """
        if not landmarks_coords:
            return {
                'upper_body_motion': 0.0,
                'lower_body_motion': 0.0,
                'torso_motion': 0.0,
                'limbs_motion': 0.0
            }

        gray = self.preprocess_frame(frame)
        h, w = gray.shape

        # Definir regiones basadas en landmarks
        # Upper body: hombros, brazos, cabeza
        # Lower body: caderas, piernas

        upper_points = []
        lower_points = []

        upper_landmarks = ['nose', 'left_shoulder', 'right_shoulder',
                          'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']
        lower_landmarks = ['left_hip', 'right_hip', 'left_knee',
                          'right_knee', 'left_ankle', 'right_ankle']

        for name in upper_landmarks:
            if name in landmarks_coords:
                x, y = landmarks_coords[name]
                upper_points.append((int(x * w), int(y * h)))

        for name in lower_landmarks:
            if name in landmarks_coords:
                x, y = landmarks_coords[name]
                lower_points.append((int(x * w), int(y * h)))

        # Crear mascaras para cada region
        upper_mask = np.zeros_like(gray)
        lower_mask = np.zeros_like(gray)

        if len(upper_points) > 0:
            upper_hull = cv2.convexHull(np.array(upper_points))
            cv2.fillConvexPoly(upper_mask, upper_hull, 255)

        if len(lower_points) > 0:
            lower_hull = cv2.convexHull(np.array(lower_points))
            cv2.fillConvexPoly(lower_mask, lower_hull, 255)

        # Analizar movimiento en cada region si tenemos frames previos
        if len(self.gray_buffer) >= 2:
            frame_diff = cv2.absdiff(self.gray_buffer[-2], gray)

            # Movimiento en upper body
            upper_motion_pixels = cv2.bitwise_and(frame_diff, frame_diff, mask=upper_mask)
            upper_motion = np.mean(upper_motion_pixels[upper_motion_pixels > self.motion_threshold])
            upper_motion = upper_motion if not np.isnan(upper_motion) else 0.0

            # Movimiento en lower body
            lower_motion_pixels = cv2.bitwise_and(frame_diff, frame_diff, mask=lower_mask)
            lower_motion = np.mean(lower_motion_pixels[lower_motion_pixels > self.motion_threshold])
            lower_motion = lower_motion if not np.isnan(lower_motion) else 0.0

            return {
                'upper_body_motion': float(upper_motion),
                'lower_body_motion': float(lower_motion),
                'dominant_motion': 'upper' if upper_motion > lower_motion else 'lower',
                'motion_ratio': upper_motion / (lower_motion + 1e-6)  # Evitar division por cero
            }

        return {
            'upper_body_motion': 0.0,
            'lower_body_motion': 0.0,
            'dominant_motion': 'none',
            'motion_ratio': 1.0
        }

    def get_comprehensive_motion_analysis(self, frame, landmarks_coords=None):
        """
        Analisis completo de movimiento combinando todos los metodos
        """
        # Metodo 1: Diferencia de frames
        frame_diff_result = self.detect_frame_difference(frame)

        # Metodo 2: Flujo optico
        optical_flow_result = self.detect_optical_flow(frame)

        # Metodo 3: Analisis por regiones (si hay landmarks)
        region_analysis = {}
        if landmarks_coords:
            region_analysis = self.analyze_motion_by_region(frame, landmarks_coords)

        # Guardar en historial
        self.motion_history.append({
            'frame_diff': frame_diff_result['motion_percentage'],
            'optical_flow': optical_flow_result['avg_flow_magnitude']
        })

        # Calcular tendencia de movimiento (ultimos N frames)
        if len(self.motion_history) >= 10:
            recent_motion = [m['frame_diff'] for m in list(self.motion_history)[-10:]]
            motion_trend = np.mean(recent_motion)
            motion_variance = np.std(recent_motion)
        else:
            motion_trend = 0.0
            motion_variance = 0.0

        # Clasificacion de nivel de movimiento
        motion_level = self._classify_motion_level(
            frame_diff_result['motion_percentage'],
            optical_flow_result['avg_flow_magnitude']
        )

        return {
            'frame_difference': frame_diff_result,
            'optical_flow': optical_flow_result,
            'region_analysis': region_analysis,
            'motion_level': motion_level,
            'motion_trend': motion_trend,
            'motion_variance': motion_variance,
            'timestamp': len(self.motion_history)
        }

    def _classify_motion_level(self, motion_percentage, flow_magnitude):
        """
        Clasifica el nivel de movimiento
        """
        # Combinar metricas
        combined_score = (motion_percentage / 10.0) + flow_magnitude

        if combined_score < 1.0:
            return 'static'  # Sin movimiento
        elif combined_score < 3.0:
            return 'minimal'  # Movimiento minimo
        elif combined_score < 8.0:
            return 'moderate'  # Movimiento moderado
        else:
            return 'high'  # Movimiento alto

    def validate_activity_prediction(self, predicted_activity, motion_analysis):
        """
        Valida si la prediccion del modelo es consistente con el movimiento real detectado
        """
        motion_level = motion_analysis['motion_level']
        frame_diff = motion_analysis['frame_difference']
        optical_flow = motion_analysis['optical_flow']

        # Actividades que requieren movimiento significativo
        dynamic_activities = [
            'caminar', 'walk', 'acercandose', 'alejandose',
            'sentadilla', 'squat', 'giro', 'turn'
        ]

        # Actividades estaticas
        static_activities = [
            'parado', 'stand', 'sentado', 'sit', 'sin movimiento'
        ]

        predicted_lower = predicted_activity.lower()

        # Verificar consistencia
        is_dynamic_activity = any(act in predicted_lower for act in dynamic_activities)
        is_static_activity = any(act in predicted_lower for act in static_activities)

        has_significant_motion = motion_level in ['moderate', 'high']
        has_minimal_motion = motion_level in ['static', 'minimal']

        # Casos de inconsistencia
        inconsistencies = []

        if is_dynamic_activity and has_minimal_motion:
            inconsistencies.append({
                'type': 'motion_mismatch',
                'message': f"Actividad dinamica '{predicted_activity}' pero movimiento {motion_level}",
                'severity': 'high',
                'suggestion': 'Considerar actividad estatica'
            })

        if is_static_activity and has_significant_motion:
            inconsistencies.append({
                'type': 'motion_mismatch',
                'message': f"Actividad estatica '{predicted_activity}' pero movimiento {motion_level}",
                'severity': 'medium',
                'suggestion': 'Considerar actividad dinamica'
            })

        # Validacion especifica para caminata
        if 'caminar' in predicted_lower or 'walk' in predicted_lower:
            if optical_flow['avg_flow_magnitude'] < 2.0:
                inconsistencies.append({
                    'type': 'walking_validation',
                    'message': 'Caminata predicha pero flujo optico bajo',
                    'severity': 'high',
                    'suggestion': 'Verificar si realmente esta caminando'
                })

        # Validacion especifica para sentadilla
        if 'sentadilla' in predicted_lower or 'squat' in predicted_lower:
            region = motion_analysis.get('region_analysis', {})
            if region and region.get('lower_body_motion', 0) < 10:
                inconsistencies.append({
                    'type': 'squat_validation',
                    'message': 'Sentadilla predicha pero bajo movimiento en piernas',
                    'severity': 'medium',
                    'suggestion': 'Verificar postura de sentadilla'
                })

        is_valid = len(inconsistencies) == 0
        confidence_adjustment = 1.0

        if len(inconsistencies) > 0:
            # Reducir confianza basado en inconsistencias
            severity_weights = {'high': 0.3, 'medium': 0.15, 'low': 0.05}
            total_penalty = sum(severity_weights.get(inc['severity'], 0.1) for inc in inconsistencies)
            confidence_adjustment = max(0.5, 1.0 - total_penalty)

        return {
            'is_valid': is_valid,
            'inconsistencies': inconsistencies,
            'confidence_adjustment': confidence_adjustment,
            'motion_level': motion_level,
            'recommendation': 'accept' if is_valid else 'review'
        }

# Instancia global
motion_detector = MotionDetector()
