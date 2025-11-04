import joblib
import numpy as np
import pandas as pd
import os
from collections import deque
from ..utils.kinematic_features import extract_kinematic_features
from ..utils.motion_detector import MotionDetector

class ActivityPredictor:
    def __init__(self, enable_motion_validation=True):
        self.model = None
        self.scaler = None
        self.label_encoder = None

        # Atributos para modelo optimizado
        self.optimized_scaler = None
        self.feature_selector = None
        self.is_optimized_model = False

        # Buffers para ventanas temporales
        self.frame_buffer = deque(maxlen=30)
        self.feature_buffer = deque(maxlen=30)
        self.window_size = 30

        # Para suavizado temporal
        self.recent_predictions = []
        self.max_history = 5

        # Detector de movimiento (NUEVO)
        self.enable_motion_validation = enable_motion_validation
        if self.enable_motion_validation:
            self.motion_detector = MotionDetector()
            print("✅ Validación de movimiento activada")

        self.load_models()
        
    def apply_feature_engineering(self, features):
        """
        ⚡ Aplicar ingeniería de características EXACTAMENTE como en simplified_har_optimizer.py
        """
        if len(features) != 16:
            return features

        try:
            # Convertir a numpy array
            features = np.array(features)

            # Mapear características básicas al formato esperado
            # Formato actual de kinematic_features: [right_knee_angle, left_knee_angle, right_hip_angle, left_hip_angle, trunk_inclination, vel_nose, vel_left_shoulder, vel_right_shoulder, vel_left_hip, vel_right_hip, vel_left_knee, vel_right_knee, vel_left_ankle, vel_right_ankle, vel_left_wrist, vel_right_wrist]
            # Formato esperado por optimizer: [left_hip_angle, right_hip_angle, left_knee_angle, right_knee_angle, trunk_inclination, vel_left_shoulder, vel_right_shoulder, vel_left_elbow, vel_right_elbow, vel_left_hip, vel_right_hip, vel_left_knee, vel_right_knee, vel_left_ankle, vel_right_ankle, vel_nose]

            # Reordenar características para que coincidan con el entrenamiento
            remapped_features = [
                features[3],  # left_hip_angle (era índice 3)
                features[2],  # right_hip_angle (era índice 2)
                features[1],  # left_knee_angle (era índice 1)
                features[0],  # right_knee_angle (era índice 0)
                features[4],  # trunk_inclination (se mantiene)
                features[6],  # vel_left_shoulder (era índice 6)
                features[7],  # vel_right_shoulder (era índice 7)
                features[14], # vel_left_elbow (usar vel_left_wrist como aproximación)
                features[15], # vel_right_elbow (usar vel_right_wrist como aproximación)
                features[8],  # vel_left_hip (era índice 8)
                features[9],  # vel_right_hip (era índice 9)
                features[10], # vel_left_knee (era índice 10)
                features[11], # vel_right_knee (era índice 11)
                features[12], # vel_left_ankle (era índice 12)
                features[13], # vel_right_ankle (era índice 13)
                features[5],  # vel_nose (era índice 5)
            ]

            # Crear DataFrame temporal para aplicar feature engineering como en el optimizer
            feature_names = [
                'left_hip_angle', 'right_hip_angle', 'left_knee_angle', 'right_knee_angle',
                'trunk_inclination',
                'vel_left_shoulder', 'vel_right_shoulder', 'vel_left_elbow', 'vel_right_elbow',
                'vel_left_hip', 'vel_right_hip', 'vel_left_knee', 'vel_right_knee',
                'vel_left_ankle', 'vel_right_ankle', 'vel_nose'
            ]

            df = pd.DataFrame([remapped_features], columns=feature_names)

            # Aplicar EXACTAMENTE el mismo feature engineering que en simplified_har_optimizer.py (líneas 176-208)

            # 1. hip_angle_diff
            df['hip_angle_diff'] = df['left_hip_angle'] - df['right_hip_angle']

            # 2. hip_angle_mean
            df['hip_angle_mean'] = (df['left_hip_angle'] + df['right_hip_angle']) / 2

            # 3. knee_angle_diff
            df['knee_angle_diff'] = df['left_knee_angle'] - df['right_knee_angle']

            # 4. knee_angle_mean
            df['knee_angle_mean'] = (df['left_knee_angle'] + df['right_knee_angle']) / 2

            # 5. total_velocity
            vel_cols = [col for col in df.columns if col.startswith('vel_')]
            df['total_velocity'] = df[vel_cols].abs().sum(axis=1)

            # 6. velocity_variance
            df['velocity_variance'] = df[vel_cols].var(axis=1)

            # 7. max_velocity
            df['max_velocity'] = df[vel_cols].abs().max(axis=1)

            # 8. left_right_vel_ratio
            left_vel_cols = [col for col in vel_cols if 'left' in col]
            right_vel_cols = [col for col in vel_cols if 'right' in col]

            df['left_right_vel_ratio'] = (
                df[left_vel_cols].abs().sum(axis=1) /
                (df[right_vel_cols].abs().sum(axis=1) + 0.001)
            )

            # Convertir de vuelta a numpy array
            enhanced_features = df.values.flatten()

            return enhanced_features

        except Exception as e:
            print(f"⚠️ Error en feature engineering: {e}")
            import traceback
            traceback.print_exc()
            return features  # Retornar características originales en caso de error
        
    def load_models(self):
        """Carga los modelos entrenados (prioriza modelo optimizado)"""
        try:
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            model_path = os.path.join(base_path, 'models')
            
            # Intentar cargar modelo optimizado primero
            optimized_model_path = os.path.join(model_path, 'optimized_har_model.pkl')
            
            if os.path.exists(optimized_model_path):
                print("🚀 Cargando modelo optimizado (Accuracy: 87.5%)...")
                model_data = joblib.load(optimized_model_path)

                if isinstance(model_data, dict):
                    # Formato del modelo optimizado
                    self.model = model_data['ensemble_model']
                    self.optimized_scaler = model_data.get('scaler')  # Scaler específico del modelo optimizado
                    self.feature_selector = model_data.get('feature_selector')  # Feature selector
                    self.label_encoder = model_data.get('label_encoder')
                    self.is_optimized_model = True
                    accuracy = model_data.get('test_accuracy', 'N/A')
                    print(f"✅ Modelo optimizado cargado (Accuracy: {accuracy})")
                    print(f"   Scaler optimizado: {'✅' if self.optimized_scaler is not None else '❌'}")
                    print(f"   Feature selector: {'✅' if self.feature_selector is not None else '❌'}")
                else:
                    # Modelo simple
                    self.model = model_data
                    self.is_optimized_model = False
                    print("✅ Modelo optimizado cargado (formato simple)")
                
                # Cargar scaler original para compatibilidad si es necesario
                if not self.is_optimized_model:
                    try:
                        self.scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))
                        print("✅ Scaler original cargado por separado")
                    except:
                        print("⚠️ Scaler original no encontrado, usando normalización básica")
                
                if self.label_encoder is None:
                    try:
                        self.label_encoder = joblib.load(os.path.join(model_path, 'label_encoder.pkl'))
                        print("✅ Label encoder cargado por separado")
                    except:
                        # Crear encoder básico para 11 actividades
                        from sklearn.preprocessing import LabelEncoder
                        self.label_encoder = LabelEncoder()
                        activities = ['stand', 'sit', 'walk', 'squat', 'bend_forward', 
                                    'incline_left', 'incline_right', 'turn', 
                                    'approach', 'walk_away', 'transition']
                        self.label_encoder.fit(activities)
                        print("✅ Label encoder creado por defecto")
                        
            else:
                # Fallback al modelo original
                print("🔄 Modelo optimizado no encontrado, cargando modelo original...")
                self.model = joblib.load(os.path.join(model_path, 'activity_model.pkl'))
                self.scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))
                self.label_encoder = joblib.load(os.path.join(model_path, 'label_encoder.pkl'))
                self.is_optimized_model = False
                print("✅ Modelo original cargado")
                
        except Exception as e:
            print(f"❌ Error cargando modelos: {e}")
            
    def is_static(self, features):
        """
        Detecta si el usuario está estático basándose en velocidades bajas
        MEJORADO: Umbrales más estrictos para evitar confundir caminata con estar parado
        """
        # Las velocidades están en los índices 5-15 (11 velocidades)
        velocities = features[5:16]

        # Calcular velocidad promedio
        avg_velocity = np.mean(np.abs(velocities))
        max_velocity = np.max(np.abs(velocities))

        # Umbrales MÁS ESTRICTOS para evitar falsos positivos
        # Solo marcar como estático si REALMENTE no hay movimiento
        is_static = avg_velocity < 0.008 and max_velocity < 0.025

        if is_static:
            print(f"🛑 Movimiento estático detectado - avg_vel: {avg_velocity:.4f}, max_vel: {max_velocity:.4f}")

        return is_static

    def is_walking(self, features):
        """
        Detecta patrón de caminata basándose en velocidades moderadas y consistentes
        """
        velocities = features[5:16]
        avg_velocity = np.mean(np.abs(velocities))
        max_velocity = np.max(np.abs(velocities))

        # Velocidades de piernas (importantes para caminata)
        leg_velocities = [features[9], features[10], features[11], features[12]]  # hips, knees
        avg_leg_velocity = np.mean(np.abs(leg_velocities))

        # Patrón de caminata: velocidad moderada, especialmente en piernas
        is_walking = (
            avg_velocity > 0.015 and  # Hay movimiento
            avg_leg_velocity > 0.02 and  # Las piernas se mueven
            max_velocity < 0.3  # Pero no es un movimiento extremo
        )

        return is_walking

    def is_squatting(self, features):
        """
        Detecta sentadillas con múltiples criterios
        MEJORADO: Más robusto para diferentes profundidades y estilos de sentadilla
        """
        # Índices: [right_knee, left_knee, right_hip, left_hip, trunk, velocidades...]
        right_knee_angle = features[0]
        left_knee_angle = features[1]
        right_hip_angle = features[2]
        left_hip_angle = features[3]
        trunk_inclination = features[4]

        # Promedios
        avg_knee_angle = (right_knee_angle + left_knee_angle) / 2
        avg_hip_angle = (right_hip_angle + left_hip_angle) / 2

        # Criterios múltiples para detectar sentadilla
        # Criterio 1: Rodillas significativamente flexionadas
        knees_bent = avg_knee_angle < 145  # Más permisivo: <145° (antes <130°)

        # Criterio 2: Caderas flexionadas (puede variar más)
        hips_bent = avg_hip_angle < 150  # Más permisivo: <150° (antes <130°)

        # Criterio 3: Simetría razonable (algunas personas no son perfectamente simétricas)
        knee_symmetry = abs(right_knee_angle - left_knee_angle) < 40  # Más tolerante: 40° (antes 30°)

        # Criterio 4: NO está parado recto (ambas rodillas deben estar flexionadas)
        not_standing = avg_knee_angle < 160  # Si rodillas > 160° definitivamente está parado

        # NUEVA LÓGICA: Múltiples niveles de detección
        # Nivel 1: Sentadilla clara (todos los criterios)
        clear_squat = knees_bent and hips_bent and knee_symmetry and not_standing

        # Nivel 2: Sentadilla parcial o estilo diferente (criterios relajados)
        partial_squat = (
            avg_knee_angle < 150 and  # Rodillas algo flexionadas
            avg_hip_angle < 155 and   # Caderas algo flexionadas
            not_standing and
            knee_symmetry
        )

        is_squat = clear_squat or partial_squat

        if is_squat:
            squat_type = "completa" if clear_squat else "parcial"
            print(f"🏋️ Sentadilla {squat_type} detectada: Rodillas={avg_knee_angle:.1f}°, Caderas={avg_hip_angle:.1f}°")

        return is_squat

    def detect_bending_type(self, features, landmarks_coords):
        """
        Detecta el TIPO de inclinación que está realizando la persona
        NUEVO: Considera múltiples patrones de inclinación
        """
        # Índices de características
        right_knee_angle = features[0]
        left_knee_angle = features[1]
        right_hip_angle = features[2]
        left_hip_angle = features[3]
        trunk_inclination = features[4]

        # Velocidades (para detectar si es dinámico)
        velocities = features[5:16]
        avg_velocity = np.mean(np.abs(velocities))

        # === ANÁLISIS GEOMÉTRICO ===

        # 1. INCLINACIÓN FRONTAL (hacia adelante)
        # Características: caderas muy flexionadas, tronco hacia adelante
        avg_hip_angle = (right_hip_angle + left_hip_angle) / 2
        forward_bend = (
            avg_hip_angle < 140 and  # Caderas flexionadas
            abs(trunk_inclination) > 15 and  # Tronco inclinado
            abs(right_knee_angle - left_knee_angle) < 35  # Simétrico
        )

        # 2. INCLINACIÓN LATERAL (derecha o izquierda)
        # Características: asimetría en caderas/rodillas
        hip_asymmetry = abs(right_hip_angle - left_hip_angle)
        knee_asymmetry = abs(right_knee_angle - left_knee_angle)

        lateral_bend = (
            (hip_asymmetry > 15 or knee_asymmetry > 15) and  # Asimetría significativa
            not forward_bend  # No es inclinación frontal
        )

        # Determinar dirección de inclinación lateral
        bend_direction = None
        if lateral_bend:
            if right_hip_angle < left_hip_angle - 10:
                bend_direction = "derecha"
            elif left_hip_angle < right_hip_angle - 10:
                bend_direction = "izquierda"

        # 3. INCLINACIÓN LEVE (persona con poca flexibilidad o movimiento sutil)
        # Más permisivo para personas que no se inclinan mucho
        subtle_bend = (
            abs(trunk_inclination) > 12 and  # Inclinación moderada (antes 20°)
            avg_hip_angle < 155 and  # Caderas algo flexionadas
            not forward_bend and
            not lateral_bend
        )

        # === RESULTADO ===
        bend_type = None
        bend_confidence = 0.0

        if forward_bend:
            bend_type = "frontal"
            bend_confidence = 0.80
            print(f"🤸 Inclinación FRONTAL detectada: Caderas={avg_hip_angle:.1f}°, Tronco={trunk_inclination:.1f}°")

        elif lateral_bend and bend_direction:
            bend_type = f"lateral_{bend_direction}"
            bend_confidence = 0.75
            print(f"🤸 Inclinación LATERAL ({bend_direction}) detectada: Asimetría caderas={hip_asymmetry:.1f}°")

        elif subtle_bend:
            bend_type = "leve"
            bend_confidence = 0.65
            print(f"🤸 Inclinación LEVE detectada: Tronco={trunk_inclination:.1f}°")

        return bend_type, bend_confidence

    def is_bending(self, features, landmarks_coords=None):
        """
        Detecta si hay cualquier tipo de inclinación
        MEJORADO: Considera múltiples tipos de inclinación
        """
        trunk_inclination = features[4]
        right_hip_angle = features[2]
        left_hip_angle = features[3]
        avg_hip_angle = (right_hip_angle + left_hip_angle) / 2

        # Criterios más permisivos para detectar CUALQUIER inclinación
        # Inclinación por tronco (más permisivo: >12° antes 20°)
        trunk_bent = abs(trunk_inclination) > 12

        # O inclinación por caderas flexionadas
        hips_bent = avg_hip_angle < 155

        # O asimetría significativa (inclinación lateral)
        hip_asymmetry = abs(right_hip_angle - left_hip_angle) > 15

        is_bent = trunk_bent or (hips_bent and not self.is_squatting(features)) or hip_asymmetry

        return is_bent

    def validate_with_motion_detection(self, activity, confidence, frame, landmarks_coords):
        """
        NUEVO: Valida la prediccion usando deteccion de movimiento real
        Ajusta confianza segun consistencia con movimiento detectado
        """
        if not self.enable_motion_validation or frame is None:
            return activity, confidence

        try:
            # Analizar movimiento real en el frame
            motion_analysis = self.motion_detector.get_comprehensive_motion_analysis(
                frame, landmarks_coords
            )

            # Validar prediccion contra movimiento real
            validation = self.motion_detector.validate_activity_prediction(
                activity, motion_analysis
            )

            # Ajustar confianza segun validacion
            if not validation['is_valid']:
                # Reducir confianza si hay inconsistencias
                adjusted_confidence = confidence * validation['confidence_adjustment']

                print(f"⚠️ Validación de movimiento:")
                print(f"   Nivel movimiento: {motion_analysis['motion_level']}")
                print(f"   Confianza ajustada: {confidence:.2f} → {adjusted_confidence:.2f}")

                # Mostrar advertencias mas importantes
                for inc in validation['inconsistencies'][:2]:
                    print(f"   ! {inc['message']}")

                return activity, adjusted_confidence

            return activity, confidence

        except Exception as e:
            # Si falla validacion, retornar sin cambios
            print(f"⚠️ Error en validacion de movimiento: {e}")
            return activity, confidence

    def predict_activity(self, landmarks_coords, frame=None):
        """Predice actividad basada en coordenadas de landmarks usando ventanas temporales"""
        if not self.model or not landmarks_coords:
            return "Sistema inicializando...", 0.0

        try:
            # Extraer características cinemáticas del frame actual (16 características)
            features = extract_kinematic_features(landmarks_coords)

            # Verificar dimensiones
            if len(features) != 16:
                print(f"Debug: Features length = {len(features)}, expected 16")
                return "Procesando datos...", 0.0

            # FILTRO DE MOVIMIENTO ESTÁTICO - detectar si no hay movimiento real
            # MEJORADO: Solo aplicar si NO detectamos patrón de caminata
            if self.is_static(features):
                # Verificar que no sea caminata lenta
                if not self.is_walking(features):
                    return "Parado sin movimiento", 0.95
                else:
                    print("⚠️ Velocidades bajas pero patrón de caminata detectado - continuando con predicción del modelo")
            
            # Aplicar ingeniería de características para modelo optimizado
            enhanced_features = self.apply_feature_engineering(features)
            
            # Agregar features al buffer (usar las características mejoradas)
            self.feature_buffer.append(enhanced_features)
            
            # Si no tenemos suficientes frames, devolver estado de inicialización
            if len(self.feature_buffer) < self.window_size:
                frames_needed = self.window_size - len(self.feature_buffer)
                return f"Inicializando ({frames_needed} frames restantes)...", 0.0
            
            # Crear ventana de características 
            window_features = np.array(list(self.feature_buffer))  
            
            # Verificar si estamos usando el modelo optimizado
            optimized_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'optimized_har_model.pkl')
            is_using_optimized = os.path.exists(optimized_model_path)
            
            if is_using_optimized and hasattr(self, 'is_optimized_model') and self.is_optimized_model:
                # Para el modelo optimizado: usar las características mejoradas directamente de la ventana
                # El modelo optimizado usa solo las características mejoradas de un frame (no ventana temporal)
                # Tomar el último frame de la ventana que ya tiene características mejoradas (24 features)
                latest_enhanced_features = enhanced_features  # 24 features del frame actual

                X_prediction = np.array(latest_enhanced_features).reshape(1, -1)

                # Aplicar feature selection si está disponible (reduce de 24 a 20 características)
                if hasattr(self, 'feature_selector') and self.feature_selector is not None:
                    X_prediction = self.feature_selector.transform(X_prediction)
                    print(f"Debug: After feature selection - shape: {X_prediction.shape}")

                # Usar el scaler del modelo optimizado
                if hasattr(self, 'optimized_scaler') and self.optimized_scaler is not None:
                    features_scaled = self.optimized_scaler.transform(X_prediction)
                else:
                    # Normalización básica si no hay scaler optimizado
                    features_scaled = (X_prediction - np.mean(X_prediction)) / (np.std(X_prediction) + 1e-8)
            else:
                # Para el modelo original: usar método tradicional con ventana temporal
                window_flattened = window_features.flatten()
                features_scaled = self.scaler.transform([window_flattened])
            
            # Predecir con la ventana procesada
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Decodificar
            activity = self.label_encoder.inverse_transform([prediction])[0]
            confidence = np.max(probabilities)

            # Debug: mostrar top 3 predicciones
            classes = self.label_encoder.classes_
            prob_dict = dict(zip(classes, probabilities))
            sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

            print(f"Debug: Top 3 predicciones (ventana temporal):")
            for i, (class_name, prob) in enumerate(sorted_probs[:3]):
                print(f"  {i+1}. {class_name}: {prob:.3f}")

            # ===== POST-PROCESAMIENTO 1: Detectores Basados en Ángulos =====
            # Estos detectores son MUY confiables porque usan geometría real

            # Detector de sentadillas (MEJORADO)
            if self.is_squatting(features):
                # Si detectamos sentadilla física, buscar en las predicciones
                squat_found = False
                for act, prob in sorted_probs:
                    if "sentadilla" in act.lower() or "squat" in act.lower():
                        squat_found = True
                        # Forzar sentadilla si geometría es clara, incluso con baja confianza
                        if prob < 0.65:  # Aumentado de 0.5 a 0.65 para ser más agresivo
                            print(f"🔄 Corrección: Geometría indica SENTADILLA")
                            print(f"   ✅ Cambiado a: {act} (conf ajustada: 0.78)")
                            activity = act
                            confidence = 0.78
                        break

                # Si geometría dice sentadilla pero NO está en top predictions, forzar
                if not squat_found:
                    print(f"🔄 Corrección FUERTE: Geometría indica SENTADILLA pero modelo no la detectó")
                    # Buscar cualquier actividad relacionada con sentadillas
                    for act, prob in sorted_probs:
                        if "sentadilla" in act.lower() or "squat" in act.lower() or "agacharse" in act.lower():
                            activity = act
                            confidence = 0.75
                            print(f"   ✅ Forzado a: {activity} (conf: {confidence:.2f})")
                            break

            # Detector de inclinaciones (MEJORADO con tipos)
            if self.is_bending(features, landmarks_coords) and not self.is_squatting(features):
                # Detectar el TIPO específico de inclinación
                bend_type, bend_conf = self.detect_bending_type(features, landmarks_coords)

                if bend_type:
                    # Buscar actividad de inclinación que coincida con el tipo
                    bend_found = False
                    for act, prob in sorted_probs:
                        act_lower = act.lower()

                        # Matching mejorado basado en tipo de inclinación
                        if bend_type == "frontal" and ("adelante" in act_lower or "bend_forward" in act_lower or "inclin" in act_lower):
                            bend_found = True
                            if prob < 0.65:
                                print(f"🔄 Corrección: Geometría indica INCLINACIÓN FRONTAL")
                                print(f"   ✅ Cambiado a: {act} (conf ajustada: {bend_conf:.2f})")
                                activity = act
                                confidence = bend_conf
                            break

                        elif bend_type.startswith("lateral_"):
                            direction = bend_type.split("_")[1]
                            if direction in act_lower or "inclin" in act_lower:
                                bend_found = True
                                if prob < 0.65:
                                    print(f"🔄 Corrección: Geometría indica INCLINACIÓN LATERAL ({direction})")
                                    print(f"   ✅ Cambiado a: {act} (conf ajustada: {bend_conf:.2f})")
                                    activity = act
                                    confidence = bend_conf
                                break

                        elif bend_type == "leve" and "inclin" in act_lower:
                            bend_found = True
                            if prob < 0.60:
                                print(f"🔄 Corrección: Geometría indica INCLINACIÓN LEVE")
                                print(f"   ✅ Cambiado a: {act} (conf ajustada: {bend_conf:.2f})")
                                activity = act
                                confidence = bend_conf
                            break

            # ===== POST-PROCESAMIENTO 2: Corregir acercándose vs alejándose =====
            # Usar cambio de escala corporal para diferenciar dirección
            from ..utils.kinematic_features import kinematic_extractor

            scale_direction = kinematic_extractor.get_scale_change_direction()

            # Si el modelo predice "Caminar acercándose" o "Caminar alejándose",
            # verificar con el cambio de escala real
            if "acercandose" in activity.lower() or "approach" in activity.lower():
                if scale_direction == 'moving_away':
                    print(f"🔄 Corrección: Cambio de escala indica ALEJÁNDOSE (no acercándose)")
                    # Buscar predicción de "alejándose" en las probabilidades
                    for act, prob in sorted_probs:
                        if "alejandose" in act.lower() or "walk_away" in act.lower() or "espaldas" in act.lower():
                            activity = act
                            confidence = prob
                            print(f"   ✅ Cambiado a: {activity} (conf: {confidence:.3f})")
                            break
                else:
                    print(f"✅ Cambio de escala confirma ACERCÁNDOSE")

            elif "alejandose" in activity.lower() or "walk_away" in activity.lower() or "espaldas" in activity.lower():
                if scale_direction == 'approaching':
                    print(f"🔄 Corrección: Cambio de escala indica ACERCÁNDOSE (no alejándose)")
                    # Buscar predicción de "acercándose" en las probabilidades
                    for act, prob in sorted_probs:
                        if "acercandose" in act.lower() or "approach" in act.lower():
                            activity = act
                            confidence = prob
                            print(f"   ✅ Cambiado a: {activity} (conf: {confidence:.3f})")
                            break
                else:
                    print(f"✅ Cambio de escala confirma ALEJÁNDOSE")
            
            # Análisis de certeza
            top_prob = sorted_probs[0][1]
            second_prob = sorted_probs[1][1]
            prob_gap = top_prob - second_prob
            
            # Filtros de calidad
            if prob_gap < 0.03 and top_prob < 0.5:
                print(f"⚠️  Predicción incierta - Gap pequeño: {prob_gap:.3f}")
                return "Analizando secuencia...", top_prob
            
            # Filtro anti-sesgo para clases problemáticas
            problematic_classes = ["Caminar alejandose (espaldas)"]
            if activity in problematic_classes and confidence < 0.4:
                print(f"⚠️  Clase problemática '{activity}' con baja confianza - Usando segunda opción")
                if len(sorted_probs) > 1:
                    activity = sorted_probs[1][0]
                    confidence = sorted_probs[1][1]
            
            # Suavizado temporal mejorado
            self.recent_predictions.append((activity, confidence))
            if len(self.recent_predictions) > self.max_history:
                self.recent_predictions.pop(0)
            
            # Consistencia temporal más estricta
            if len(self.recent_predictions) >= 3:
                recent_activities = [pred[0] for pred in self.recent_predictions[-3:]]
                activity_counts = {}
                for act in recent_activities:
                    activity_counts[act] = activity_counts.get(act, 0) + 1
                
                # Si la actividad actual es completamente nueva y la confianza no es alta
                if activity_counts.get(activity, 0) == 1 and confidence < 0.6:
                    # Buscar la actividad más consistente
                    most_consistent = max(activity_counts.items(), key=lambda x: x[1])
                    if most_consistent[1] > 1:
                        print(f"📊 Suavizado temporal: {activity} -> {most_consistent[0]} (consistencia)")
                        activity = most_consistent[0]
            
            # Umbral mínimo de confianza
            if confidence < 0.2:
                return "Detectando patrón...", confidence
            
            # Añadir indicador de modelo optimizado
            if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'optimized_har_model.pkl')):
                activity += " 🚀"  # Indicador de modelo optimizado

            # === VALIDACION FINAL CON DETECCION DE MOVIMIENTO ===
            # Valida que la prediccion sea consistente con el movimiento real detectado
            if self.enable_motion_validation and frame is not None:
                activity, confidence = self.validate_with_motion_detection(
                    activity, confidence, frame, landmarks_coords
                )

            return activity, confidence
            
        except Exception as e:
            print(f"Error en predicción: {e}")
            import traceback
            traceback.print_exc()
            return "Error de procesamiento...", 0.0
