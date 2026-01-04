import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
from PIL import Image
import logging
from pathlib import Path
import pickle
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import time

class MLSystem:
    def __init__(self):
        """Initialize the machine learning system"""
        self.logger = logging.getLogger(__name__)
        
        # Initialize paths
        self.echo_dir = Path.home() / '.deep_tree_echo'
        self.ml_dir = self.echo_dir / 'ml'
        self.ml_dir.mkdir(parents=True, exist_ok=True)
        self.activity_file = self.ml_dir / 'activity.json'
        self.activities = []
        self._load_activities()
        
        # Initialize models directory
        self.models_dir = Path.home() / '.deep_tree_echo' / 'models'
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize component models
        self.visual_model = None
        self.behavior_model = None
        self.pattern_model = None
        self.echo_value_model = None
        self.reflex_model = None  # New: For lightning-fast gaming reflexes
        self.tactical_model = None  # New: For strategic decision-making
        
        # Initialize interaction history
        self.interaction_history = []
        
        # Gaming-specific performance tracking
        self.reflex_training_data = []
        self.tactical_decisions = []
        
        # Load existing models if available
        self._load_models()
        
    def _load_models(self):
        """Load pre-trained models"""
        try:
            # Load or create visual model
            visual_path = self.models_dir / 'visual_model'
            if visual_path.exists():
                self.visual_model = models.load_model(visual_path)
            else:
                self.visual_model = self._create_visual_model()
                
            # Load or create behavior model
            behavior_path = self.models_dir / 'behavior_model'
            if behavior_path.exists():
                self.behavior_model = models.load_model(behavior_path)
            else:
                self.behavior_model = self._create_behavior_model()
                
            # Load or create pattern model
            pattern_path = self.models_dir / 'pattern_model'
            if pattern_path.exists():
                self.pattern_model = models.load_model(pattern_path)
            else:
                self.pattern_model = self._create_pattern_model()
                
            # Load or create echo value model
            echo_value_path = self.models_dir / 'echo_value_model'
            if echo_value_path.exists():
                self.echo_value_model = models.load_model(echo_value_path)
            else:
                self.echo_value_model = self._create_echo_value_model()
            
            # Load or create reflex model (for gaming)
            reflex_path = self.models_dir / 'reflex_model'
            if reflex_path.exists():
                self.reflex_model = models.load_model(reflex_path)
            else:
                self.reflex_model = self._create_reflex_model()
            
            # Load or create tactical model (for gaming strategy)
            tactical_path = self.models_dir / 'tactical_model'
            if tactical_path.exists():
                self.tactical_model = models.load_model(tactical_path)
            else:
                self.tactical_model = self._create_tactical_model()
                
            self.logger.info("Successfully loaded ML models including gaming models")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            
    def _load_activities(self):
        """Load existing activities"""
        if self.activity_file.exists():
            try:
                with open(self.activity_file) as f:
                    self.activities = json.load(f)
            except:
                self.activities = []
                
    def _save_activities(self):
        """Save activities to file"""
        with open(self.activity_file, 'w') as f:
            json.dump(self.activities[-1000:], f)
            
    def _log_activity(self, description: str, data: Optional[Dict] = None):
        """Log an ML activity"""
        activity = {
            'time': time.time(),
            'description': description,
            'data': data or {}
        }
        self.activities.append(activity)
        self._save_activities()
        
    def _create_visual_model(self):
        """Create visual recognition model"""
        model = models.Sequential([
            layers.Input(shape=(224, 224, 3)),
            layers.Conv2D(64, 3, activation='relu', padding='same'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation='relu', padding='same'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation='relu', padding='same'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(4)  # x, y, width, height
        ])
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        return model
        
    def _create_behavior_model(self):
        """Create behavior learning model"""
        model = models.Sequential([
            layers.Input(shape=(4,)),  # start_x, start_y, end_x, end_y
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(8)  # 8 control points for movement
        ])
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        return model
        
    def _create_pattern_model(self):
        """Create pattern recognition model"""
        model = models.Sequential([
            layers.Input(shape=(100,)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(8)
        ])
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        return model
        
    def _create_echo_value_model(self):
        """Create echo value prediction model"""
        model = models.Sequential([
            layers.Input(shape=(6,)),  # features: content length, complexity, child echoes, node depth, sibling echoes, historical echo
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)  # predicted echo value
        ])
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        return model
        
    def detect_element(self, screenshot: np.ndarray, template: np.ndarray,
                      threshold: float = 0.8) -> Optional[Dict]:
        """Detect UI element using visual model and template matching"""
        try:
            # Check input shapes
            if screenshot is None or template is None:
                self.logger.error("Invalid input: screenshot or template is None")
                return None
                
            # Ensure screenshot is in RGB format
            if len(screenshot.shape) == 2:
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_GRAY2RGB)
            elif screenshot.shape[2] == 4:
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2RGB)
            elif screenshot.shape[2] == 3 and screenshot.dtype == np.uint8:
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
                
            # Resize screenshot to model input size
            resized = cv2.resize(screenshot, (224, 224))
            
            # Normalize pixel values
            normalized = resized.astype(np.float32) / 255.0
            
            # Ensure correct input shape
            model_input = np.expand_dims(normalized, axis=0)
            
            # Get model prediction
            prediction = self.visual_model.predict(model_input, verbose=0)
            
            # Convert prediction to bounding box
            x, y, w, h = prediction[0]
            x = int(x * screenshot.shape[1])
            y = int(y * screenshot.shape[0])
            w = int(w * screenshot.shape[1])
            h = int(h * screenshot.shape[0])
            
            # Use template matching as fallback
            if w <= 0 or h <= 0 or x < 0 or y < 0:
                result = cv2.matchTemplate(
                    screenshot,
                    template,
                    cv2.TM_CCOEFF_NORMED
                )
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val >= threshold:
                    return {
                        'confidence': float(max_val),
                        'location': max_loc,
                        'size': template.shape[:2]
                    }
            else:
                return {
                    'confidence': 1.0,
                    'location': (x, y),
                    'size': (w, h)
                }
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting element: {str(e)}")
            return None
            
    def optimize_movement(self, start_pos: Tuple[int, int],
                         end_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Optimize mouse movement path"""
        try:
            # Prepare input features
            features = np.array([
                start_pos[0], start_pos[1],
                end_pos[0], end_pos[1]
            ]).reshape(1, -1)
            
            # Get model prediction
            control_points = self.behavior_model.predict(features, verbose=0)[0]
            
            # Generate path points
            distance = np.linalg.norm(
                np.array(end_pos) - np.array(start_pos)
            )
            num_points = max(int(distance / 10), 5)
            
            points = []
            for i in range(num_points):
                t = i / (num_points - 1)
                
                # Add learned variation to path
                variation_x = control_points[i % 4] * 0.1
                variation_y = control_points[(i + 4) % 8] * 0.1
                
                # Calculate point with natural curve and learned variation
                x = int(start_pos[0] + (end_pos[0] - start_pos[0]) * t +
                       np.sin(t * np.pi) * variation_x * distance)
                y = int(start_pos[1] + (end_pos[1] - start_pos[1]) * t +
                       np.sin(t * np.pi) * variation_y * distance)
                
                points.append((x, y))
                
            return points
            
        except Exception as e:
            self.logger.error(f"Error optimizing movement: {str(e)}")
            
            # Return fallback linear path
            num_points = max(int(np.linalg.norm(
                np.array(end_pos) - np.array(start_pos)
            ) / 10), 5)
            
            points = []
            for i in range(num_points):
                t = i / (num_points - 1)
                x = int(start_pos[0] + (end_pos[0] - start_pos[0]) * t)
                y = int(start_pos[1] + (end_pos[1] - start_pos[1]) * t)
                points.append((x, y))
                
            return points
            
    def learn_from_interaction(self, interaction_type: str,
                             start_state: Dict,
                             end_state: Dict,
                             success: bool):
        """Learn from interaction outcomes"""
        try:
            # Record interaction
            interaction = {
                'type': interaction_type,
                'start_state': start_state,
                'end_state': end_state,
                'success': success,
                'timestamp': datetime.now().isoformat()
            }
            self.interaction_history.append(interaction)
            
            # Save history periodically
            if len(self.interaction_history) % 100 == 0:
                self._save_interaction_history()
                
            # Update behavior model if enough data
            if len(self.interaction_history) >= 1000:
                self._update_behavior_model()
                
        except Exception as e:
            self.logger.error(f"Error learning from interaction: {str(e)}")
            
    def _save_interaction_history(self):
        """Save interaction history to disk"""
        try:
            history_path = self.models_dir / 'interaction_history.json'
            with open(history_path, 'w') as f:
                json.dump(self.interaction_history[-1000:], f)
                
        except Exception as e:
            self.logger.error(f"Error saving interaction history: {str(e)}")
            
    def _update_behavior_model(self):
        """Update behavior model based on interaction history"""
        try:
            # Prepare training data
            successful_interactions = [
                i for i in self.interaction_history
                if i['success'] and i['type'] == 'mouse_movement'
            ]
            
            if len(successful_interactions) < 100:
                return
                
            # Create training batches
            X = []
            y = []
            
            for interaction in successful_interactions[-1000:]:
                start = interaction['start_state'].get('position')
                end = interaction['end_state'].get('position')
                path = interaction['end_state'].get('path', [])
                
                if start and end and path:
                    # Extract movement patterns
                    X.append([
                        start[0], start[1],
                        end[0], end[1]
                    ])
                    
                    # Use path points as control points
                    control_points = []
                    for i in range(0, len(path), len(path)//8):
                        if len(control_points) < 8:
                            point = path[i]
                            control_points.append(
                                (point[0] - start[0]) / 10
                            )
                            control_points.append(
                                (point[1] - start[1]) / 10
                            )
                    y.append(control_points[:8])
                    
            if not X:
                return
                
            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)
            
            # Train model
            self.behavior_model.fit(
                X, y,
                epochs=10,
                batch_size=32,
                verbose=0
            )
            
            # Save updated model
            self.behavior_model.save(self.models_dir / 'behavior_model')
            
            self.logger.info("Successfully updated behavior model")
            
        except Exception as e:
            self.logger.error(f"Error updating behavior model: {str(e)}")
            
    def analyze_patterns(self, interactions: List[Dict]) -> Dict:
        """Analyze interaction patterns"""
        try:
            patterns = {
                'timing': {},
                'movement': {},
                'success_rate': {}
            }
            
            # Analyze timing patterns
            timestamps = [
                datetime.fromisoformat(i['timestamp'])
                for i in interactions
            ]
            if len(timestamps) > 1:
                intervals = [
                    (timestamps[i+1] - timestamps[i]).total_seconds()
                    for i in range(len(timestamps)-1)
                ]
                patterns['timing']['mean_interval'] = np.mean(intervals)
                patterns['timing']['std_interval'] = np.std(intervals)
                
            # Analyze movement patterns
            movements = [
                i for i in interactions
                if i['type'] == 'mouse_movement'
            ]
            if movements:
                distances = []
                speeds = []
                for m in movements:
                    start = m['start_state'].get('position')
                    end = m['end_state'].get('position')
                    if start and end:
                        distance = np.sqrt(
                            (end[0] - start[0])**2 +
                            (end[1] - start[1])**2
                        )
                        duration = float(m['end_state'].get('duration', 1.0))
                        distances.append(distance)
                        speeds.append(distance / duration)
                        
                patterns['movement']['mean_distance'] = np.mean(distances)
                patterns['movement']['mean_speed'] = np.mean(speeds)
                
            # Analyze success rates
            total = len(interactions)
            successful = len([i for i in interactions if i['success']])
            patterns['success_rate']['overall'] = successful / total
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error analyzing patterns: {str(e)}")
            return {}
            
    async def update_models(self):
        """Update ML models"""
        self._log_activity("Starting model update")
        try:
            # Update visual model
            self._log_activity("Updating visual model")
            await self.visual_model.update()
            
            # Update behavior model
            self._log_activity("Updating behavior model")
            await self.behavior_model.update()
            
            # Update pattern model
            self._log_activity("Updating pattern model")
            await self.pattern_model.update()
            
            # Update echo value model
            self._log_activity("Updating echo value model")
            await self.echo_value_model.update()
            
            self._log_activity("Model update complete")
            
        except Exception as e:
            self._log_activity("Error updating models", {'error': str(e)})
            raise
            
    def predict_echo_value(self, features: List[float]) -> float:
        """Predict echo value using the echo value model"""
        try:
            features = np.array(features).reshape(1, -1)
            prediction = self.echo_value_model.predict(features, verbose=0)
            return float(prediction[0][0])
        except Exception as e:
            self.logger.error(f"Error predicting echo value: {str(e)}")
            return 0.0
    
    def _create_reflex_model(self):
        """Create reflex optimization model for lightning-fast gaming responses"""
        model = models.Sequential([
            layers.Input(shape=(8,)),  # input: target_x, target_y, current_x, current_y, velocity_x, velocity_y, threat_level, urgency
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.1),  # Minimal dropout for speed
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(4)  # output: optimal_x, optimal_y, confidence, reaction_time
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def _create_tactical_model(self):
        """Create tactical decision-making model for strategic mastery"""
        model = models.Sequential([
            layers.Input(shape=(16,)),  # Complex situation features
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(8)  # Action scores for different tactical options
        ])
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def predict_reflex_response(self, target_position: Tuple[float, float],
                               current_position: Tuple[float, float],
                               velocity: Tuple[float, float] = (0, 0),
                               threat_level: float = 0.5,
                               urgency: float = 0.8) -> Dict[str, Any]:
        """
        Predict optimal reflex response for gaming scenarios
        
        Args:
            target_position: Target coordinates
            current_position: Current position
            velocity: Current velocity
            threat_level: How threatening the situation is (0-1)
            urgency: How urgent the response needs to be (0-1)
            
        Returns:
            Dict with optimal response and timing
        """
        try:
            # Prepare input features
            features = np.array([
                target_position[0], target_position[1],
                current_position[0], current_position[1],
                velocity[0], velocity[1],
                threat_level, urgency
            ]).reshape(1, -1)
            
            # Get prediction
            prediction = self.reflex_model.predict(features, verbose=0)[0]
            
            result = {
                'optimal_position': (prediction[0], prediction[1]),
                'confidence': float(prediction[2]),
                'estimated_reaction_time': float(prediction[3]),
                'should_execute': prediction[2] > 0.6  # Execute if confidence > 60%
            }
            
            # Record for training
            self.reflex_training_data.append({
                'input': features.tolist(),
                'output': prediction.tolist(),
                'timestamp': time.time()
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error predicting reflex response: {str(e)}")
            return {
                'optimal_position': target_position,
                'confidence': 0.0,
                'estimated_reaction_time': 0.2,
                'should_execute': False
            }
    
    def predict_tactical_action(self, situation_features: np.ndarray) -> Dict[str, Any]:
        """
        Predict best tactical action for strategic gameplay
        
        Args:
            situation_features: Array of 16 features describing the tactical situation
            
        Returns:
            Dict with recommended action and confidence
        """
        try:
            if len(situation_features) != 16:
                self.logger.error(f"Expected 16 features, got {len(situation_features)}")
                return {'action_index': 0, 'confidence': 0.0, 'action_scores': []}
            
            features = situation_features.reshape(1, -1)
            
            # Get prediction
            prediction = self.tactical_model.predict(features, verbose=0)[0]
            
            # Find best action
            best_action_idx = int(np.argmax(prediction))
            
            result = {
                'action_index': best_action_idx,
                'confidence': float(prediction[best_action_idx]),
                'action_scores': prediction.tolist(),
                'top_3_actions': [int(i) for i in np.argsort(prediction)[-3:][::-1]]
            }
            
            # Record for training
            self.tactical_decisions.append({
                'features': features.tolist(),
                'prediction': prediction.tolist(),
                'timestamp': time.time()
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error predicting tactical action: {str(e)}")
            return {
                'action_index': 0,
                'confidence': 0.0,
                'action_scores': []
            }
    
    def train_reflex_model(self, training_data: List[Dict[str, Any]]) -> bool:
        """
        Train reflex model with gaming performance data
        
        Args:
            training_data: List of training examples with inputs and outcomes
            
        Returns:
            Success status
        """
        try:
            if len(training_data) < 100:
                self.logger.warning("Insufficient training data for reflex model")
                return False
            
            X = []
            y = []
            
            for example in training_data:
                X.append(example['input'])
                y.append(example['output'])
            
            X = np.array(X)
            y = np.array(y)
            
            # Train with emphasis on speed (fewer epochs but optimized)
            self.reflex_model.fit(
                X, y,
                epochs=5,  # Fast training for quick adaptation
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            # Save updated model
            self.reflex_model.save(self.models_dir / 'reflex_model')
            
            self.logger.info("Successfully trained reflex model")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training reflex model: {str(e)}")
            return False
    
    def train_tactical_model(self, training_data: List[Dict[str, Any]]) -> bool:
        """
        Train tactical model with strategic decision data
        
        Args:
            training_data: List of tactical decisions with outcomes
            
        Returns:
            Success status
        """
        try:
            if len(training_data) < 50:
                self.logger.warning("Insufficient training data for tactical model")
                return False
            
            X = []
            y = []
            
            for example in training_data:
                X.append(example['features'])
                # Use successful action as label
                action_label = np.zeros(8)
                action_label[example['successful_action']] = 1
                y.append(action_label)
            
            X = np.array(X)
            y = np.array(y)
            
            # Train with more epochs for strategic learning
            self.tactical_model.fit(
                X, y,
                epochs=20,
                batch_size=16,
                validation_split=0.2,
                verbose=0
            )
            
            # Save updated model
            self.tactical_model.save(self.models_dir / 'tactical_model')
            
            self.logger.info("Successfully trained tactical model")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training tactical model: {str(e)}")
            return False
    
    def optimize_for_performance(self) -> Dict[str, Any]:
        """
        Optimize ML models for gaming performance (low latency)
        
        Returns:
            Optimization results
        """
        optimizations = {
            'reflex_model_optimized': False,
            'tactical_model_optimized': False,
            'average_inference_time_ms': 0.0,
            'recommendations': []
        }
        
        try:
            # Test inference times
            test_input_reflex = np.random.random((1, 8))
            test_input_tactical = np.random.random((1, 16))
            
            # Time reflex model
            start = time.time()
            for _ in range(100):
                self.reflex_model.predict(test_input_reflex, verbose=0)
            reflex_time = (time.time() - start) / 100
            
            # Time tactical model
            start = time.time()
            for _ in range(100):
                self.tactical_model.predict(test_input_tactical, verbose=0)
            tactical_time = (time.time() - start) / 100
            
            avg_time = (reflex_time + tactical_time) / 2
            optimizations['average_inference_time_ms'] = avg_time * 1000
            
            # Check if optimization is needed
            if reflex_time > 0.01:  # More than 10ms is too slow for gaming
                optimizations['recommendations'].append(
                    "Reflex model too slow - consider model quantization"
                )
            else:
                optimizations['reflex_model_optimized'] = True
            
            if tactical_time > 0.05:  # Tactical can be slightly slower
                optimizations['recommendations'].append(
                    "Tactical model could be optimized further"
                )
            else:
                optimizations['tactical_model_optimized'] = True
            
            self.logger.info(f"Performance check: {avg_time*1000:.2f}ms average inference")
            
        except Exception as e:
            self.logger.error(f"Error optimizing for performance: {str(e)}")
        
        return optimizations

