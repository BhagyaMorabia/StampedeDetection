"""
Adaptive Threshold Optimizer for STAMPede Detection System
Uses machine learning to dynamically adjust detection thresholds based on environmental conditions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
import json
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
from collections import deque
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ThresholdConfig:
    """Configuration for threshold optimization"""
    min_confidence: float = 0.05
    max_confidence: float = 0.5
    default_confidence: float = 0.15
    learning_rate: float = 0.01
    update_frequency: int = 100  # Update every 100 detections
    history_size: int = 1000  # Keep last 1000 records
    min_samples: int = 50  # Minimum samples needed for training

@dataclass
class EnvironmentalFactors:
    """Environmental factors affecting detection accuracy"""
    lighting_condition: float = 0.5  # 0-1 scale
    weather_condition: float = 0.5  # 0-1 scale (0=clear, 1=storm)
    time_of_day: float = 0.5  # 0-1 scale (0=night, 1=day)
    crowd_density: float = 0.0  # Current crowd density
    camera_angle: float = 0.5  # 0-1 scale (0=side, 1=overhead)
    image_quality: float = 0.8  # 0-1 scale
    motion_blur: float = 0.0  # 0-1 scale
    occlusion_level: float = 0.0  # 0-1 scale

@dataclass
class DetectionRecord:
    """Record of detection with environmental context"""
    timestamp: float
    confidence_threshold: float
    people_detected: int
    true_people_count: Optional[int] = None  # Ground truth if available
    false_positives: int = 0
    false_negatives: int = 0
    environmental_factors: EnvironmentalFactors = None
    accuracy_score: float = 0.0

class AdaptiveThresholdOptimizer:
    """ML-based adaptive threshold optimization system"""
    
    def __init__(self, config: Optional[ThresholdConfig] = None):
        self.config = config or ThresholdConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.detection_history = deque(maxlen=self.config.history_size)
        self.performance_history = deque(maxlen=100)
        self.current_threshold = self.config.default_confidence
        
        # Model performance tracking
        self.model_accuracy = 0.0
        self.last_update_time = time.time()
        self.update_count = 0
        
        # Create model directory
        os.makedirs("models", exist_ok=True)
        
    def extract_features(self, environmental_factors: EnvironmentalFactors, 
                        detection_context: Dict[str, Any]) -> np.ndarray:
        """Extract features for ML model"""
        features = [
            environmental_factors.lighting_condition,
            environmental_factors.weather_condition,
            environmental_factors.time_of_day,
            environmental_factors.crowd_density,
            environmental_factors.camera_angle,
            environmental_factors.image_quality,
            environmental_factors.motion_blur,
            environmental_factors.occlusion_level,
            detection_context.get('frame_resolution', 1280),
            detection_context.get('fps', 30),
            detection_context.get('processing_time', 0.033),
            detection_context.get('gpu_memory_usage', 0.5),
            detection_context.get('temperature', 25.0),
            detection_context.get('humidity', 50.0),
            detection_context.get('wind_speed', 0.0),
            detection_context.get('event_type', 0),  # 0=normal, 1=concert, 2=sports, etc.
            detection_context.get('venue_capacity', 1000),
            detection_context.get('current_capacity_ratio', 0.1),
            detection_context.get('hour_of_day', 12),
            detection_context.get('day_of_week', 1),
            detection_context.get('month', 1),
            detection_context.get('is_holiday', 0),
            detection_context.get('is_weekend', 0),
        ]
        
        return np.array(features, dtype=np.float32)
    
    def calculate_optimal_threshold(self, environmental_factors: EnvironmentalFactors,
                                 detection_context: Dict[str, Any]) -> float:
        """Calculate optimal threshold using ML model"""
        
        if not self.is_trained or len(self.detection_history) < self.config.min_samples:
            return self.current_threshold
        
        try:
            # Extract features
            features = self.extract_features(environmental_factors, detection_context)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Predict optimal threshold
            predicted_threshold = self.model.predict(features_scaled)[0]
            
            # Clamp to valid range
            optimal_threshold = np.clip(predicted_threshold, 
                                      self.config.min_confidence, 
                                      self.config.max_confidence)
            
            # Smooth threshold changes to avoid sudden jumps
            threshold_diff = optimal_threshold - self.current_threshold
            if abs(threshold_diff) > 0.05:  # Limit change to 0.05 per update
                optimal_threshold = self.current_threshold + np.sign(threshold_diff) * 0.05
            
            return float(optimal_threshold)
            
        except Exception as e:
            print(f"⚠️ Threshold optimization error: {e}")
            return self.current_threshold
    
    def update_model(self, detection_record: DetectionRecord):
        """Update the ML model with new detection data"""
        self.detection_history.append(detection_record)
        self.update_count += 1
        
        # Check if we need to retrain
        if (self.update_count % self.config.update_frequency == 0 and 
            len(self.detection_history) >= self.config.min_samples):
            self._retrain_model()
    
    def _retrain_model(self):
        """Retrain the ML model with current data"""
        try:
            if len(self.detection_history) < self.config.min_samples:
                return
            
            # Prepare training data
            X = []
            y = []
            
            for record in self.detection_history:
                if record.environmental_factors is not None:
                    # Create detection context from record
                    detection_context = {
                        'frame_resolution': 1280,
                        'fps': 30,
                        'processing_time': 0.033,
                        'gpu_memory_usage': 0.5,
                        'temperature': 25.0,
                        'humidity': 50.0,
                        'wind_speed': 0.0,
                        'event_type': 0,
                        'venue_capacity': 1000,
                        'current_capacity_ratio': record.environmental_factors.crowd_density,
                        'hour_of_day': datetime.fromtimestamp(record.timestamp).hour,
                        'day_of_week': datetime.fromtimestamp(record.timestamp).weekday(),
                        'month': datetime.fromtimestamp(record.timestamp).month,
                        'is_holiday': 0,
                        'is_weekend': 1 if datetime.fromtimestamp(record.timestamp).weekday() >= 5 else 0,
                    }
                    
                    features = self.extract_features(record.environmental_factors, detection_context)
                    X.append(features)
                    
                    # Calculate target threshold based on accuracy
                    if record.accuracy_score > 0.8:
                        # High accuracy - can use lower threshold
                        target_threshold = max(record.confidence_threshold - 0.02, self.config.min_confidence)
                    elif record.accuracy_score < 0.6:
                        # Low accuracy - need higher threshold
                        target_threshold = min(record.confidence_threshold + 0.02, self.config.max_confidence)
                    else:
                        # Medium accuracy - keep current threshold
                        target_threshold = record.confidence_threshold
                    
                    y.append(target_threshold)
            
            if len(X) < self.config.min_samples:
                return
            
            X = np.array(X)
            y = np.array(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Try multiple models and select best
            models = {
                'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42),
                'GradientBoosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
                'Ridge': Ridge(alpha=1.0),
                'LinearRegression': LinearRegression()
            }
            
            best_model = None
            best_score = -np.inf
            
            for name, model in models.items():
                try:
                    model.fit(X_train_scaled, y_train)
                    score = model.score(X_test_scaled, y_test)
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
                        
                except Exception as e:
                    print(f"⚠️ Model {name} training failed: {e}")
                    continue
            
            if best_model is not None and best_score > 0.3:  # Minimum acceptable score
                self.model = best_model
                self.is_trained = True
                self.model_accuracy = best_score
                
                # Save model
                model_path = "models/adaptive_threshold_model.pkl"
                joblib.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'accuracy': self.model_accuracy,
                    'timestamp': time.time()
                }, model_path)
                
                print(f"✅ Threshold model retrained - Accuracy: {self.model_accuracy:.3f}")
            
        except Exception as e:
            print(f"⚠️ Model retraining failed: {e}")
    
    def load_model(self, model_path: str = "models/adaptive_threshold_model.pkl"):
        """Load pre-trained model"""
        try:
            if os.path.exists(model_path):
                model_data = joblib.load(model_path)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.model_accuracy = model_data['accuracy']
                self.is_trained = True
                print(f"✅ Loaded threshold model - Accuracy: {self.model_accuracy:.3f}")
                return True
        except Exception as e:
            print(f"⚠️ Failed to load model: {e}")
        return False
    
    def get_current_threshold(self) -> float:
        """Get current optimized threshold"""
        return self.current_threshold
    
    def update_threshold(self, new_threshold: float):
        """Update current threshold"""
        self.current_threshold = np.clip(new_threshold, 
                                       self.config.min_confidence, 
                                       self.config.max_confidence)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'is_trained': self.is_trained,
            'model_accuracy': self.model_accuracy,
            'current_threshold': self.current_threshold,
            'detection_history_size': len(self.detection_history),
            'update_count': self.update_count,
            'last_update_time': self.last_update_time,
            'config': {
                'min_confidence': self.config.min_confidence,
                'max_confidence': self.config.max_confidence,
                'default_confidence': self.config.default_confidence,
                'update_frequency': self.config.update_frequency,
                'history_size': self.config.history_size,
                'min_samples': self.config.min_samples
            }
        }
    
    def simulate_environmental_conditions(self) -> EnvironmentalFactors:
        """Simulate environmental conditions for testing"""
        current_time = datetime.now()
        
        # Simulate lighting based on time of day
        hour = current_time.hour
        if 6 <= hour <= 18:
            lighting = 0.8 + 0.2 * np.sin((hour - 6) * np.pi / 12)
        else:
            lighting = 0.2 + 0.1 * np.sin((hour - 18) * np.pi / 6)
        
        # Simulate weather (simplified)
        weather = 0.3 + 0.4 * np.random.random()
        
        # Time of day
        time_of_day = hour / 24.0
        
        return EnvironmentalFactors(
            lighting_condition=lighting,
            weather_condition=weather,
            time_of_day=time_of_day,
            crowd_density=np.random.random() * 0.5,
            camera_angle=0.7 + 0.3 * np.random.random(),
            image_quality=0.6 + 0.4 * np.random.random(),
            motion_blur=np.random.random() * 0.3,
            occlusion_level=np.random.random() * 0.2
        )

# Example usage and testing
if __name__ == "__main__":
    # Initialize optimizer
    optimizer = AdaptiveThresholdOptimizer()
    
    # Load existing model if available
    optimizer.load_model()
    
    # Simulate some training data
    print("🧪 Simulating training data...")
    for i in range(100):
        # Simulate environmental conditions
        env_factors = optimizer.simulate_environmental_conditions()
        
        # Simulate detection context
        detection_context = {
            'frame_resolution': 1280,
            'fps': 30,
            'processing_time': 0.033,
            'gpu_memory_usage': 0.5,
            'temperature': 20 + 10 * np.random.random(),
            'humidity': 40 + 20 * np.random.random(),
            'wind_speed': np.random.random() * 5,
            'event_type': np.random.randint(0, 3),
            'venue_capacity': 1000,
            'current_capacity_ratio': env_factors.crowd_density,
            'hour_of_day': datetime.now().hour,
            'day_of_week': datetime.now().weekday(),
            'month': datetime.now().month,
            'is_holiday': 0,
            'is_weekend': 1 if datetime.now().weekday() >= 5 else 0,
        }
        
        # Calculate optimal threshold
        optimal_threshold = optimizer.calculate_optimal_threshold(env_factors, detection_context)
        
        # Simulate detection record
        record = DetectionRecord(
            timestamp=time.time(),
            confidence_threshold=optimal_threshold,
            people_detected=np.random.randint(10, 50),
            environmental_factors=env_factors,
            accuracy_score=0.7 + 0.3 * np.random.random()
        )
        
        # Update model
        optimizer.update_model(record)
        optimizer.update_threshold(optimal_threshold)
        
        if i % 20 == 0:
            print(f"📊 Iteration {i}: Threshold={optimal_threshold:.3f}, Accuracy={record.accuracy_score:.3f}")
    
    # Get final performance stats
    stats = optimizer.get_performance_stats()
    print(f"\n📈 Final Performance Stats:")
    print(f"   Model Trained: {stats['is_trained']}")
    print(f"   Model Accuracy: {stats['model_accuracy']:.3f}")
    print(f"   Current Threshold: {stats['current_threshold']:.3f}")
    print(f"   Detection History: {stats['detection_history_size']} records")
    print(f"   Update Count: {stats['update_count']}")
