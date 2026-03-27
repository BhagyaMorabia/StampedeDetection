"""
Behavior Analysis & Movement Classification System for STAMPede Detection
Classifies crowd movements and detects panic situations using computer vision and ML
"""

import numpy as np
import pandas as pd
import cv2
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
import json
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
from collections import deque
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MovementVector:
    """Represents a movement vector"""
    x: float
    y: float
    magnitude: float
    direction: float  # radians
    timestamp: float

@dataclass
class BehaviorPattern:
    """Represents a crowd behavior pattern"""
    timestamp: float
    people_count: int
    movement_vectors: List[MovementVector]
    average_speed: float
    speed_variance: float
    direction_consistency: float
    acceleration_pattern: float
    clustering_level: float
    dispersion_level: float
    panic_indicators: Dict[str, float]

@dataclass
class BehaviorClassification:
    """Result of behavior classification"""
    timestamp: float
    behavior_type: str
    confidence: float
    panic_score: float
    risk_level: str
    description: str
    recommended_action: str
    movement_characteristics: Dict[str, float]

class MovementBehaviorAnalyzer:
    """Analyzes crowd movement patterns and classifies behavior"""
    
    def __init__(self):
        self.behavior_classifier = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
        # Movement tracking
        self.previous_positions = {}
        self.movement_history = deque(maxlen=100)
        self.behavior_history = deque(maxlen=500)
        
        # Behavior categories
        self.behavior_types = [
            'normal_walking',
            'crowded_walking', 
            'running',
            'panic_running',
            'stationary',
            'scattered_movement',
            'organized_flow',
            'chaotic_movement',
            'evacuation_pattern',
            'gathering_pattern'
        ]
        
        # Panic indicators
        self.panic_indicators = {
            'high_speed': 0.0,
            'direction_change': 0.0,
            'acceleration_spike': 0.0,
            'clustering_breakdown': 0.0,
            'dispersion_increase': 0.0,
            'movement_irregularity': 0.0
        }
        
        # Performance tracking
        self.classification_accuracy = 0.0
        self.panic_detection_accuracy = 0.0
        
        # Create model directory
        os.makedirs("models", exist_ok=True)
    
    def extract_movement_features(self, behavior_pattern: BehaviorPattern) -> np.ndarray:
        """Extract features from behavior pattern for classification"""
        features = [
            behavior_pattern.people_count,
            behavior_pattern.average_speed,
            behavior_pattern.speed_variance,
            behavior_pattern.direction_consistency,
            behavior_pattern.acceleration_pattern,
            behavior_pattern.clustering_level,
            behavior_pattern.dispersion_level,
            # Panic indicators
            behavior_pattern.panic_indicators['high_speed'],
            behavior_pattern.panic_indicators['direction_change'],
            behavior_pattern.panic_indicators['acceleration_spike'],
            behavior_pattern.panic_indicators['clustering_breakdown'],
            behavior_pattern.panic_indicators['dispersion_increase'],
            behavior_pattern.panic_indicators['movement_irregularity'],
            # Derived features
            behavior_pattern.average_speed * behavior_pattern.speed_variance,  # Speed instability
            behavior_pattern.direction_consistency * behavior_pattern.clustering_level,  # Organization
            behavior_pattern.acceleration_pattern * behavior_pattern.panic_indicators['acceleration_spike'],  # Panic acceleration
            # Temporal features
            datetime.fromtimestamp(behavior_pattern.timestamp).hour,
            datetime.fromtimestamp(behavior_pattern.timestamp).minute,
            datetime.fromtimestamp(behavior_pattern.timestamp).weekday(),
            # Movement complexity
            len(behavior_pattern.movement_vectors),
            np.std([v.magnitude for v in behavior_pattern.movement_vectors]) if behavior_pattern.movement_vectors else 0,
            np.mean([abs(v.direction) for v in behavior_pattern.movement_vectors]) if behavior_pattern.movement_vectors else 0,
        ]
        
        return np.array(features, dtype=np.float32)
    
    def calculate_panic_score(self, behavior_pattern: BehaviorPattern) -> float:
        """Calculate panic score based on movement characteristics"""
        panic_score = 0.0
        
        # High speed indicator
        if behavior_pattern.average_speed > 2.0:  # m/s
            panic_score += 0.3
        
        # Speed variance (erratic movement)
        if behavior_pattern.speed_variance > 1.0:
            panic_score += 0.2
        
        # Direction inconsistency
        if behavior_pattern.direction_consistency < 0.3:
            panic_score += 0.2
        
        # High acceleration
        if behavior_pattern.acceleration_pattern > 0.5:
            panic_score += 0.2
        
        # Clustering breakdown
        if behavior_pattern.clustering_level < 0.3:
            panic_score += 0.1
        
        # Panic indicators
        panic_score += sum(behavior_pattern.panic_indicators.values()) * 0.1
        
        return min(panic_score, 1.0)
    
    def classify_behavior(self, behavior_pattern: BehaviorPattern) -> BehaviorClassification:
        """Classify crowd behavior pattern"""
        
        if not self.is_trained:
            # Fallback classification based on rules
            panic_score = self.calculate_panic_score(behavior_pattern)
            
            if panic_score > 0.7:
                behavior_type = "panic_running"
                confidence = 0.8
                risk_level = "critical"
                description = "Panic running detected - immediate evacuation needed"
                action = "evacuate_immediately"
            elif panic_score > 0.5:
                behavior_type = "chaotic_movement"
                confidence = 0.7
                risk_level = "high"
                description = "Chaotic movement patterns detected"
                action = "increase_monitoring"
            elif behavior_pattern.average_speed > 1.5:
                behavior_type = "running"
                confidence = 0.6
                risk_level = "medium"
                description = "Running movement detected"
                action = "investigate_cause"
            elif behavior_pattern.people_count > 50:
                behavior_type = "crowded_walking"
                confidence = 0.7
                risk_level = "low"
                description = "Crowded but normal walking"
                action = "monitor_density"
            else:
                behavior_type = "normal_walking"
                confidence = 0.8
                risk_level = "low"
                description = "Normal walking patterns"
                action = "continue_monitoring"
        else:
            try:
                # Use ML model for classification
                features = self.extract_movement_features(behavior_pattern)
                features_scaled = self.scaler.transform(features.reshape(1, -1))
                
                # Predict behavior type
                behavior_type_encoded = self.behavior_classifier.predict(features_scaled)[0]
                behavior_type = self.label_encoder.inverse_transform([behavior_type_encoded])[0]
                
                # Get confidence
                confidence_scores = self.behavior_classifier.predict_proba(features_scaled)[0]
                confidence = np.max(confidence_scores)
                
                # Calculate panic score
                panic_score = self.calculate_panic_score(behavior_pattern)
                
                # Determine risk level and action
                risk_level, description, action = self._determine_risk_and_action(
                    behavior_type, panic_score, confidence
                )
                
            except Exception as e:
                print(f"⚠️ Behavior classification error: {e}")
                # Fallback to rule-based classification
                panic_score = self.calculate_panic_score(behavior_pattern)
                behavior_type = "normal_walking"
                confidence = 0.5
                risk_level = "low"
                description = "Classification error - using fallback"
                action = "continue_monitoring"
        
        # Calculate movement characteristics
        movement_characteristics = {
            'average_speed': behavior_pattern.average_speed,
            'speed_variance': behavior_pattern.speed_variance,
            'direction_consistency': behavior_pattern.direction_consistency,
            'acceleration_pattern': behavior_pattern.acceleration_pattern,
            'clustering_level': behavior_pattern.clustering_level,
            'dispersion_level': behavior_pattern.dispersion_level,
            'panic_score': panic_score
        }
        
        return BehaviorClassification(
            timestamp=behavior_pattern.timestamp,
            behavior_type=behavior_type,
            confidence=confidence,
            panic_score=panic_score,
            risk_level=risk_level,
            description=description,
            recommended_action=action,
            movement_characteristics=movement_characteristics
        )
    
    def _determine_risk_and_action(self, behavior_type: str, panic_score: float, 
                                 confidence: float) -> Tuple[str, str, str]:
        """Determine risk level and recommended action"""
        
        if behavior_type in ['panic_running', 'chaotic_movement']:
            return "critical", "Dangerous crowd behavior detected", "evacuate_immediately"
        elif behavior_type in ['running', 'evacuation_pattern']:
            return "high", "Rapid movement detected - investigate cause", "investigate_cause"
        elif behavior_type in ['crowded_walking', 'scattered_movement']:
            return "medium", "Unusual crowd patterns - monitor closely", "monitor_closely"
        elif behavior_type in ['normal_walking', 'organized_flow']:
            return "low", "Normal crowd behavior", "continue_monitoring"
        else:
            return "low", "Unknown behavior pattern", "continue_monitoring"
    
    def analyze_movement_from_detections(self, detections: List[Dict], 
                                       previous_detections: List[Dict],
                                       frame_time: float) -> BehaviorPattern:
        """Analyze movement patterns from detection data"""
        
        # Extract current positions
        current_positions = {}
        for detection in detections:
            person_id = detection.get('id', len(current_positions))
            center_x = detection.get('center_x', 0)
            center_y = detection.get('center_y', 0)
            current_positions[person_id] = (center_x, center_y)
        
        # Calculate movement vectors
        movement_vectors = []
        speeds = []
        accelerations = []
        
        for person_id, current_pos in current_positions.items():
            if person_id in self.previous_positions:
                prev_pos = self.previous_positions[person_id]
                
                # Calculate movement vector
                dx = current_pos[0] - prev_pos[0]
                dy = current_pos[1] - prev_pos[1]
                magnitude = np.sqrt(dx**2 + dy**2)
                direction = np.arctan2(dy, dx)
                
                movement_vector = MovementVector(
                    x=dx, y=dy, magnitude=magnitude, 
                    direction=direction, timestamp=frame_time
                )
                movement_vectors.append(movement_vector)
                speeds.append(magnitude)
                
                # Calculate acceleration if we have previous speed
                if person_id in self.previous_positions:
                    prev_speed = self.previous_positions.get(person_id + '_speed', 0)
                    acceleration = magnitude - prev_speed
                    accelerations.append(acceleration)
                
                # Store speed for next frame
                self.previous_positions[person_id + '_speed'] = magnitude
        
        # Update previous positions
        self.previous_positions = current_positions.copy()
        
        # Calculate behavior metrics
        average_speed = np.mean(speeds) if speeds else 0.0
        speed_variance = np.var(speeds) if speeds else 0.0
        
        # Direction consistency
        if movement_vectors:
            directions = [v.direction for v in movement_vectors]
            direction_consistency = 1.0 - np.std(directions) / np.pi
        else:
            direction_consistency = 0.0
        
        # Acceleration pattern
        acceleration_pattern = np.mean(accelerations) if accelerations else 0.0
        
        # Clustering level (how close people are to each other)
        clustering_level = self._calculate_clustering_level(current_positions)
        
        # Dispersion level
        dispersion_level = self._calculate_dispersion_level(current_positions)
        
        # Calculate panic indicators
        panic_indicators = self._calculate_panic_indicators(
            movement_vectors, speeds, accelerations, clustering_level, dispersion_level
        )
        
        return BehaviorPattern(
            timestamp=frame_time,
            people_count=len(detections),
            movement_vectors=movement_vectors,
            average_speed=average_speed,
            speed_variance=speed_variance,
            direction_consistency=direction_consistency,
            acceleration_pattern=acceleration_pattern,
            clustering_level=clustering_level,
            dispersion_level=dispersion_level,
            panic_indicators=panic_indicators
        )
    
    def _calculate_clustering_level(self, positions: Dict[int, Tuple[float, float]]) -> float:
        """Calculate how clustered the crowd is"""
        if len(positions) < 2:
            return 0.0
        
        positions_list = list(positions.values())
        distances = []
        
        for i in range(len(positions_list)):
            for j in range(i + 1, len(positions_list)):
                dist = np.sqrt((positions_list[i][0] - positions_list[j][0])**2 + 
                             (positions_list[i][1] - positions_list[j][1])**2)
                distances.append(dist)
        
        if distances:
            avg_distance = np.mean(distances)
            # Normalize clustering level (closer = higher clustering)
            clustering_level = max(0, 1.0 - avg_distance / 100.0)  # Assuming 100px is max distance
            return clustering_level
        
        return 0.0
    
    def _calculate_dispersion_level(self, positions: Dict[int, Tuple[float, float]]) -> float:
        """Calculate how dispersed the crowd is"""
        if len(positions) < 2:
            return 0.0
        
        positions_list = list(positions.values())
        
        # Calculate center of mass
        center_x = np.mean([pos[0] for pos in positions_list])
        center_y = np.mean([pos[1] for pos in positions_list])
        
        # Calculate distances from center
        distances_from_center = [
            np.sqrt((pos[0] - center_x)**2 + (pos[1] - center_y)**2) 
            for pos in positions_list
        ]
        
        # Dispersion is the standard deviation of distances from center
        dispersion_level = np.std(distances_from_center) / 50.0  # Normalize
        return min(dispersion_level, 1.0)
    
    def _calculate_panic_indicators(self, movement_vectors: List[MovementVector], 
                                 speeds: List[float], accelerations: List[float],
                                 clustering_level: float, dispersion_level: float) -> Dict[str, float]:
        """Calculate panic indicators"""
        indicators = {}
        
        # High speed indicator
        indicators['high_speed'] = 1.0 if np.mean(speeds) > 2.0 else 0.0
        
        # Direction change indicator
        if movement_vectors:
            direction_changes = []
            for i in range(1, len(movement_vectors)):
                angle_diff = abs(movement_vectors[i].direction - movement_vectors[i-1].direction)
                direction_changes.append(min(angle_diff, 2*np.pi - angle_diff))
            indicators['direction_change'] = np.mean(direction_changes) / np.pi
        else:
            indicators['direction_change'] = 0.0
        
        # Acceleration spike indicator
        indicators['acceleration_spike'] = 1.0 if np.mean(accelerations) > 0.5 else 0.0
        
        # Clustering breakdown indicator
        indicators['clustering_breakdown'] = 1.0 - clustering_level
        
        # Dispersion increase indicator
        indicators['dispersion_increase'] = dispersion_level
        
        # Movement irregularity indicator
        if speeds:
            speed_cv = np.std(speeds) / np.mean(speeds) if np.mean(speeds) > 0 else 0
            indicators['movement_irregularity'] = min(speed_cv, 1.0)
        else:
            indicators['movement_irregularity'] = 0.0
        
        return indicators
    
    def train_model(self, training_data: List[Tuple[BehaviorPattern, str]]):
        """Train the behavior classification model"""
        try:
            if len(training_data) < 50:
                print("⚠️ Insufficient training data (need at least 50 samples)")
                return False
            
            # Prepare training data
            X = []
            y = []
            
            for pattern, behavior_type in training_data:
                features = self.extract_movement_features(pattern)
                X.append(features)
                y.append(behavior_type)
            
            X = np.array(X)
            y = np.array(y)
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.behavior_classifier = RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight='balanced'
            )
            self.behavior_classifier.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.behavior_classifier.predict(X_test_scaled)
            accuracy = np.mean(y_pred == y_test)
            self.classification_accuracy = accuracy
            
            # Save model
            model_path = "models/behavior_classification_model.pkl"
            joblib.dump({
                'classifier': self.behavior_classifier,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'accuracy': accuracy,
                'timestamp': time.time()
            }, model_path)
            
            self.is_trained = True
            print(f"✅ Behavior classification model trained - Accuracy: {accuracy:.3f}")
            
            # Print classification report
            print("\n📊 Classification Report:")
            print(classification_report(y_test, y_pred, 
                                      target_names=self.label_encoder.classes_))
            
            return True
            
        except Exception as e:
            print(f"⚠️ Model training failed: {e}")
            return False
    
    def load_model(self, model_path: str = "models/behavior_classification_model.pkl"):
        """Load pre-trained behavior classification model"""
        try:
            if os.path.exists(model_path):
                model_data = joblib.load(model_path)
                self.behavior_classifier = model_data['classifier']
                self.scaler = model_data['scaler']
                self.label_encoder = model_data['label_encoder']
                self.classification_accuracy = model_data['accuracy']
                self.is_trained = True
                print(f"✅ Loaded behavior classification model - Accuracy: {self.classification_accuracy:.3f}")
                return True
        except Exception as e:
            print(f"⚠️ Failed to load behavior model: {e}")
        return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'is_trained': self.is_trained,
            'classification_accuracy': self.classification_accuracy,
            'panic_detection_accuracy': self.panic_detection_accuracy,
            'movement_history_size': len(self.movement_history),
            'behavior_history_size': len(self.behavior_history),
            'behavior_types': self.behavior_types,
            'panic_indicators': list(self.panic_indicators.keys())
        }
    
    def simulate_behavior_pattern(self, behavior_type: str = "normal_walking") -> BehaviorPattern:
        """Simulate behavior patterns for testing"""
        base_time = time.time()
        
        if behavior_type == "normal_walking":
            people_count = np.random.randint(20, 40)
            average_speed = np.random.uniform(0.5, 1.5)
            speed_variance = np.random.uniform(0.1, 0.3)
            direction_consistency = np.random.uniform(0.6, 0.9)
            acceleration_pattern = np.random.uniform(0.0, 0.2)
            clustering_level = np.random.uniform(0.4, 0.7)
            dispersion_level = np.random.uniform(0.2, 0.5)
            
        elif behavior_type == "panic_running":
            people_count = np.random.randint(30, 60)
            average_speed = np.random.uniform(2.5, 4.0)
            speed_variance = np.random.uniform(0.8, 1.5)
            direction_consistency = np.random.uniform(0.1, 0.4)
            acceleration_pattern = np.random.uniform(0.6, 1.0)
            clustering_level = np.random.uniform(0.1, 0.3)
            dispersion_level = np.random.uniform(0.7, 1.0)
            
        elif behavior_type == "running":
            people_count = np.random.randint(25, 45)
            average_speed = np.random.uniform(2.0, 3.0)
            speed_variance = np.random.uniform(0.4, 0.8)
            direction_consistency = np.random.uniform(0.3, 0.6)
            acceleration_pattern = np.random.uniform(0.3, 0.6)
            clustering_level = np.random.uniform(0.2, 0.5)
            dispersion_level = np.random.uniform(0.4, 0.7)
            
        else:  # random
            people_count = np.random.randint(10, 50)
            average_speed = np.random.uniform(0.0, 4.0)
            speed_variance = np.random.uniform(0.0, 2.0)
            direction_consistency = np.random.uniform(0.0, 1.0)
            acceleration_pattern = np.random.uniform(0.0, 1.0)
            clustering_level = np.random.uniform(0.0, 1.0)
            dispersion_level = np.random.uniform(0.0, 1.0)
        
        # Generate movement vectors
        movement_vectors = []
        for _ in range(people_count // 4):
            magnitude = np.random.uniform(0.0, average_speed * 2)
            direction = np.random.uniform(-np.pi, np.pi)
            movement_vectors.append(MovementVector(
                x=magnitude * np.cos(direction),
                y=magnitude * np.sin(direction),
                magnitude=magnitude,
                direction=direction,
                timestamp=base_time
            ))
        
        # Calculate panic indicators
        panic_indicators = {
            'high_speed': 1.0 if average_speed > 2.0 else 0.0,
            'direction_change': 1.0 - direction_consistency,
            'acceleration_spike': acceleration_pattern,
            'clustering_breakdown': 1.0 - clustering_level,
            'dispersion_increase': dispersion_level,
            'movement_irregularity': speed_variance / max(average_speed, 0.1)
        }
        
        return BehaviorPattern(
            timestamp=base_time,
            people_count=people_count,
            movement_vectors=movement_vectors,
            average_speed=average_speed,
            speed_variance=speed_variance,
            direction_consistency=direction_consistency,
            acceleration_pattern=acceleration_pattern,
            clustering_level=clustering_level,
            dispersion_level=dispersion_level,
            panic_indicators=panic_indicators
        )

# Example usage and testing
if __name__ == "__main__":
    # Initialize behavior analyzer
    analyzer = MovementBehaviorAnalyzer()
    
    # Load existing model if available
    analyzer.load_model()
    
    # Simulate training data
    print("🧪 Simulating training data...")
    training_data = []
    
    for behavior_type in analyzer.behavior_types:
        for _ in range(20):  # 20 samples per behavior type
            pattern = analyzer.simulate_behavior_pattern(behavior_type)
            training_data.append((pattern, behavior_type))
    
    # Train model
    analyzer.train_model(training_data)
    
    # Test behavior classification
    print("\n🔍 Testing behavior classification...")
    test_patterns = [
        ("normal_walking", analyzer.simulate_behavior_pattern("normal_walking")),
        ("panic_running", analyzer.simulate_behavior_pattern("panic_running")),
        ("running", analyzer.simulate_behavior_pattern("running")),
        ("random", analyzer.simulate_behavior_pattern("random"))
    ]
    
    for pattern_type, pattern in test_patterns:
        classification = analyzer.classify_behavior(pattern)
        print(f"🎯 {pattern_type.upper()}:")
        print(f"   Behavior Type: {classification.behavior_type}")
        print(f"   Confidence: {classification.confidence:.3f}")
        print(f"   Panic Score: {classification.panic_score:.3f}")
        print(f"   Risk Level: {classification.risk_level}")
        print(f"   Description: {classification.description}")
        print(f"   Action: {classification.recommended_action}")
        print()
    
    # Get performance stats
    stats = analyzer.get_performance_stats()
    print(f"📈 Behavior Analysis Statistics:")
    print(f"   Model Trained: {stats['is_trained']}")
    print(f"   Classification Accuracy: {stats['classification_accuracy']:.3f}")
    print(f"   Panic Detection Accuracy: {stats['panic_detection_accuracy']:.3f}")
    print(f"   Behavior Types: {len(stats['behavior_types'])}")
    print(f"   Panic Indicators: {len(stats['panic_indicators'])}")
