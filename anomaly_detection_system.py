"""
Anomaly Detection System for STAMPede Detection System
Identifies unusual crowd patterns that don't fit normal behavior using ML algorithms
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
import json
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import joblib
from collections import deque
import warnings
warnings.filterwarnings('ignore')

@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection"""
    contamination: float = 0.1  # Expected proportion of anomalies
    n_estimators: int = 100  # Number of trees in Isolation Forest
    max_samples: int = 256  # Maximum samples per tree
    random_state: int = 42
    window_size: int = 60  # Time window for pattern analysis (seconds)
    min_samples: int = 50  # Minimum samples needed for training
    update_frequency: int = 200  # Update model every N detections
    anomaly_threshold: float = 0.3  # Threshold for anomaly score

@dataclass
class CrowdPattern:
    """Represents a crowd pattern at a specific time"""
    timestamp: float
    people_count: int
    density: float
    flow_intensity: float
    movement_direction: str
    spatial_distribution: List[float]  # Density in different areas
    velocity_vectors: List[Tuple[float, float]]  # Movement vectors
    acceleration_pattern: float
    clustering_coefficient: float
    entropy: float  # Disorder measure

@dataclass
class AnomalyResult:
    """Result of anomaly detection"""
    timestamp: float
    anomaly_score: float
    anomaly_type: str
    confidence: float
    description: str
    affected_areas: List[int]
    severity_level: str  # low, medium, high, critical
    recommended_action: str

class CrowdAnomalyDetector:
    """Advanced anomaly detection system for crowd behavior"""
    
    def __init__(self, config: Optional[AnomalyConfig] = None):
        self.config = config or AnomalyConfig()
        self.isolation_forest = IsolationForest(
            contamination=self.config.contamination,
            n_estimators=self.config.n_estimators,
            max_samples=self.config.max_samples,
            random_state=self.config.random_state
        )
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        
        self.is_trained = False
        self.pattern_history = deque(maxlen=1000)
        self.anomaly_history = deque(maxlen=100)
        self.update_count = 0
        
        # Pattern analysis
        self.normal_patterns = deque(maxlen=500)
        self.anomaly_patterns = deque(maxlen=100)
        
        # Performance tracking
        self.detection_accuracy = 0.0
        self.false_positive_rate = 0.0
        self.last_update_time = time.time()
        
        # Create model directory
        os.makedirs("models", exist_ok=True)
    
    def extract_pattern_features(self, pattern: CrowdPattern) -> np.ndarray:
        """Extract features from crowd pattern for anomaly detection"""
        features = [
            pattern.people_count,
            pattern.density,
            pattern.flow_intensity,
            pattern.acceleration_pattern,
            pattern.clustering_coefficient,
            pattern.entropy,
            # Spatial distribution features
            np.mean(pattern.spatial_distribution),
            np.std(pattern.spatial_distribution),
            np.max(pattern.spatial_distribution),
            np.min(pattern.spatial_distribution),
            # Movement features
            len(pattern.velocity_vectors),
            np.mean([v[0] for v in pattern.velocity_vectors]) if pattern.velocity_vectors else 0,
            np.mean([v[1] for v in pattern.velocity_vectors]) if pattern.velocity_vectors else 0,
            np.std([v[0] for v in pattern.velocity_vectors]) if pattern.velocity_vectors else 0,
            np.std([v[1] for v in pattern.velocity_vectors]) if pattern.velocity_vectors else 0,
            # Temporal features
            datetime.fromtimestamp(pattern.timestamp).hour,
            datetime.fromtimestamp(pattern.timestamp).minute,
            datetime.fromtimestamp(pattern.timestamp).weekday(),
            datetime.fromtimestamp(pattern.timestamp).month,
            # Derived features
            pattern.people_count * pattern.density,  # Crowd pressure
            pattern.flow_intensity * pattern.acceleration_pattern,  # Movement intensity
            pattern.clustering_coefficient * pattern.entropy,  # Disorder measure
        ]
        
        return np.array(features, dtype=np.float32)
    
    def detect_anomaly(self, pattern: CrowdPattern) -> AnomalyResult:
        """Detect anomalies in crowd pattern"""
        
        if not self.is_trained or len(self.pattern_history) < self.config.min_samples:
            return AnomalyResult(
                timestamp=pattern.timestamp,
                anomaly_score=0.0,
                anomaly_type="insufficient_data",
                confidence=0.0,
                description="Insufficient data for anomaly detection",
                affected_areas=[],
                severity_level="low",
                recommended_action="continue_monitoring"
            )
        
        try:
            # Extract features
            features = self.extract_pattern_features(pattern)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            features_pca = self.pca.transform(features_scaled)
            
            # Get anomaly score
            anomaly_score = self.isolation_forest.decision_function(features_pca)[0]
            is_anomaly = self.isolation_forest.predict(features_pca)[0] == -1
            
            # Classify anomaly type
            anomaly_type, confidence, description, severity, action = self._classify_anomaly(
                pattern, anomaly_score, is_anomaly
            )
            
            # Find affected areas
            affected_areas = self._find_affected_areas(pattern)
            
            result = AnomalyResult(
                timestamp=pattern.timestamp,
                anomaly_score=float(anomaly_score),
                anomaly_type=anomaly_type,
                confidence=confidence,
                description=description,
                affected_areas=affected_areas,
                severity_level=severity,
                recommended_action=action
            )
            
            # Store anomaly if detected
            if is_anomaly and anomaly_score < -self.config.anomaly_threshold:
                self.anomaly_history.append(result)
            
            return result
            
        except Exception as e:
            print(f"⚠️ Anomaly detection error: {e}")
            return AnomalyResult(
                timestamp=pattern.timestamp,
                anomaly_score=0.0,
                anomaly_type="error",
                confidence=0.0,
                description=f"Detection error: {str(e)}",
                affected_areas=[],
                severity_level="low",
                recommended_action="check_system"
            )
    
    def _classify_anomaly(self, pattern: CrowdPattern, anomaly_score: float, 
                         is_anomaly: bool) -> Tuple[str, float, str, str, str]:
        """Classify the type of anomaly"""
        
        if not is_anomaly:
            return "normal", 0.0, "Normal crowd pattern", "low", "continue_monitoring"
        
        # Analyze pattern characteristics
        density_threshold = 6.0  # people/m²
        flow_threshold = 0.7
        acceleration_threshold = 0.5
        
        if pattern.density > density_threshold:
            if pattern.flow_intensity > flow_threshold:
                if pattern.acceleration_pattern > acceleration_threshold:
                    return "stampede_risk", 0.9, "High density with rapid movement - STAMPEDE RISK", "critical", "evacuate_immediately"
                else:
                    return "high_density", 0.7, "High density with moderate movement", "high", "increase_monitoring"
            else:
                return "density_anomaly", 0.6, "Unusually high density", "medium", "monitor_closely"
        
        elif pattern.flow_intensity > flow_threshold:
            if pattern.acceleration_pattern > acceleration_threshold:
                return "panic_movement", 0.8, "Rapid panic-like movement detected", "high", "investigate_cause"
            else:
                return "flow_anomaly", 0.5, "Unusual crowd flow pattern", "medium", "monitor_movement"
        
        elif pattern.clustering_coefficient > 0.8:
            return "clustering_anomaly", 0.4, "Unusual clustering pattern", "medium", "check_obstacles"
        
        elif pattern.entropy > 0.9:
            return "disorder_anomaly", 0.5, "High disorder in crowd movement", "medium", "investigate_disturbance"
        
        else:
            return "general_anomaly", 0.3, "Unusual crowd pattern detected", "low", "continue_monitoring"
    
    def _find_affected_areas(self, pattern: CrowdPattern) -> List[int]:
        """Find areas most affected by the anomaly"""
        affected_areas = []
        
        # Find areas with highest density
        max_density = max(pattern.spatial_distribution) if pattern.spatial_distribution else 0
        threshold = max_density * 0.8
        
        for i, density in enumerate(pattern.spatial_distribution):
            if density > threshold:
                affected_areas.append(i)
        
        return affected_areas
    
    def update_model(self, pattern: CrowdPattern, is_anomaly: bool = False):
        """Update the anomaly detection model with new pattern"""
        self.pattern_history.append(pattern)
        self.update_count += 1
        
        # Store pattern in appropriate category
        if is_anomaly:
            self.anomaly_patterns.append(pattern)
        else:
            self.normal_patterns.append(pattern)
        
        # Retrain model periodically
        if (self.update_count % self.config.update_frequency == 0 and 
            len(self.pattern_history) >= self.config.min_samples):
            self._retrain_model()
    
    def _retrain_model(self):
        """Retrain the anomaly detection model"""
        try:
            if len(self.pattern_history) < self.config.min_samples:
                return
            
            # Prepare training data
            X = []
            for pattern in self.pattern_history:
                features = self.extract_pattern_features(pattern)
                X.append(features)
            
            X = np.array(X)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Apply PCA for dimensionality reduction
            X_pca = self.pca.fit_transform(X_scaled)
            
            # Train Isolation Forest
            self.isolation_forest.fit(X_pca)
            self.is_trained = True
            
            # Calculate performance metrics
            self._calculate_performance_metrics()
            
            # Save model
            model_path = "models/anomaly_detection_model.pkl"
            joblib.dump({
                'isolation_forest': self.isolation_forest,
                'scaler': self.scaler,
                'pca': self.pca,
                'timestamp': time.time(),
                'performance': {
                    'detection_accuracy': self.detection_accuracy,
                    'false_positive_rate': self.false_positive_rate
                }
            }, model_path)
            
            print(f"✅ Anomaly detection model retrained - Accuracy: {self.detection_accuracy:.3f}")
            
        except Exception as e:
            print(f"⚠️ Anomaly model retraining failed: {e}")
    
    def _calculate_performance_metrics(self):
        """Calculate performance metrics for the model"""
        try:
            if len(self.normal_patterns) < 10 or len(self.anomaly_patterns) < 5:
                return
            
            # Test on known patterns
            normal_features = [self.extract_pattern_features(p) for p in self.normal_patterns]
            anomaly_features = [self.extract_pattern_features(p) for p in self.anomaly_patterns]
            
            if normal_features and anomaly_features:
                X_test = np.array(normal_features + anomaly_features)
                X_test_scaled = self.scaler.transform(X_test)
                X_test_pca = self.pca.transform(X_test_scaled)
                
                predictions = self.isolation_forest.predict(X_test_pca)
                
                # Calculate metrics
                normal_predictions = predictions[:len(normal_features)]
                anomaly_predictions = predictions[len(normal_features):]
                
                true_negatives = np.sum(normal_predictions == 1)  # Correctly identified as normal
                false_positives = np.sum(normal_predictions == -1)  # Incorrectly identified as anomaly
                true_positives = np.sum(anomaly_predictions == -1)  # Correctly identified as anomaly
                false_negatives = np.sum(anomaly_predictions == 1)  # Incorrectly identified as normal
                
                if true_negatives + false_positives > 0:
                    self.false_positive_rate = false_positives / (true_negatives + false_positives)
                
                if true_positives + false_negatives > 0:
                    recall = true_positives / (true_positives + false_negatives)
                    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                    
                    if precision + recall > 0:
                        self.detection_accuracy = 2 * (precision * recall) / (precision + recall)
                
        except Exception as e:
            print(f"⚠️ Performance calculation failed: {e}")
    
    def load_model(self, model_path: str = "models/anomaly_detection_model.pkl"):
        """Load pre-trained anomaly detection model"""
        try:
            if os.path.exists(model_path):
                model_data = joblib.load(model_path)
                self.isolation_forest = model_data['isolation_forest']
                self.scaler = model_data['scaler']
                self.pca = model_data['pca']
                self.is_trained = True
                
                if 'performance' in model_data:
                    self.detection_accuracy = model_data['performance'].get('detection_accuracy', 0.0)
                    self.false_positive_rate = model_data['performance'].get('false_positive_rate', 0.0)
                
                print(f"✅ Loaded anomaly detection model - Accuracy: {self.detection_accuracy:.3f}")
                return True
        except Exception as e:
            print(f"⚠️ Failed to load anomaly model: {e}")
        return False
    
    def get_anomaly_statistics(self) -> Dict[str, Any]:
        """Get anomaly detection statistics"""
        recent_anomalies = [a for a in self.anomaly_history 
                          if time.time() - a.timestamp < 3600]  # Last hour
        
        anomaly_types = {}
        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        for anomaly in recent_anomalies:
            anomaly_types[anomaly.anomaly_type] = anomaly_types.get(anomaly.anomaly_type, 0) + 1
            severity_counts[anomaly.severity_level] += 1
        
        return {
            'is_trained': self.is_trained,
            'detection_accuracy': self.detection_accuracy,
            'false_positive_rate': self.false_positive_rate,
            'total_patterns': len(self.pattern_history),
            'normal_patterns': len(self.normal_patterns),
            'anomaly_patterns': len(self.anomaly_patterns),
            'recent_anomalies': len(recent_anomalies),
            'anomaly_types': anomaly_types,
            'severity_distribution': severity_counts,
            'last_update': self.last_update_time,
            'update_count': self.update_count
        }
    
    def simulate_crowd_pattern(self, anomaly_type: str = "normal") -> CrowdPattern:
        """Simulate crowd patterns for testing"""
        base_time = time.time()
        
        if anomaly_type == "normal":
            people_count = np.random.randint(20, 40)
            density = np.random.uniform(1.0, 3.0)
            flow_intensity = np.random.uniform(0.2, 0.5)
            acceleration = np.random.uniform(0.1, 0.3)
            clustering = np.random.uniform(0.3, 0.6)
            entropy = np.random.uniform(0.4, 0.7)
            
        elif anomaly_type == "high_density":
            people_count = np.random.randint(50, 80)
            density = np.random.uniform(6.0, 10.0)
            flow_intensity = np.random.uniform(0.3, 0.6)
            acceleration = np.random.uniform(0.2, 0.4)
            clustering = np.random.uniform(0.6, 0.8)
            entropy = np.random.uniform(0.6, 0.8)
            
        elif anomaly_type == "panic":
            people_count = np.random.randint(30, 60)
            density = np.random.uniform(4.0, 8.0)
            flow_intensity = np.random.uniform(0.7, 1.0)
            acceleration = np.random.uniform(0.6, 1.0)
            clustering = np.random.uniform(0.7, 0.9)
            entropy = np.random.uniform(0.8, 1.0)
            
        else:  # random
            people_count = np.random.randint(10, 70)
            density = np.random.uniform(0.5, 8.0)
            flow_intensity = np.random.uniform(0.1, 1.0)
            acceleration = np.random.uniform(0.0, 1.0)
            clustering = np.random.uniform(0.1, 0.9)
            entropy = np.random.uniform(0.2, 1.0)
        
        # Generate spatial distribution
        spatial_distribution = np.random.uniform(0.1, density, 16).tolist()
        
        # Generate velocity vectors
        velocity_vectors = []
        for _ in range(people_count // 4):
            vx = np.random.uniform(-flow_intensity, flow_intensity)
            vy = np.random.uniform(-flow_intensity, flow_intensity)
            velocity_vectors.append((vx, vy))
        
        return CrowdPattern(
            timestamp=base_time,
            people_count=people_count,
            density=density,
            flow_intensity=flow_intensity,
            movement_direction="mixed",
            spatial_distribution=spatial_distribution,
            velocity_vectors=velocity_vectors,
            acceleration_pattern=acceleration,
            clustering_coefficient=clustering,
            entropy=entropy
        )

# Example usage and testing
if __name__ == "__main__":
    # Initialize anomaly detector
    detector = CrowdAnomalyDetector()
    
    # Load existing model if available
    detector.load_model()
    
    # Simulate training data
    print("🧪 Simulating training data...")
    for i in range(200):
        # Generate normal patterns
        pattern = detector.simulate_crowd_pattern("normal")
        detector.update_model(pattern, is_anomaly=False)
        
        # Occasionally add anomalies
        if i % 20 == 0:
            anomaly_pattern = detector.simulate_crowd_pattern("high_density")
            detector.update_model(anomaly_pattern, is_anomaly=True)
        
        if i % 50 == 0:
            print(f"📊 Training iteration {i}: Patterns={len(detector.pattern_history)}")
    
    # Test anomaly detection
    print("\n🔍 Testing anomaly detection...")
    test_patterns = [
        ("normal", detector.simulate_crowd_pattern("normal")),
        ("high_density", detector.simulate_crowd_pattern("high_density")),
        ("panic", detector.simulate_crowd_pattern("panic")),
        ("random", detector.simulate_crowd_pattern("random"))
    ]
    
    for pattern_type, pattern in test_patterns:
        result = detector.detect_anomaly(pattern)
        print(f"🎯 {pattern_type.upper()}:")
        print(f"   Anomaly Score: {result.anomaly_score:.3f}")
        print(f"   Type: {result.anomaly_type}")
        print(f"   Severity: {result.severity_level}")
        print(f"   Description: {result.description}")
        print(f"   Action: {result.recommended_action}")
        print()
    
    # Get statistics
    stats = detector.get_anomaly_statistics()
    print(f"📈 Anomaly Detection Statistics:")
    print(f"   Model Trained: {stats['is_trained']}")
    print(f"   Detection Accuracy: {stats['detection_accuracy']:.3f}")
    print(f"   False Positive Rate: {stats['false_positive_rate']:.3f}")
    print(f"   Total Patterns: {stats['total_patterns']}")
    print(f"   Recent Anomalies: {stats['recent_anomalies']}")
    print(f"   Anomaly Types: {stats['anomaly_types']}")
    print(f"   Severity Distribution: {stats['severity_distribution']}")
