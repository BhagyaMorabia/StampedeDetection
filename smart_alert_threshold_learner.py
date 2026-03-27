"""
Smart Alert Threshold Learning System for STAMPede Detection
Uses machine learning to learn optimal alert thresholds for each specific location and context
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
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
from collections import deque, defaultdict
import warnings
warnings.filterwarnings('ignore')

@dataclass
class AlertContext:
    """Context information for alert threshold learning"""
    venue_id: str
    venue_type: str  # stadium, concert_hall, shopping_mall, etc.
    event_type: str  # sports, concert, festival, etc.
    time_of_day: float  # 0-24 hours
    day_of_week: int  # 0-6
    season: str  # spring, summer, fall, winter
    weather_condition: str  # clear, rain, snow, etc.
    lighting_condition: float  # 0-1 scale
    crowd_demographics: Dict[str, float]  # age groups, etc.
    historical_incidents: int  # number of past incidents
    venue_capacity: int
    current_capacity_ratio: float
    emergency_exits: int
    security_personnel: int
    crowd_management_measures: List[str]

@dataclass
class AlertThreshold:
    """Learned alert threshold for specific context"""
    context: AlertContext
    density_threshold: float
    people_count_threshold: int
    movement_threshold: float
    panic_threshold: float
    confidence: float
    learning_confidence: float
    last_updated: float
    sample_count: int

@dataclass
class AlertFeedback:
    """Feedback on alert performance"""
    timestamp: float
    alert_triggered: bool
    actual_incident: bool
    false_positive: bool
    false_negative: bool
    response_time: float  # seconds
    crowd_reaction: str  # calm, concerned, panicked
    intervention_effectiveness: float  # 0-1 scale
    context: AlertContext

@dataclass
class ThresholdLearningResult:
    """Result of threshold learning"""
    context: AlertContext
    learned_thresholds: AlertThreshold
    improvement_score: float
    confidence: float
    recommendations: List[str]
    risk_assessment: str

class SmartAlertThresholdLearner:
    """Advanced alert threshold learning system"""
    
    def __init__(self, learning_rate: float = 0.1, min_samples: int = 50):
        self.learning_rate = learning_rate
        self.min_samples = min_samples
        
        # Learning models
        self.density_model = None
        self.people_model = None
        self.movement_model = None
        self.panic_model = None
        
        # Feature processing
        self.scaler = StandardScaler()
        self.context_encoder = LabelEncoder()
        self.is_trained = False
        
        # Data storage
        self.feedback_history = deque(maxlen=10000)
        self.threshold_history = deque(maxlen=1000)
        self.context_patterns = defaultdict(list)
        
        # Performance tracking
        self.learning_accuracy = 0.0
        self.false_positive_reduction = 0.0
        self.false_negative_reduction = 0.0
        
        # Default thresholds
        self.default_thresholds = {
            'density': 6.0,  # people/m²
            'people_count': 50,
            'movement': 0.7,  # movement intensity
            'panic': 0.8  # panic score
        }
        
        # Create model directory
        os.makedirs("models", exist_ok=True)
    
    def extract_context_features(self, context: AlertContext) -> np.ndarray:
        """Extract features from alert context"""
        features = [
            # Venue features
            hash(context.venue_id) % 1000,  # Venue ID hash
            hash(context.venue_type) % 100,  # Venue type hash
            hash(context.event_type) % 100,  # Event type hash
            
            # Temporal features
            context.time_of_day,
            context.day_of_week,
            hash(context.season) % 10,  # Season hash
            
            # Environmental features
            hash(context.weather_condition) % 20,  # Weather hash
            context.lighting_condition,
            
            # Crowd features
            context.crowd_demographics.get('adults', 0.5),
            context.crowd_demographics.get('children', 0.1),
            context.crowd_demographics.get('elderly', 0.1),
            
            # Venue capacity features
            context.venue_capacity,
            context.current_capacity_ratio,
            context.emergency_exits,
            context.security_personnel,
            
            # Historical features
            context.historical_incidents,
            
            # Derived features
            context.current_capacity_ratio * context.venue_capacity,  # Current people
            context.security_personnel / max(context.venue_capacity, 1),  # Security ratio
            context.emergency_exits / max(context.venue_capacity, 1),  # Exit ratio
            
            # Time-based features
            np.sin(2 * np.pi * context.time_of_day / 24),  # Hour sine
            np.cos(2 * np.pi * context.time_of_day / 24),  # Hour cosine
            np.sin(2 * np.pi * context.day_of_week / 7),  # Day sine
            np.cos(2 * np.pi * context.day_of_week / 7),  # Day cosine
            
            # Risk indicators
            1 if context.historical_incidents > 0 else 0,  # Has incidents
            1 if context.current_capacity_ratio > 0.8 else 0,  # High capacity
            1 if context.time_of_day >= 22 or context.time_of_day <= 6 else 0,  # Night time
            1 if context.day_of_week >= 5 else 0,  # Weekend
        ]
        
        return np.array(features, dtype=np.float32)
    
    def learn_thresholds(self, feedback_data: List[AlertFeedback]) -> Dict[str, AlertThreshold]:
        """Learn optimal thresholds from feedback data"""
        
        if len(feedback_data) < self.min_samples:
            print(f"⚠️ Insufficient feedback data (need {self.min_samples}, have {len(feedback_data)})")
            return {}
        
        # Group feedback by context
        context_groups = defaultdict(list)
        for feedback in feedback_data:
            context_key = self._get_context_key(feedback.context)
            context_groups[context_key].append(feedback)
        
        learned_thresholds = {}
        
        for context_key, feedbacks in context_groups.items():
            if len(feedbacks) < 10:  # Need minimum samples per context
                continue
            
            print(f"🔄 Learning thresholds for context: {context_key}")
            
            # Extract features and targets
            X = []
            y_density = []
            y_people = []
            y_movement = []
            y_panic = []
            
            for feedback in feedbacks:
                context_features = self.extract_context_features(feedback.context)
                X.append(context_features)
                
                # Calculate optimal thresholds based on feedback
                if feedback.actual_incident:
                    # Lower thresholds for contexts with actual incidents
                    y_density.append(self.default_thresholds['density'] * 0.8)
                    y_people.append(int(self.default_thresholds['people_count'] * 0.8))
                    y_movement.append(self.default_thresholds['movement'] * 0.8)
                    y_panic.append(self.default_thresholds['panic'] * 0.8)
                elif feedback.false_positive:
                    # Higher thresholds for contexts with false positives
                    y_density.append(self.default_thresholds['density'] * 1.2)
                    y_people.append(int(self.default_thresholds['people_count'] * 1.2))
                    y_movement.append(self.default_thresholds['movement'] * 1.2)
                    y_panic.append(self.default_thresholds['panic'] * 1.2)
                else:
                    # Keep default thresholds
                    y_density.append(self.default_thresholds['density'])
                    y_people.append(self.default_thresholds['people_count'])
                    y_movement.append(self.default_thresholds['movement'])
                    y_panic.append(self.default_thresholds['panic'])
            
            X = np.array(X)
            y_density = np.array(y_density)
            y_people = np.array(y_people)
            y_movement = np.array(y_movement)
            y_panic = np.array(y_panic)
            
            # Train models for each threshold type
            thresholds = {}
            
            for threshold_type, y_target in [
                ('density', y_density),
                ('people', y_people),
                ('movement', y_movement),
                ('panic', y_panic)
            ]:
                try:
                    # Scale features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Train model
                    model = RandomForestRegressor(n_estimators=50, random_state=42)
                    model.fit(X_scaled, y_target)
                    
                    # Calculate confidence
                    predictions = model.predict(X_scaled)
                    mse = mean_squared_error(y_target, predictions)
                    confidence = max(0.1, min(0.9, 1.0 - mse / np.var(y_target)))
                    
                    thresholds[threshold_type] = {
                        'model': model,
                        'scaler': scaler,
                        'confidence': confidence,
                        'value': np.mean(y_target)
                    }
                    
                except Exception as e:
                    print(f"⚠️ Failed to learn {threshold_type} threshold: {e}")
                    continue
            
            if len(thresholds) >= 2:  # Need at least 2 threshold types
                # Create learned threshold
                context = feedbacks[0].context  # Use first context as representative
                
                learned_threshold = AlertThreshold(
                    context=context,
                    density_threshold=thresholds.get('density', {}).get('value', self.default_thresholds['density']),
                    people_count_threshold=int(thresholds.get('people', {}).get('value', self.default_thresholds['people_count'])),
                    movement_threshold=thresholds.get('movement', {}).get('value', self.default_thresholds['movement']),
                    panic_threshold=thresholds.get('panic', {}).get('value', self.default_thresholds['panic']),
                    confidence=np.mean([t.get('confidence', 0.5) for t in thresholds.values()]),
                    learning_confidence=len(feedbacks) / 100.0,  # Based on sample count
                    last_updated=time.time(),
                    sample_count=len(feedbacks)
                )
                
                learned_thresholds[context_key] = learned_threshold
                
                # Store threshold history
                self.threshold_history.append(learned_threshold)
                
                print(f"✅ Learned thresholds for {context_key}:")
                print(f"   Density: {learned_threshold.density_threshold:.2f}")
                print(f"   People: {learned_threshold.people_count_threshold}")
                print(f"   Movement: {learned_threshold.movement_threshold:.2f}")
                print(f"   Panic: {learned_threshold.panic_threshold:.2f}")
                print(f"   Confidence: {learned_threshold.confidence:.3f}")
        
        self.is_trained = len(learned_thresholds) > 0
        return learned_thresholds
    
    def _get_context_key(self, context: AlertContext) -> str:
        """Generate a key for grouping similar contexts"""
        return f"{context.venue_type}_{context.event_type}_{context.season}"
    
    def get_optimal_thresholds(self, context: AlertContext) -> AlertThreshold:
        """Get optimal thresholds for a specific context"""
        
        context_key = self._get_context_key(context)
        
        # Look for exact match first
        for threshold in self.threshold_history:
            if self._get_context_key(threshold.context) == context_key:
                return threshold
        
        # Look for similar contexts
        similar_thresholds = []
        for threshold in self.threshold_history:
            similarity = self._calculate_context_similarity(context, threshold.context)
            if similarity > 0.7:  # 70% similarity threshold
                similar_thresholds.append((threshold, similarity))
        
        if similar_thresholds:
            # Use most similar context
            best_threshold, best_similarity = max(similar_thresholds, key=lambda x: x[1])
            
            # Adjust thresholds based on context differences
            adjusted_threshold = self._adjust_thresholds_for_context(
                best_threshold, context, best_similarity
            )
            return adjusted_threshold
        
        # Fallback to default thresholds
        return AlertThreshold(
            context=context,
            density_threshold=self.default_thresholds['density'],
            people_count_threshold=self.default_thresholds['people_count'],
            movement_threshold=self.default_thresholds['movement'],
            panic_threshold=self.default_thresholds['panic'],
            confidence=0.3,  # Low confidence for default
            learning_confidence=0.0,
            last_updated=time.time(),
            sample_count=0
        )
    
    def _calculate_context_similarity(self, context1: AlertContext, context2: AlertContext) -> float:
        """Calculate similarity between two contexts"""
        similarities = []
        
        # Venue type similarity
        similarities.append(1.0 if context1.venue_type == context2.venue_type else 0.0)
        
        # Event type similarity
        similarities.append(1.0 if context1.event_type == context2.event_type else 0.0)
        
        # Time similarity (closer times are more similar)
        time_diff = abs(context1.time_of_day - context2.time_of_day)
        time_similarity = max(0, 1.0 - time_diff / 12.0)  # 12-hour window
        similarities.append(time_similarity)
        
        # Day similarity
        day_similarity = 1.0 if context1.day_of_week == context2.day_of_week else 0.5
        similarities.append(day_similarity)
        
        # Season similarity
        similarities.append(1.0 if context1.season == context2.season else 0.0)
        
        # Weather similarity
        similarities.append(1.0 if context1.weather_condition == context2.weather_condition else 0.0)
        
        # Capacity similarity
        capacity_diff = abs(context1.current_capacity_ratio - context2.current_capacity_ratio)
        capacity_similarity = max(0, 1.0 - capacity_diff)
        similarities.append(capacity_similarity)
        
        return np.mean(similarities)
    
    def _adjust_thresholds_for_context(self, base_threshold: AlertThreshold, 
                                     target_context: AlertContext, 
                                     similarity: float) -> AlertThreshold:
        """Adjust thresholds based on context differences"""
        
        # Calculate adjustment factors
        adjustments = {}
        
        # Adjust based on historical incidents
        if target_context.historical_incidents > base_threshold.context.historical_incidents:
            adjustments['density'] = 0.9  # Lower threshold for venues with more incidents
            adjustments['people'] = 0.9
            adjustments['movement'] = 0.9
            adjustments['panic'] = 0.9
        
        # Adjust based on capacity
        if target_context.current_capacity_ratio > base_threshold.context.current_capacity_ratio:
            adjustments['density'] = adjustments.get('density', 1.0) * 0.95
            adjustments['people'] = adjustments.get('people', 1.0) * 0.95
        
        # Adjust based on security personnel
        security_ratio_target = target_context.security_personnel / max(target_context.venue_capacity, 1)
        security_ratio_base = base_threshold.context.security_personnel / max(base_threshold.context.venue_capacity, 1)
        
        if security_ratio_target < security_ratio_base:
            adjustments['density'] = adjustments.get('density', 1.0) * 0.9
            adjustments['people'] = adjustments.get('people', 1.0) * 0.9
        
        # Apply adjustments
        adjusted_threshold = AlertThreshold(
            context=target_context,
            density_threshold=base_threshold.density_threshold * adjustments.get('density', 1.0),
            people_count_threshold=int(base_threshold.people_count_threshold * adjustments.get('people', 1.0)),
            movement_threshold=base_threshold.movement_threshold * adjustments.get('movement', 1.0),
            panic_threshold=base_threshold.panic_threshold * adjustments.get('panic', 1.0),
            confidence=base_threshold.confidence * similarity,
            learning_confidence=base_threshold.learning_confidence * similarity,
            last_updated=time.time(),
            sample_count=base_threshold.sample_count
        )
        
        return adjusted_threshold
    
    def add_feedback(self, feedback: AlertFeedback):
        """Add new feedback for learning"""
        self.feedback_history.append(feedback)
        
        # Update context patterns
        context_key = self._get_context_key(feedback.context)
        self.context_patterns[context_key].append(feedback)
        
        # Retrain if we have enough new data
        if len(self.feedback_history) % 100 == 0:  # Retrain every 100 feedbacks
            self.learn_thresholds(list(self.feedback_history))
    
    def evaluate_threshold_performance(self, context: AlertContext, 
                                    current_density: float, 
                                    current_people: int,
                                    current_movement: float,
                                    current_panic: float) -> ThresholdLearningResult:
        """Evaluate how well current thresholds would perform"""
        
        optimal_thresholds = self.get_optimal_thresholds(context)
        
        # Simulate alert decisions
        density_alert = current_density > optimal_thresholds.density_threshold
        people_alert = current_people > optimal_thresholds.people_count_threshold
        movement_alert = current_movement > optimal_thresholds.movement_threshold
        panic_alert = current_panic > optimal_thresholds.panic_threshold
        
        # Calculate improvement score
        improvement_score = 0.0
        
        # Compare with default thresholds
        default_density_alert = current_density > self.default_thresholds['density']
        default_people_alert = current_people > self.default_thresholds['people_count']
        default_movement_alert = current_movement > self.default_thresholds['movement']
        default_panic_alert = current_panic > self.default_thresholds['panic']
        
        # Calculate improvements
        if density_alert != default_density_alert:
            improvement_score += 0.25
        if people_alert != default_people_alert:
            improvement_score += 0.25
        if movement_alert != default_movement_alert:
            improvement_score += 0.25
        if panic_alert != default_panic_alert:
            improvement_score += 0.25
        
        # Generate recommendations
        recommendations = []
        
        if optimal_thresholds.confidence < 0.5:
            recommendations.append("Collect more feedback data for this context")
        
        if optimal_thresholds.learning_confidence < 0.3:
            recommendations.append("Increase sample size for better learning")
        
        if current_density > optimal_thresholds.density_threshold * 0.8:
            recommendations.append("Monitor density closely - approaching threshold")
        
        if current_people > optimal_thresholds.people_count_threshold * 0.8:
            recommendations.append("Monitor people count closely - approaching threshold")
        
        # Risk assessment
        risk_factors = 0
        if density_alert:
            risk_factors += 1
        if people_alert:
            risk_factors += 1
        if movement_alert:
            risk_factors += 1
        if panic_alert:
            risk_factors += 1
        
        if risk_factors >= 3:
            risk_assessment = "critical"
        elif risk_factors >= 2:
            risk_assessment = "high"
        elif risk_factors >= 1:
            risk_assessment = "medium"
        else:
            risk_assessment = "low"
        
        return ThresholdLearningResult(
            context=context,
            learned_thresholds=optimal_thresholds,
            improvement_score=improvement_score,
            confidence=optimal_thresholds.confidence,
            recommendations=recommendations,
            risk_assessment=risk_assessment
        )
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning system statistics"""
        return {
            'is_trained': self.is_trained,
            'learning_accuracy': self.learning_accuracy,
            'false_positive_reduction': self.false_positive_reduction,
            'false_negative_reduction': self.false_negative_reduction,
            'feedback_history_size': len(self.feedback_history),
            'threshold_history_size': len(self.threshold_history),
            'context_patterns_count': len(self.context_patterns),
            'min_samples': self.min_samples,
            'learning_rate': self.learning_rate,
            'default_thresholds': self.default_thresholds
        }
    
    def save_models(self, model_path: str = "models/smart_alert_thresholds.pkl"):
        """Save learned models and thresholds"""
        model_data = {
            'threshold_history': list(self.threshold_history),
            'context_patterns': dict(self.context_patterns),
            'learning_statistics': self.get_learning_statistics(),
            'timestamp': time.time()
        }
        
        joblib.dump(model_data, model_path)
        print(f"✅ Smart alert thresholds saved to {model_path}")
    
    def load_models(self, model_path: str = "models/smart_alert_thresholds.pkl"):
        """Load learned models and thresholds"""
        try:
            if os.path.exists(model_path):
                model_data = joblib.load(model_path)
                self.threshold_history = deque(model_data['threshold_history'], maxlen=1000)
                self.context_patterns = defaultdict(list, model_data['context_patterns'])
                
                stats = model_data['learning_statistics']
                self.learning_accuracy = stats['learning_accuracy']
                self.false_positive_reduction = stats['false_positive_reduction']
                self.false_negative_reduction = stats['false_negative_reduction']
                
                self.is_trained = len(self.threshold_history) > 0
                print(f"✅ Loaded smart alert thresholds - {len(self.threshold_history)} thresholds")
                return True
        except Exception as e:
            print(f"⚠️ Failed to load smart alert thresholds: {e}")
        return False
    
    def simulate_alert_feedback(self, context: AlertContext) -> AlertFeedback:
        """Simulate alert feedback for testing"""
        
        # Simulate realistic feedback based on context
        if context.historical_incidents > 0:
            # Venues with incidents are more likely to have actual incidents
            actual_incident = np.random.random() < 0.3
        else:
            actual_incident = np.random.random() < 0.1
        
        if actual_incident:
            alert_triggered = np.random.random() < 0.8  # 80% detection rate
            false_positive = False
            false_negative = not alert_triggered
        else:
            alert_triggered = np.random.random() < 0.2  # 20% false positive rate
            false_positive = alert_triggered
            false_negative = False
        
        return AlertFeedback(
            timestamp=time.time(),
            alert_triggered=alert_triggered,
            actual_incident=actual_incident,
            false_positive=false_positive,
            false_negative=false_negative,
            response_time=np.random.uniform(30, 300),  # 30 seconds to 5 minutes
            crowd_reaction=np.random.choice(['calm', 'concerned', 'panicked']),
            intervention_effectiveness=np.random.uniform(0.3, 1.0),
            context=context
        )
    
    def simulate_alert_context(self, venue_type: str = "stadium") -> AlertContext:
        """Simulate alert context for testing"""
        
        venue_types = {
            "stadium": {"capacity": 50000, "exits": 20, "security": 100},
            "concert_hall": {"capacity": 5000, "exits": 8, "security": 20},
            "shopping_mall": {"capacity": 10000, "exits": 15, "security": 30},
            "festival": {"capacity": 20000, "exits": 12, "security": 50}
        }
        
        venue_info = venue_types.get(venue_type, venue_types["stadium"])
        
        return AlertContext(
            venue_id=f"venue_{np.random.randint(1, 100)}",
            venue_type=venue_type,
            event_type=np.random.choice(['sports', 'concert', 'festival', 'exhibition']),
            time_of_day=np.random.uniform(0, 24),
            day_of_week=np.random.randint(0, 7),
            season=np.random.choice(['spring', 'summer', 'fall', 'winter']),
            weather_condition=np.random.choice(['clear', 'rain', 'snow', 'fog']),
            lighting_condition=np.random.uniform(0.3, 1.0),
            crowd_demographics={
                'adults': np.random.uniform(0.6, 0.9),
                'children': np.random.uniform(0.05, 0.2),
                'elderly': np.random.uniform(0.05, 0.15)
            },
            historical_incidents=np.random.randint(0, 5),
            venue_capacity=venue_info['capacity'],
            current_capacity_ratio=np.random.uniform(0.1, 0.9),
            emergency_exits=venue_info['exits'],
            security_personnel=venue_info['security'],
            crowd_management_measures=['barriers', 'signage', 'staff']
        )

# Example usage and testing
if __name__ == "__main__":
    # Initialize smart alert learner
    learner = SmartAlertThresholdLearner()
    
    # Load existing models if available
    learner.load_models()
    
    # Simulate training data
    print("🧪 Simulating training data...")
    feedback_data = []
    
    venue_types = ["stadium", "concert_hall", "shopping_mall", "festival"]
    
    for venue_type in venue_types:
        for _ in range(25):  # 25 feedback samples per venue type
            context = learner.simulate_alert_context(venue_type)
            feedback = learner.simulate_alert_feedback(context)
            feedback_data.append(feedback)
    
    # Learn thresholds
    print("\n🔄 Learning optimal thresholds...")
    learned_thresholds = learner.learn_thresholds(feedback_data)
    
    # Test threshold learning
    print("\n🔍 Testing threshold learning...")
    test_context = learner.simulate_alert_context("stadium")
    
    # Simulate current conditions
    current_density = 5.5
    current_people = 45
    current_movement = 0.6
    current_panic = 0.7
    
    result = learner.evaluate_threshold_performance(
        test_context, current_density, current_people, current_movement, current_panic
    )
    
    print(f"🎯 Threshold Learning Results:")
    print(f"   Context: {test_context.venue_type} - {test_context.event_type}")
    print(f"   Learned Density Threshold: {result.learned_thresholds.density_threshold:.2f}")
    print(f"   Learned People Threshold: {result.learned_thresholds.people_count_threshold}")
    print(f"   Learned Movement Threshold: {result.learned_thresholds.movement_threshold:.2f}")
    print(f"   Learned Panic Threshold: {result.learned_thresholds.panic_threshold:.2f}")
    print(f"   Confidence: {result.confidence:.3f}")
    print(f"   Improvement Score: {result.improvement_score:.3f}")
    print(f"   Risk Assessment: {result.risk_assessment}")
    print(f"   Recommendations: {result.recommendations}")
    
    # Get statistics
    stats = learner.get_learning_statistics()
    print(f"\n📈 Learning Statistics:")
    print(f"   Model Trained: {stats['is_trained']}")
    print(f"   Learning Accuracy: {stats['learning_accuracy']:.3f}")
    print(f"   False Positive Reduction: {stats['false_positive_reduction']:.3f}")
    print(f"   False Negative Reduction: {stats['false_negative_reduction']:.3f}")
    print(f"   Feedback History: {stats['feedback_history_size']} samples")
    print(f"   Threshold History: {stats['threshold_history_size']} thresholds")
    print(f"   Context Patterns: {stats['context_patterns_count']} patterns")
    
    # Save models
    learner.save_models()
