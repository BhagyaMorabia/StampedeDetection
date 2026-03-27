"""
ML Integration System for STAMPede Detection
Integrates all AI/ML features into a unified system for comprehensive crowd analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
import json
import os
from collections import deque, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import all ML modules
from adaptive_threshold_optimizer import AdaptiveThresholdOptimizer, EnvironmentalFactors as ThresholdEnvFactors
from anomaly_detection_system import CrowdAnomalyDetector, CrowdPattern
from behavior_analysis_system import MovementBehaviorAnalyzer, BehaviorPattern
from predictive_density_forecaster import CrowdDensityForecaster, DensityRecord
from person_reidentification_system import PersonReIdentifier, PersonDetection
from smart_alert_threshold_learner import SmartAlertThresholdLearner, AlertContext
from crowd_simulation_system import CrowdSimulator, SimulationResult
from environmental_integration_system import EnvironmentalIntegrator, EnvironmentalFactors

@dataclass
class UnifiedDetectionResult:
    """Unified result from all ML systems"""
    timestamp: float
    camera_id: int
    
    # Core detection results
    people_count: int
    density: float
    confidence: float
    
    # ML-enhanced results
    adaptive_threshold: float
    anomaly_score: float
    anomaly_type: str
    behavior_classification: str
    panic_score: float
    predicted_density_5min: float
    predicted_density_10min: float
    predicted_density_15min: float
    reid_global_id: int
    smart_alert_level: str
    environmental_impact: Dict[str, float]
    
    # Risk assessment
    overall_risk_score: float
    risk_level: str
    recommended_actions: List[str]
    
    # Confidence metrics
    ml_confidence: float
    system_confidence: float

@dataclass
class SystemConfiguration:
    """Configuration for the integrated ML system"""
    enable_adaptive_thresholds: bool = True
    enable_anomaly_detection: bool = True
    enable_behavior_analysis: bool = True
    enable_density_forecasting: bool = True
    enable_person_reid: bool = True
    enable_smart_alerts: bool = True
    enable_crowd_simulation: bool = True
    enable_environmental_integration: bool = True
    
    # Performance settings
    processing_mode: str = "balanced"  # "fast", "balanced", "accurate"
    update_frequency: float = 1.0  # seconds
    confidence_threshold: float = 0.7
    
    # Integration settings
    fusion_method: str = "weighted_average"  # "weighted_average", "ensemble", "bayesian"
    fallback_mode: bool = True

class IntegratedMLSystem:
    """Unified ML system integrating all AI/ML features"""
    
    def __init__(self, config: Optional[SystemConfiguration] = None):
        self.config = config or SystemConfiguration()
        
        # Initialize all ML components
        self.adaptive_threshold_optimizer = None
        self.anomaly_detector = None
        self.behavior_analyzer = None
        self.density_forecaster = None
        self.person_reidentifier = None
        self.smart_alert_learner = None
        self.crowd_simulator = None
        self.environmental_integrator = None
        
        # System state
        self.is_initialized = False
        self.is_running = False
        self.last_update_time = 0.0
        
        # Data storage
        self.detection_history = deque(maxlen=10000)
        self.alert_history = deque(maxlen=1000)
        self.performance_metrics = defaultdict(list)
        
        # Integration weights (learned over time)
        self.component_weights = {
            'adaptive_thresholds': 0.15,
            'anomaly_detection': 0.20,
            'behavior_analysis': 0.20,
            'density_forecasting': 0.15,
            'person_reid': 0.10,
            'smart_alerts': 0.10,
            'environmental': 0.10
        }
        
        # Performance tracking
        self.processing_times = defaultdict(list)
        self.accuracy_metrics = {}
        self.system_health = 1.0
        
        # Create model directory
        os.makedirs("models", exist_ok=True)
    
    def initialize_system(self, config_overrides: Dict[str, Any] = None):
        """Initialize all ML components"""
        
        print("🚀 Initializing Integrated ML System...")
        
        try:
            # Initialize adaptive threshold optimizer
            if self.config.enable_adaptive_thresholds:
                print("   📊 Initializing Adaptive Threshold Optimizer...")
                self.adaptive_threshold_optimizer = AdaptiveThresholdOptimizer()
                self.adaptive_threshold_optimizer.load_model()
            
            # Initialize anomaly detector
            if self.config.enable_anomaly_detection:
                print("   🔍 Initializing Anomaly Detection System...")
                self.anomaly_detector = CrowdAnomalyDetector()
                self.anomaly_detector.load_model()
            
            # Initialize behavior analyzer
            if self.config.enable_behavior_analysis:
                print("   🎯 Initializing Behavior Analysis System...")
                self.behavior_analyzer = MovementBehaviorAnalyzer()
                self.behavior_analyzer.load_model()
            
            # Initialize density forecaster
            if self.config.enable_density_forecasting:
                print("   📈 Initializing Density Forecasting System...")
                self.density_forecaster = CrowdDensityForecaster()
                self.density_forecaster.load_models()
            
            # Initialize person re-identifier
            if self.config.enable_person_reid:
                print("   👥 Initializing Person Re-identification System...")
                self.person_reidentifier = PersonReIdentifier()
            
            # Initialize smart alert learner
            if self.config.enable_smart_alerts:
                print("   🚨 Initializing Smart Alert Threshold Learner...")
                self.smart_alert_learner = SmartAlertThresholdLearner()
                self.smart_alert_learner.load_models()
            
            # Initialize crowd simulator
            if self.config.enable_crowd_simulation:
                print("   🎮 Initializing Crowd Simulation System...")
                self.crowd_simulator = CrowdSimulator()
            
            # Initialize environmental integrator
            if self.config.enable_environmental_integration:
                print("   🌍 Initializing Environmental Integration System...")
                self.environmental_integrator = EnvironmentalIntegrator()
            
            self.is_initialized = True
            print("✅ All ML components initialized successfully!")
            
            # Load system configuration overrides
            if config_overrides:
                self._apply_config_overrides(config_overrides)
            
            return True
            
        except Exception as e:
            print(f"❌ System initialization failed: {e}")
            return False
    
    def _apply_config_overrides(self, overrides: Dict[str, Any]):
        """Apply configuration overrides"""
        for key, value in overrides.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                print(f"   🔧 Applied override: {key} = {value}")
    
    def process_detection(self, detection_data: Dict[str, Any], 
                         frame: np.ndarray = None) -> UnifiedDetectionResult:
        """Process detection through all ML systems"""
        
        if not self.is_initialized:
            raise RuntimeError("System not initialized. Call initialize_system() first.")
        
        start_time = time.time()
        timestamp = time.time()
        
        try:
            # Extract basic detection information
            camera_id = detection_data.get('camera_id', 0)
            people_count = detection_data.get('people_count', 0)
            density = detection_data.get('density', 0.0)
            confidence = detection_data.get('confidence', 0.5)
            
            # Initialize result
            result = UnifiedDetectionResult(
                timestamp=timestamp,
                camera_id=camera_id,
                people_count=people_count,
                density=density,
                confidence=confidence,
                adaptive_threshold=0.15,  # Default
                anomaly_score=0.0,
                anomaly_type="normal",
                behavior_classification="normal_walking",
                panic_score=0.0,
                predicted_density_5min=density,
                predicted_density_10min=density,
                predicted_density_15min=density,
                reid_global_id=0,
                smart_alert_level="safe",
                environmental_impact={},
                overall_risk_score=0.0,
                risk_level="low",
                recommended_actions=[],
                ml_confidence=0.5,
                system_confidence=0.5
            )
            
            # Process through each ML component
            component_results = {}
            
            # 1. Adaptive Threshold Optimization
            if self.adaptive_threshold_optimizer:
                try:
                    comp_start = time.time()
                    
                    # Create environmental factors for threshold optimization
                    env_factors = self._create_threshold_environmental_factors(detection_data)
                    detection_context = self._create_detection_context(detection_data)
                    
                    optimal_threshold = self.adaptive_threshold_optimizer.calculate_optimal_threshold(
                        env_factors, detection_context
                    )
                    
                    result.adaptive_threshold = optimal_threshold
                    component_results['adaptive_thresholds'] = {
                        'threshold': optimal_threshold,
                        'confidence': self.adaptive_threshold_optimizer.model_accuracy
                    }
                    
                    self.processing_times['adaptive_thresholds'].append(time.time() - comp_start)
                    
                except Exception as e:
                    print(f"⚠️ Adaptive threshold processing error: {e}")
            
            # 2. Anomaly Detection
            if self.anomaly_detector:
                try:
                    comp_start = time.time()
                    
                    # Create crowd pattern from detection data
                    crowd_pattern = self._create_crowd_pattern(detection_data)
                    
                    anomaly_result = self.anomaly_detector.detect_anomaly(crowd_pattern)
                    
                    result.anomaly_score = anomaly_result.anomaly_score
                    result.anomaly_type = anomaly_result.anomaly_type
                    component_results['anomaly_detection'] = {
                        'score': anomaly_result.anomaly_score,
                        'type': anomaly_result.anomaly_type,
                        'confidence': anomaly_result.confidence
                    }
                    
                    self.processing_times['anomaly_detection'].append(time.time() - comp_start)
                    
                except Exception as e:
                    print(f"⚠️ Anomaly detection processing error: {e}")
            
            # 3. Behavior Analysis
            if self.behavior_analyzer:
                try:
                    comp_start = time.time()
                    
                    # Create behavior pattern from detection data
                    behavior_pattern = self._create_behavior_pattern(detection_data)
                    
                    behavior_result = self.behavior_analyzer.classify_behavior(behavior_pattern)
                    
                    result.behavior_classification = behavior_result.behavior_type
                    result.panic_score = behavior_result.panic_score
                    component_results['behavior_analysis'] = {
                        'classification': behavior_result.behavior_type,
                        'panic_score': behavior_result.panic_score,
                        'confidence': behavior_result.confidence
                    }
                    
                    self.processing_times['behavior_analysis'].append(time.time() - comp_start)
                    
                except Exception as e:
                    print(f"⚠️ Behavior analysis processing error: {e}")
            
            # 4. Density Forecasting
            if self.density_forecaster:
                try:
                    comp_start = time.time()
                    
                    # Create density record
                    density_record = self._create_density_record(detection_data)
                    self.density_forecaster.add_density_record(density_record)
                    
                    # Get forecasts
                    forecast_5min = self.density_forecaster.predict_density(timestamp, 5)
                    forecast_10min = self.density_forecaster.predict_density(timestamp, 10)
                    forecast_15min = self.density_forecaster.predict_density(timestamp, 15)
                    
                    result.predicted_density_5min = forecast_5min.predicted_density
                    result.predicted_density_10min = forecast_10min.predicted_density
                    result.predicted_density_15min = forecast_15min.predicted_density
                    
                    component_results['density_forecasting'] = {
                        'forecast_5min': forecast_5min.predicted_density,
                        'forecast_10min': forecast_10min.predicted_density,
                        'forecast_15min': forecast_15min.predicted_density,
                        'confidence': forecast_5min.confidence
                    }
                    
                    self.processing_times['density_forecasting'].append(time.time() - comp_start)
                    
                except Exception as e:
                    print(f"⚠️ Density forecasting processing error: {e}")
            
            # 5. Person Re-identification
            if self.person_reidentifier and frame is not None:
                try:
                    comp_start = time.time()
                    
                    # Create person detections from detection data
                    person_detections = self._create_person_detections(detection_data, camera_id)
                    
                    reid_results = []
                    for detection in person_detections:
                        reid_result = self.person_reidentifier.reidentify_person(detection, frame)
                        reid_results.append(reid_result)
                    
                    # Use most confident re-id result
                    if reid_results:
                        best_reid = max(reid_results, key=lambda x: x.confidence)
                        result.reid_global_id = best_reid.global_id
                    
                    component_results['person_reid'] = {
                        'global_id': result.reid_global_id,
                        'confidence': max([r.confidence for r in reid_results]) if reid_results else 0.0
                    }
                    
                    self.processing_times['person_reid'].append(time.time() - comp_start)
                    
                except Exception as e:
                    print(f"⚠️ Person re-identification processing error: {e}")
            
            # 6. Smart Alert Learning
            if self.smart_alert_learner:
                try:
                    comp_start = time.time()
                    
                    # Create alert context
                    alert_context = self._create_alert_context(detection_data)
                    
                    # Get optimal thresholds
                    optimal_thresholds = self.smart_alert_learner.get_optimal_thresholds(alert_context)
                    
                    # Evaluate current conditions
                    evaluation = self.smart_alert_learner.evaluate_threshold_performance(
                        alert_context, density, people_count, 0.5, result.panic_score
                    )
                    
                    result.smart_alert_level = evaluation.risk_assessment
                    component_results['smart_alerts'] = {
                        'alert_level': evaluation.risk_assessment,
                        'confidence': evaluation.confidence
                    }
                    
                    self.processing_times['smart_alerts'].append(time.time() - comp_start)
                    
                except Exception as e:
                    print(f"⚠️ Smart alert processing error: {e}")
            
            # 7. Environmental Integration
            if self.environmental_integrator:
                try:
                    comp_start = time.time()
                    
                    # Get environmental factors
                    environmental_factors = self.environmental_integrator.simulate_environmental_factors()
                    
                    # Calculate environmental impact
                    environmental_impact = self.environmental_integrator.calculate_environmental_impact(
                        environmental_factors
                    )
                    
                    # Apply environmental impact
                    base_values = {
                        'density': density,
                        'movement_intensity': 0.5,
                        'panic_threshold': 0.8,
                        'risk_score': 0.0,
                        'evacuation_time': 300
                    }
                    
                    modified_values = self.environmental_integrator.apply_environmental_impact(
                        base_values, environmental_impact
                    )
                    
                    result.environmental_impact = modified_values
                    component_results['environmental'] = {
                        'impact': environmental_impact,
                        'modified_values': modified_values
                    }
                    
                    self.processing_times['environmental'].append(time.time() - comp_start)
                    
                except Exception as e:
                    print(f"⚠️ Environmental integration processing error: {e}")
            
            # 8. Unified Risk Assessment
            result.overall_risk_score = self._calculate_unified_risk_score(component_results)
            result.risk_level = self._determine_risk_level(result.overall_risk_score)
            result.recommended_actions = self._generate_recommendations(component_results, result)
            
            # 9. Confidence Calculation
            result.ml_confidence = self._calculate_ml_confidence(component_results)
            result.system_confidence = self._calculate_system_confidence(result)
            
            # Store result
            self.detection_history.append(result)
            
            # Update performance metrics
            total_processing_time = time.time() - start_time
            self.processing_times['total'].append(total_processing_time)
            
            # Update system health
            self._update_system_health()
            
            return result
            
        except Exception as e:
            print(f"❌ Unified processing error: {e}")
            # Return fallback result
            return self._create_fallback_result(detection_data, timestamp)
    
    def _create_threshold_environmental_factors(self, detection_data: Dict[str, Any]) -> ThresholdEnvFactors:
        """Create environmental factors for threshold optimization"""
        return ThresholdEnvFactors(
            lighting_condition=detection_data.get('lighting_condition', 0.8),
            weather_condition=detection_data.get('weather_condition', 0.5),
            time_of_day=detection_data.get('time_of_day', 0.5),
            crowd_density=detection_data.get('density', 0.0),
            camera_angle=detection_data.get('camera_angle', 0.7),
            image_quality=detection_data.get('image_quality', 0.8),
            motion_blur=detection_data.get('motion_blur', 0.0),
            occlusion_level=detection_data.get('occlusion_level', 0.0)
        )
    
    def _create_detection_context(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create detection context for ML components"""
        return {
            'frame_resolution': detection_data.get('frame_resolution', 1280),
            'fps': detection_data.get('fps', 30),
            'processing_time': detection_data.get('processing_time', 0.033),
            'gpu_memory_usage': detection_data.get('gpu_memory_usage', 0.5),
            'temperature': detection_data.get('temperature', 25.0),
            'humidity': detection_data.get('humidity', 50.0),
            'wind_speed': detection_data.get('wind_speed', 0.0),
            'event_type': detection_data.get('event_type', 0),
            'venue_capacity': detection_data.get('venue_capacity', 1000),
            'current_capacity_ratio': detection_data.get('density', 0.0) / 10.0,
            'hour_of_day': datetime.now().hour,
            'day_of_week': datetime.now().weekday(),
            'month': datetime.now().month,
            'is_holiday': 0,
            'is_weekend': 1 if datetime.now().weekday() >= 5 else 0,
        }
    
    def _create_crowd_pattern(self, detection_data: Dict[str, Any]) -> CrowdPattern:
        """Create crowd pattern for anomaly detection"""
        return CrowdPattern(
            timestamp=time.time(),
            people_count=detection_data.get('people_count', 0),
            density=detection_data.get('density', 0.0),
            flow_intensity=detection_data.get('flow_intensity', 0.5),
            movement_direction="mixed",
            spatial_distribution=[detection_data.get('density', 0.0)] * 16,
            velocity_vectors=[(0.1, 0.1)] * (detection_data.get('people_count', 0) // 4),
            acceleration_pattern=detection_data.get('acceleration_pattern', 0.0),
            clustering_coefficient=detection_data.get('clustering_coefficient', 0.5),
            entropy=detection_data.get('entropy', 0.5)
        )
    
    def _create_behavior_pattern(self, detection_data: Dict[str, Any]) -> BehaviorPattern:
        """Create behavior pattern for behavior analysis"""
        return BehaviorPattern(
            timestamp=time.time(),
            people_count=detection_data.get('people_count', 0),
            movement_vectors=[],
            average_speed=detection_data.get('average_speed', 1.0),
            speed_variance=detection_data.get('speed_variance', 0.2),
            direction_consistency=detection_data.get('direction_consistency', 0.7),
            acceleration_pattern=detection_data.get('acceleration_pattern', 0.0),
            clustering_level=detection_data.get('clustering_level', 0.5),
            dispersion_level=detection_data.get('dispersion_level', 0.3),
            panic_indicators={
                'high_speed': 1.0 if detection_data.get('average_speed', 0) > 2.0 else 0.0,
                'direction_change': 1.0 - detection_data.get('direction_consistency', 0.7),
                'acceleration_spike': detection_data.get('acceleration_pattern', 0.0),
                'clustering_breakdown': 1.0 - detection_data.get('clustering_level', 0.5),
                'dispersion_increase': detection_data.get('dispersion_level', 0.3),
                'movement_irregularity': detection_data.get('speed_variance', 0.2)
            }
        )
    
    def _create_density_record(self, detection_data: Dict[str, Any]) -> DensityRecord:
        """Create density record for forecasting"""
        return DensityRecord(
            timestamp=time.time(),
            people_count=detection_data.get('people_count', 0),
            density=detection_data.get('density', 0.0),
            area_m2=detection_data.get('area_m2', 25.0),
            confidence=detection_data.get('confidence', 0.8),
            environmental_factors={
                'temperature': 25.0,
                'humidity': 50.0,
                'weather_condition': 0.5,
                'lighting_condition': 0.8,
                'wind_speed': 0.0,
                'precipitation': 0.0,
                'visibility': 1.0,
                'movement_intensity': 0.5,
                'spatial_distribution': 0.5,
                'clustering_level': 0.5,
            },
            event_context={
                'event_type': 0,
                'event_duration': 120,
                'venue_capacity': 1000,
                'capacity_ratio': detection_data.get('density', 0.0) / 10.0,
                'event_popularity': 0.5,
                'ticket_price_level': 0.5,
                'special_occasion': False,
            }
        )
    
    def _create_person_detections(self, detection_data: Dict[str, Any], camera_id: int) -> List[PersonDetection]:
        """Create person detections for re-identification"""
        detections = []
        people_count = detection_data.get('people_count', 0)
        
        for i in range(min(people_count, 10)):  # Limit to 10 detections
            detection = PersonDetection(
                id=i,
                camera_id=camera_id,
                timestamp=time.time(),
                bbox=(i * 50, i * 50, 30, 60),  # Simulated bounding box
                center=(i * 50 + 15, i * 50 + 30),
                confidence=detection_data.get('confidence', 0.8)
            )
            detections.append(detection)
        
        return detections
    
    def _create_alert_context(self, detection_data: Dict[str, Any]) -> AlertContext:
        """Create alert context for smart alert learning"""
        return AlertContext(
            venue_id=f"venue_{detection_data.get('camera_id', 0)}",
            venue_type="stadium",
            event_type="sports",
            time_of_day=datetime.now().hour,
            day_of_week=datetime.now().weekday(),
            season="summer",
            weather_condition="clear",
            lighting_condition=0.8,
            crowd_demographics={'adults': 0.7, 'children': 0.1, 'elderly': 0.2},
            historical_incidents=0,
            venue_capacity=1000,
            current_capacity_ratio=detection_data.get('density', 0.0) / 10.0,
            emergency_exits=5,
            security_personnel=20,
            crowd_management_measures=['barriers', 'signage']
        )
    
    def _calculate_unified_risk_score(self, component_results: Dict[str, Any]) -> float:
        """Calculate unified risk score from all components"""
        
        risk_factors = []
        
        # Anomaly detection risk
        if 'anomaly_detection' in component_results:
            anomaly_score = component_results['anomaly_detection']['score']
            risk_factors.append(('anomaly', abs(anomaly_score), 0.2))
        
        # Behavior analysis risk
        if 'behavior_analysis' in component_results:
            panic_score = component_results['behavior_analysis']['panic_score']
            risk_factors.append(('panic', panic_score, 0.25))
        
        # Density forecasting risk
        if 'density_forecasting' in component_results:
            forecast_5min = component_results['density_forecasting']['forecast_5min']
            if forecast_5min > 6.0:  # High density threshold
                risk_factors.append(('density', (forecast_5min - 6.0) / 4.0, 0.2))
        
        # Environmental risk
        if 'environmental' in component_results:
            env_impact = component_results['environmental']['impact']
            risk_factors.append(('environmental', env_impact.risk_score_modifier - 1.0, 0.15))
        
        # Smart alert risk
        if 'smart_alerts' in component_results:
            alert_level = component_results['smart_alerts']['alert_level']
            risk_values = {'low': 0.0, 'medium': 0.3, 'high': 0.6, 'critical': 1.0}
            risk_factors.append(('alert', risk_values.get(alert_level, 0.0), 0.2))
        
        # Calculate weighted risk score
        total_weight = sum(weight for _, _, weight in risk_factors)
        if total_weight == 0:
            return 0.0
        
        weighted_risk = sum(risk * weight for _, risk, weight in risk_factors)
        return min(1.0, weighted_risk / total_weight)
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level from risk score"""
        if risk_score >= 0.8:
            return "critical"
        elif risk_score >= 0.6:
            return "high"
        elif risk_score >= 0.4:
            return "medium"
        elif risk_score >= 0.2:
            return "low"
        else:
            return "minimal"
    
    def _generate_recommendations(self, component_results: Dict[str, Any], 
                                result: UnifiedDetectionResult) -> List[str]:
        """Generate recommendations based on all components"""
        
        recommendations = []
        
        # Risk-based recommendations
        if result.risk_level == "critical":
            recommendations.extend([
                "CRITICAL RISK: Evacuate immediately",
                "Call emergency services",
                "Implement emergency protocols"
            ])
        elif result.risk_level == "high":
            recommendations.extend([
                "HIGH RISK: Increase security personnel",
                "Prepare evacuation procedures",
                "Monitor crowd closely"
            ])
        elif result.risk_level == "medium":
            recommendations.extend([
                "MEDIUM RISK: Monitor crowd density",
                "Prepare crowd control measures",
                "Increase staff presence"
            ])
        
        # Component-specific recommendations
        if 'anomaly_detection' in component_results:
            anomaly_type = component_results['anomaly_detection']['type']
            if anomaly_type in ['stampede_risk', 'panic_running']:
                recommendations.append("Anomaly detected: Investigate immediately")
        
        if 'behavior_analysis' in component_results:
            behavior = component_results['behavior_analysis']['classification']
            if behavior in ['panic_running', 'chaotic_movement']:
                recommendations.append("Panic behavior detected: Implement crowd control")
        
        if 'density_forecasting' in component_results:
            forecast_5min = component_results['density_forecasting']['forecast_5min']
            if forecast_5min > 6.0:
                recommendations.append("High density predicted: Prepare for crowd surge")
        
        if 'environmental' in component_results:
            env_recommendations = self.environmental_integrator.get_environmental_recommendations(
                component_results['environmental']['impact']
            )
            recommendations.extend(env_recommendations[:3])  # Limit to top 3
        
        return list(set(recommendations))  # Remove duplicates
    
    def _calculate_ml_confidence(self, component_results: Dict[str, Any]) -> float:
        """Calculate ML confidence from component results"""
        
        confidences = []
        weights = []
        
        for component, results in component_results.items():
            if 'confidence' in results:
                confidences.append(results['confidence'])
                weights.append(self.component_weights.get(component, 0.1))
        
        if not confidences:
            return 0.5
        
        # Weighted average
        total_weight = sum(weights)
        if total_weight == 0:
            return np.mean(confidences)
        
        weighted_confidence = sum(c * w for c, w in zip(confidences, weights)) / total_weight
        return weighted_confidence
    
    def _calculate_system_confidence(self, result: UnifiedDetectionResult) -> float:
        """Calculate overall system confidence"""
        
        # Base confidence from detection
        base_confidence = result.confidence
        
        # ML confidence
        ml_confidence = result.ml_confidence
        
        # Component availability factor
        available_components = sum([
            self.config.enable_adaptive_thresholds,
            self.config.enable_anomaly_detection,
            self.config.enable_behavior_analysis,
            self.config.enable_density_forecasting,
            self.config.enable_person_reid,
            self.config.enable_smart_alerts,
            self.config.enable_environmental_integration
        ])
        
        availability_factor = available_components / 7.0
        
        # System health factor
        health_factor = self.system_health
        
        # Combined confidence
        system_confidence = (base_confidence * 0.3 + 
                           ml_confidence * 0.4 + 
                           availability_factor * 0.2 + 
                           health_factor * 0.1)
        
        return min(1.0, max(0.0, system_confidence))
    
    def _create_fallback_result(self, detection_data: Dict[str, Any], timestamp: float) -> UnifiedDetectionResult:
        """Create fallback result when ML processing fails"""
        
        return UnifiedDetectionResult(
            timestamp=timestamp,
            camera_id=detection_data.get('camera_id', 0),
            people_count=detection_data.get('people_count', 0),
            density=detection_data.get('density', 0.0),
            confidence=detection_data.get('confidence', 0.5),
            adaptive_threshold=0.15,
            anomaly_score=0.0,
            anomaly_type="normal",
            behavior_classification="normal_walking",
            panic_score=0.0,
            predicted_density_5min=detection_data.get('density', 0.0),
            predicted_density_10min=detection_data.get('density', 0.0),
            predicted_density_15min=detection_data.get('density', 0.0),
            reid_global_id=0,
            smart_alert_level="safe",
            environmental_impact={},
            overall_risk_score=0.0,
            risk_level="low",
            recommended_actions=["Continue monitoring"],
            ml_confidence=0.3,
            system_confidence=0.3
        )
    
    def _update_system_health(self):
        """Update system health based on performance metrics"""
        
        # Calculate average processing times
        avg_processing_times = {}
        for component, times in self.processing_times.items():
            if times:
                avg_processing_times[component] = np.mean(times[-10:])  # Last 10 measurements
        
        # Health based on processing performance
        if 'total' in avg_processing_times:
            total_time = avg_processing_times['total']
            if total_time < 0.1:  # Very fast
                health_factor = 1.0
            elif total_time < 0.5:  # Fast
                health_factor = 0.9
            elif total_time < 1.0:  # Acceptable
                health_factor = 0.8
            elif total_time < 2.0:  # Slow
                health_factor = 0.6
            else:  # Very slow
                health_factor = 0.4
            
            self.system_health = health_factor
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        stats = {
            'system_status': {
                'is_initialized': self.is_initialized,
                'is_running': self.is_running,
                'system_health': self.system_health,
                'last_update_time': self.last_update_time
            },
            'component_status': {
                'adaptive_thresholds': self.adaptive_threshold_optimizer is not None,
                'anomaly_detection': self.anomaly_detector is not None,
                'behavior_analysis': self.behavior_analyzer is not None,
                'density_forecasting': self.density_forecaster is not None,
                'person_reid': self.person_reidentifier is not None,
                'smart_alerts': self.smart_alert_learner is not None,
                'crowd_simulation': self.crowd_simulator is not None,
                'environmental_integration': self.environmental_integrator is not None
            },
            'performance_metrics': {
                'detection_history_size': len(self.detection_history),
                'alert_history_size': len(self.alert_history),
                'average_processing_time': np.mean(self.processing_times.get('total', [0])),
                'component_weights': self.component_weights
            },
            'configuration': {
                'processing_mode': self.config.processing_mode,
                'update_frequency': self.config.update_frequency,
                'confidence_threshold': self.config.confidence_threshold,
                'fusion_method': self.config.fusion_method,
                'fallback_mode': self.config.fallback_mode
            }
        }
        
        # Add component-specific statistics
        if self.adaptive_threshold_optimizer:
            stats['adaptive_thresholds'] = self.adaptive_threshold_optimizer.get_performance_stats()
        
        if self.anomaly_detector:
            stats['anomaly_detection'] = self.anomaly_detector.get_anomaly_statistics()
        
        if self.behavior_analyzer:
            stats['behavior_analysis'] = self.behavior_analyzer.get_performance_stats()
        
        if self.density_forecaster:
            stats['density_forecasting'] = self.density_forecaster.get_forecast_statistics()
        
        if self.person_reidentifier:
            stats['person_reid'] = self.person_reidentifier.get_track_statistics()
        
        if self.smart_alert_learner:
            stats['smart_alerts'] = self.smart_alert_learner.get_learning_statistics()
        
        if self.environmental_integrator:
            stats['environmental_integration'] = self.environmental_integrator.get_integration_statistics()
        
        return stats
    
    def save_system_state(self, file_path: str = "models/integrated_ml_system.pkl"):
        """Save system state and models"""
        
        import joblib
        
        system_state = {
            'config': self.config,
            'component_weights': self.component_weights,
            'system_health': self.system_health,
            'performance_metrics': dict(self.performance_metrics),
            'timestamp': time.time()
        }
        
        joblib.dump(system_state, file_path)
        print(f"✅ Integrated ML system state saved to {file_path}")
    
    def load_system_state(self, file_path: str = "models/integrated_ml_system.pkl"):
        """Load system state and models"""
        
        import joblib
        
        try:
            if os.path.exists(file_path):
                system_state = joblib.load(file_path)
                
                self.config = system_state.get('config', self.config)
                self.component_weights = system_state.get('component_weights', self.component_weights)
                self.system_health = system_state.get('system_health', 1.0)
                self.performance_metrics = defaultdict(list, system_state.get('performance_metrics', {}))
                
                print(f"✅ Integrated ML system state loaded from {file_path}")
                return True
        except Exception as e:
            print(f"⚠️ Failed to load system state: {e}")
        
        return False

# Example usage and testing
if __name__ == "__main__":
    # Initialize integrated ML system
    print("🚀 Initializing Integrated ML System...")
    
    config = SystemConfiguration(
        enable_adaptive_thresholds=True,
        enable_anomaly_detection=True,
        enable_behavior_analysis=True,
        enable_density_forecasting=True,
        enable_person_reid=True,
        enable_smart_alerts=True,
        enable_crowd_simulation=False,  # Disable for testing
        enable_environmental_integration=True,
        processing_mode="balanced",
        update_frequency=1.0,
        confidence_threshold=0.7
    )
    
    ml_system = IntegratedMLSystem(config)
    
    # Initialize system
    if ml_system.initialize_system():
        print("✅ System initialized successfully!")
        
        # Simulate detection data
        print("\n🧪 Testing integrated ML processing...")
        
        detection_data = {
            'camera_id': 0,
            'people_count': 25,
            'density': 3.2,
            'confidence': 0.85,
            'flow_intensity': 0.6,
            'average_speed': 1.2,
            'speed_variance': 0.3,
            'direction_consistency': 0.7,
            'acceleration_pattern': 0.2,
            'clustering_coefficient': 0.6,
            'dispersion_level': 0.4,
            'area_m2': 25.0,
            'frame_resolution': 1280,
            'fps': 30,
            'processing_time': 0.033,
            'gpu_memory_usage': 0.6,
            'temperature': 25.0,
            'humidity': 60.0,
            'wind_speed': 2.0,
            'event_type': 1,
            'venue_capacity': 1000,
            'lighting_condition': 0.8,
            'weather_condition': 0.3,
            'time_of_day': 0.6,
            'camera_angle': 0.7,
            'image_quality': 0.9,
            'motion_blur': 0.1,
            'occlusion_level': 0.2
        }
        
        # Process detection
        result = ml_system.process_detection(detection_data)
        
        # Display results
        print(f"\n📊 Integrated ML Processing Results:")
        print(f"   Timestamp: {result.timestamp}")
        print(f"   Camera ID: {result.camera_id}")
        print(f"   People Count: {result.people_count}")
        print(f"   Density: {result.density:.2f} people/m²")
        print(f"   Adaptive Threshold: {result.adaptive_threshold:.3f}")
        print(f"   Anomaly Score: {result.anomaly_score:.3f} ({result.anomaly_type})")
        print(f"   Behavior: {result.behavior_classification}")
        print(f"   Panic Score: {result.panic_score:.3f}")
        print(f"   Predicted Density (5min): {result.predicted_density_5min:.2f}")
        print(f"   Predicted Density (10min): {result.predicted_density_10min:.2f}")
        print(f"   Predicted Density (15min): {result.predicted_density_15min:.2f}")
        print(f"   ReID Global ID: {result.reid_global_id}")
        print(f"   Smart Alert Level: {result.smart_alert_level}")
        print(f"   Overall Risk Score: {result.overall_risk_score:.3f}")
        print(f"   Risk Level: {result.risk_level}")
        print(f"   ML Confidence: {result.ml_confidence:.3f}")
        print(f"   System Confidence: {result.system_confidence:.3f}")
        
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(result.recommended_actions, 1):
            print(f"   {i}. {rec}")
        
        # Get system statistics
        stats = ml_system.get_system_statistics()
        print(f"\n📈 System Statistics:")
        print(f"   System Health: {stats['system_status']['system_health']:.3f}")
        print(f"   Detection History: {stats['performance_metrics']['detection_history_size']} records")
        print(f"   Average Processing Time: {stats['performance_metrics']['average_processing_time']:.3f}s")
        print(f"   Component Weights: {stats['performance_metrics']['component_weights']}")
        
        # Save system state
        ml_system.save_system_state()
        
    else:
        print("❌ System initialization failed!")
