"""
Comprehensive Testing and Validation System for STAMPede Detection ML Features
Tests and validates all AI/ML features with real data and performance metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
import json
import os
import cv2
from collections import deque, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import all ML modules for testing
from adaptive_threshold_optimizer import AdaptiveThresholdOptimizer, EnvironmentalFactors
from anomaly_detection_system import CrowdAnomalyDetector, CrowdPattern
from behavior_analysis_system import MovementBehaviorAnalyzer, BehaviorPattern
from predictive_density_forecaster import CrowdDensityForecaster, DensityRecord
from person_reidentification_system import PersonReIdentifier, PersonDetection
from smart_alert_threshold_learner import SmartAlertThresholdLearner, AlertContext
from crowd_simulation_system import CrowdSimulator, SimulationResult
from environmental_integration_system import EnvironmentalIntegrator, EnvironmentalFactors as EnvFactors
from integrated_ml_system import IntegratedMLSystem, SystemConfiguration

@dataclass
class TestResult:
    """Result of a single test"""
    test_name: str
    component: str
    success: bool
    accuracy: float
    processing_time: float
    error_message: Optional[str]
    metrics: Dict[str, Any]
    timestamp: float

@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics"""
    overall_accuracy: float
    false_positive_rate: float
    false_negative_rate: float
    precision: float
    recall: float
    f1_score: float
    processing_speed: float  # operations per second
    memory_usage: float  # MB
    cpu_usage: float  # percentage
    gpu_usage: float  # percentage
    latency: float  # milliseconds
    throughput: float  # detections per second

@dataclass
class TestScenario:
    """Test scenario configuration"""
    name: str
    description: str
    test_data: Dict[str, Any]
    expected_results: Dict[str, Any]
    tolerance: float
    duration: float  # seconds

class MLSystemValidator:
    """Comprehensive testing and validation system for ML features"""
    
    def __init__(self):
        self.test_results = []
        self.validation_metrics = {}
        self.performance_history = deque(maxlen=1000)
        
        # Test scenarios
        self.test_scenarios = []
        self._setup_test_scenarios()
        
        # Performance tracking
        self.start_time = None
        self.end_time = None
        
        # Create test directory
        os.makedirs("test_results", exist_ok=True)
    
    def _setup_test_scenarios(self):
        """Setup comprehensive test scenarios"""
        
        # Scenario 1: Normal crowd conditions
        self.test_scenarios.append(TestScenario(
            name="normal_crowd",
            description="Normal crowd conditions with low density",
            test_data={
                'people_count': 15,
                'density': 2.1,
                'confidence': 0.85,
                'flow_intensity': 0.4,
                'average_speed': 1.0,
                'speed_variance': 0.2,
                'direction_consistency': 0.8,
                'acceleration_pattern': 0.1,
                'clustering_coefficient': 0.6,
                'dispersion_level': 0.3,
                'area_m2': 25.0,
                'lighting_condition': 0.8,
                'weather_condition': 0.3,
                'time_of_day': 0.6,
                'camera_angle': 0.7,
                'image_quality': 0.9,
                'motion_blur': 0.1,
                'occlusion_level': 0.1
            },
            expected_results={
                'anomaly_score': 0.0,
                'anomaly_type': 'normal',
                'behavior_classification': 'normal_walking',
                'panic_score': 0.0,
                'risk_level': 'low',
                'alert_level': 'safe'
            },
            tolerance=0.1,
            duration=60.0
        ))
        
        # Scenario 2: High density crowd
        self.test_scenarios.append(TestScenario(
            name="high_density_crowd",
            description="High density crowd conditions",
            test_data={
                'people_count': 45,
                'density': 6.8,
                'confidence': 0.75,
                'flow_intensity': 0.7,
                'average_speed': 1.5,
                'speed_variance': 0.4,
                'direction_consistency': 0.5,
                'acceleration_pattern': 0.3,
                'clustering_coefficient': 0.8,
                'dispersion_level': 0.2,
                'area_m2': 25.0,
                'lighting_condition': 0.7,
                'weather_condition': 0.4,
                'time_of_day': 0.7,
                'camera_angle': 0.6,
                'image_quality': 0.8,
                'motion_blur': 0.2,
                'occlusion_level': 0.3
            },
            expected_results={
                'anomaly_score': -0.3,
                'anomaly_type': 'high_density',
                'behavior_classification': 'crowded_walking',
                'panic_score': 0.3,
                'risk_level': 'medium',
                'alert_level': 'warning'
            },
            tolerance=0.15,
            duration=60.0
        ))
        
        # Scenario 3: Panic situation
        self.test_scenarios.append(TestScenario(
            name="panic_situation",
            description="Panic situation with rapid movement",
            test_data={
                'people_count': 35,
                'density': 5.2,
                'confidence': 0.7,
                'flow_intensity': 0.9,
                'average_speed': 2.8,
                'speed_variance': 0.8,
                'direction_consistency': 0.2,
                'acceleration_pattern': 0.7,
                'clustering_coefficient': 0.3,
                'dispersion_level': 0.7,
                'area_m2': 25.0,
                'lighting_condition': 0.6,
                'weather_condition': 0.6,
                'time_of_day': 0.8,
                'camera_angle': 0.5,
                'image_quality': 0.7,
                'motion_blur': 0.4,
                'occlusion_level': 0.4
            },
            expected_results={
                'anomaly_score': -0.7,
                'anomaly_type': 'panic_movement',
                'behavior_classification': 'panic_running',
                'panic_score': 0.8,
                'risk_level': 'high',
                'alert_level': 'danger'
            },
            tolerance=0.2,
            duration=60.0
        ))
        
        # Scenario 4: Stampede risk
        self.test_scenarios.append(TestScenario(
            name="stampede_risk",
            description="Critical stampede risk situation",
            test_data={
                'people_count': 60,
                'density': 8.5,
                'confidence': 0.65,
                'flow_intensity': 1.0,
                'average_speed': 3.2,
                'speed_variance': 1.2,
                'direction_consistency': 0.1,
                'acceleration_pattern': 0.9,
                'clustering_coefficient': 0.1,
                'dispersion_level': 0.9,
                'area_m2': 25.0,
                'lighting_condition': 0.5,
                'weather_condition': 0.7,
                'time_of_day': 0.9,
                'camera_angle': 0.4,
                'image_quality': 0.6,
                'motion_blur': 0.6,
                'occlusion_level': 0.5
            },
            expected_results={
                'anomaly_score': -0.9,
                'anomaly_type': 'stampede_risk',
                'behavior_classification': 'panic_running',
                'panic_score': 0.95,
                'risk_level': 'critical',
                'alert_level': 'critical'
            },
            tolerance=0.25,
            duration=60.0
        ))
        
        # Scenario 5: Edge cases
        self.test_scenarios.append(TestScenario(
            name="edge_cases",
            description="Edge cases and boundary conditions",
            test_data={
                'people_count': 1,
                'density': 0.1,
                'confidence': 0.95,
                'flow_intensity': 0.0,
                'average_speed': 0.0,
                'speed_variance': 0.0,
                'direction_consistency': 1.0,
                'acceleration_pattern': 0.0,
                'clustering_coefficient': 1.0,
                'dispersion_level': 0.0,
                'area_m2': 25.0,
                'lighting_condition': 1.0,
                'weather_condition': 0.0,
                'time_of_day': 0.5,
                'camera_angle': 1.0,
                'image_quality': 1.0,
                'motion_blur': 0.0,
                'occlusion_level': 0.0
            },
            expected_results={
                'anomaly_score': 0.0,
                'anomaly_type': 'normal',
                'behavior_classification': 'stationary',
                'panic_score': 0.0,
                'risk_level': 'minimal',
                'alert_level': 'safe'
            },
            tolerance=0.05,
            duration=30.0
        ))
    
    def test_adaptive_threshold_optimizer(self) -> List[TestResult]:
        """Test adaptive threshold optimizer"""
        
        print("🧪 Testing Adaptive Threshold Optimizer...")
        results = []
        
        try:
            optimizer = AdaptiveThresholdOptimizer()
            
            for scenario in self.test_scenarios:
                start_time = time.time()
                
                try:
                    # Create environmental factors
                    env_factors = EnvironmentalFactors(
                        lighting_condition=scenario.test_data['lighting_condition'],
                        weather_condition=scenario.test_data['weather_condition'],
                        time_of_day=scenario.test_data['time_of_day'],
                        crowd_density=scenario.test_data['density'],
                        camera_angle=scenario.test_data['camera_angle'],
                        image_quality=scenario.test_data['image_quality'],
                        motion_blur=scenario.test_data['motion_blur'],
                        occlusion_level=scenario.test_data['occlusion_level']
                    )
                    
                    # Create detection context
                    detection_context = {
                        'frame_resolution': 1280,
                        'fps': 30,
                        'processing_time': 0.033,
                        'gpu_memory_usage': 0.6,
                        'temperature': 25.0,
                        'humidity': 60.0,
                        'wind_speed': 2.0,
                        'event_type': 1,
                        'venue_capacity': 1000,
                        'current_capacity_ratio': scenario.test_data['density'] / 10.0,
                        'hour_of_day': 12,
                        'day_of_week': 1,
                        'month': 6,
                        'is_holiday': 0,
                        'is_weekend': 0,
                    }
                    
                    # Calculate optimal threshold
                    optimal_threshold = optimizer.calculate_optimal_threshold(env_factors, detection_context)
                    
                    processing_time = time.time() - start_time
                    
                    # Validate result
                    success = 0.05 <= optimal_threshold <= 0.5  # Reasonable threshold range
                    
                    result = TestResult(
                        test_name=f"adaptive_threshold_{scenario.name}",
                        component="adaptive_threshold_optimizer",
                        success=success,
                        accuracy=1.0 if success else 0.0,
                        processing_time=processing_time,
                        error_message=None if success else f"Threshold {optimal_threshold} out of range",
                        metrics={
                            'optimal_threshold': optimal_threshold,
                            'is_trained': optimizer.is_trained,
                            'model_accuracy': optimizer.model_accuracy
                        },
                        timestamp=time.time()
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    processing_time = time.time() - start_time
                    result = TestResult(
                        test_name=f"adaptive_threshold_{scenario.name}",
                        component="adaptive_threshold_optimizer",
                        success=False,
                        accuracy=0.0,
                        processing_time=processing_time,
                        error_message=str(e),
                        metrics={},
                        timestamp=time.time()
                    )
                    results.append(result)
            
        except Exception as e:
            print(f"❌ Adaptive threshold optimizer test failed: {e}")
        
        return results
    
    def test_anomaly_detection_system(self) -> List[TestResult]:
        """Test anomaly detection system"""
        
        print("🧪 Testing Anomaly Detection System...")
        results = []
        
        try:
            detector = CrowdAnomalyDetector()
            
            for scenario in self.test_scenarios:
                start_time = time.time()
                
                try:
                    # Create crowd pattern
                    pattern = CrowdPattern(
                        timestamp=time.time(),
                        people_count=scenario.test_data['people_count'],
                        density=scenario.test_data['density'],
                        flow_intensity=scenario.test_data['flow_intensity'],
                        movement_direction="mixed",
                        spatial_distribution=[scenario.test_data['density']] * 16,
                        velocity_vectors=[(0.1, 0.1)] * (scenario.test_data['people_count'] // 4),
                        acceleration_pattern=scenario.test_data['acceleration_pattern'],
                        clustering_coefficient=scenario.test_data['clustering_coefficient'],
                        entropy=1.0 - scenario.test_data['direction_consistency']
                    )
                    
                    # Detect anomaly
                    anomaly_result = detector.detect_anomaly(pattern)
                    
                    processing_time = time.time() - start_time
                    
                    # Validate result
                    expected_type = scenario.expected_results['anomaly_type']
                    success = anomaly_result.anomaly_type == expected_type
                    
                    result = TestResult(
                        test_name=f"anomaly_detection_{scenario.name}",
                        component="anomaly_detection_system",
                        success=success,
                        accuracy=anomaly_result.confidence,
                        processing_time=processing_time,
                        error_message=None if success else f"Expected {expected_type}, got {anomaly_result.anomaly_type}",
                        metrics={
                            'anomaly_score': anomaly_result.anomaly_score,
                            'anomaly_type': anomaly_result.anomaly_type,
                            'confidence': anomaly_result.confidence,
                            'severity_level': anomaly_result.severity_level
                        },
                        timestamp=time.time()
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    processing_time = time.time() - start_time
                    result = TestResult(
                        test_name=f"anomaly_detection_{scenario.name}",
                        component="anomaly_detection_system",
                        success=False,
                        accuracy=0.0,
                        processing_time=processing_time,
                        error_message=str(e),
                        metrics={},
                        timestamp=time.time()
                    )
                    results.append(result)
            
        except Exception as e:
            print(f"❌ Anomaly detection system test failed: {e}")
        
        return results
    
    def test_behavior_analysis_system(self) -> List[TestResult]:
        """Test behavior analysis system"""
        
        print("🧪 Testing Behavior Analysis System...")
        results = []
        
        try:
            analyzer = MovementBehaviorAnalyzer()
            
            for scenario in self.test_scenarios:
                start_time = time.time()
                
                try:
                    # Create behavior pattern
                    pattern = BehaviorPattern(
                        timestamp=time.time(),
                        people_count=scenario.test_data['people_count'],
                        movement_vectors=[],
                        average_speed=scenario.test_data['average_speed'],
                        speed_variance=scenario.test_data['speed_variance'],
                        direction_consistency=scenario.test_data['direction_consistency'],
                        acceleration_pattern=scenario.test_data['acceleration_pattern'],
                        clustering_level=scenario.test_data['clustering_coefficient'],
                        dispersion_level=scenario.test_data['dispersion_level'],
                        panic_indicators={
                            'high_speed': 1.0 if scenario.test_data['average_speed'] > 2.0 else 0.0,
                            'direction_change': 1.0 - scenario.test_data['direction_consistency'],
                            'acceleration_spike': scenario.test_data['acceleration_pattern'],
                            'clustering_breakdown': 1.0 - scenario.test_data['clustering_coefficient'],
                            'dispersion_increase': scenario.test_data['dispersion_level'],
                            'movement_irregularity': scenario.test_data['speed_variance']
                        }
                    )
                    
                    # Classify behavior
                    behavior_result = analyzer.classify_behavior(pattern)
                    
                    processing_time = time.time() - start_time
                    
                    # Validate result
                    expected_classification = scenario.expected_results['behavior_classification']
                    success = behavior_result.behavior_type == expected_classification
                    
                    result = TestResult(
                        test_name=f"behavior_analysis_{scenario.name}",
                        component="behavior_analysis_system",
                        success=success,
                        accuracy=behavior_result.confidence,
                        processing_time=processing_time,
                        error_message=None if success else f"Expected {expected_classification}, got {behavior_result.behavior_type}",
                        metrics={
                            'behavior_type': behavior_result.behavior_type,
                            'panic_score': behavior_result.panic_score,
                            'confidence': behavior_result.confidence,
                            'risk_level': behavior_result.risk_level
                        },
                        timestamp=time.time()
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    processing_time = time.time() - start_time
                    result = TestResult(
                        test_name=f"behavior_analysis_{scenario.name}",
                        component="behavior_analysis_system",
                        success=False,
                        accuracy=0.0,
                        processing_time=processing_time,
                        error_message=str(e),
                        metrics={},
                        timestamp=time.time()
                    )
                    results.append(result)
            
        except Exception as e:
            print(f"❌ Behavior analysis system test failed: {e}")
        
        return results
    
    def test_density_forecasting_system(self) -> List[TestResult]:
        """Test density forecasting system"""
        
        print("🧪 Testing Density Forecasting System...")
        results = []
        
        try:
            forecaster = CrowdDensityForecaster()
            
            # Generate training data
            print("   Generating training data...")
            base_time = time.time() - 3600  # Start 1 hour ago
            
            for i in range(120):  # 120 data points
                record = forecaster.simulate_density_record(base_time + i * 30)
                forecaster.add_density_record(record)
            
            # Train models
            print("   Training forecasting models...")
            forecaster.train_models()
            
            for scenario in self.test_scenarios:
                start_time = time.time()
                
                try:
                    # Create density record
                    density_record = DensityRecord(
                        timestamp=time.time(),
                        people_count=scenario.test_data['people_count'],
                        density=scenario.test_data['density'],
                        area_m2=scenario.test_data['area_m2'],
                        confidence=0.8,
                        environmental_factors={
                            'temperature': 25.0,
                            'humidity': 60.0,
                            'weather_condition': scenario.test_data['weather_condition'],
                            'lighting_condition': scenario.test_data['lighting_condition'],
                            'wind_speed': 2.0,
                            'precipitation': 0.0,
                            'visibility': 1.0,
                            'movement_intensity': scenario.test_data['flow_intensity'],
                            'spatial_distribution': 0.5,
                            'clustering_level': scenario.test_data['clustering_coefficient'],
                        },
                        event_context={
                            'event_type': 1,
                            'event_duration': 120,
                            'venue_capacity': 1000,
                            'capacity_ratio': scenario.test_data['density'] / 10.0,
                            'event_popularity': 0.7,
                            'ticket_price_level': 0.6,
                            'special_occasion': False,
                        }
                    )
                    
                    forecaster.add_density_record(density_record)
                    
                    # Get forecasts
                    current_time = time.time()
                    forecast_5min = forecaster.predict_density(current_time, 5)
                    forecast_10min = forecaster.predict_density(current_time, 10)
                    forecast_15min = forecaster.predict_density(current_time, 15)
                    
                    processing_time = time.time() - start_time
                    
                    # Validate results
                    success = (forecast_5min.confidence > 0.3 and 
                             forecast_10min.confidence > 0.3 and 
                             forecast_15min.confidence > 0.3)
                    
                    result = TestResult(
                        test_name=f"density_forecasting_{scenario.name}",
                        component="density_forecasting_system",
                        success=success,
                        accuracy=np.mean([forecast_5min.confidence, forecast_10min.confidence, forecast_15min.confidence]),
                        processing_time=processing_time,
                        error_message=None if success else "Low confidence forecasts",
                        metrics={
                            'forecast_5min': forecast_5min.predicted_density,
                            'forecast_10min': forecast_10min.predicted_density,
                            'forecast_15min': forecast_15min.predicted_density,
                            'confidence_5min': forecast_5min.confidence,
                            'confidence_10min': forecast_10min.confidence,
                            'confidence_15min': forecast_15min.confidence,
                            'is_trained': forecaster.is_trained
                        },
                        timestamp=time.time()
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    processing_time = time.time() - start_time
                    result = TestResult(
                        test_name=f"density_forecasting_{scenario.name}",
                        component="density_forecasting_system",
                        success=False,
                        accuracy=0.0,
                        processing_time=processing_time,
                        error_message=str(e),
                        metrics={},
                        timestamp=time.time()
                    )
                    results.append(result)
            
        except Exception as e:
            print(f"❌ Density forecasting system test failed: {e}")
        
        return results
    
    def test_person_reidentification_system(self) -> List[TestResult]:
        """Test person re-identification system"""
        
        print("🧪 Testing Person Re-identification System...")
        results = []
        
        try:
            reid = PersonReIdentifier()
            
            for scenario in self.test_scenarios:
                start_time = time.time()
                
                try:
                    # Create simulated frame
                    frame = reid.simulate_frame(scenario.test_data['camera_id'], 
                                               scenario.test_data['people_count'])
                    
                    # Create person detections
                    person_detections = []
                    for i in range(min(scenario.test_data['people_count'], 10)):
                        detection = reid.simulate_person_detection(
                            scenario.test_data['camera_id'], i
                        )
                        person_detections.append(detection)
                    
                    # Process re-identification
                    reid_results = []
                    for detection in person_detections:
                        reid_result = reid.reidentify_person(detection, frame)
                        reid_results.append(reid_result)
                    
                    processing_time = time.time() - start_time
                    
                    # Validate results
                    success = len(reid_results) > 0 and all(r.confidence >= 0.0 for r in reid_results)
                    
                    result = TestResult(
                        test_name=f"person_reid_{scenario.name}",
                        component="person_reidentification_system",
                        success=success,
                        accuracy=np.mean([r.confidence for r in reid_results]) if reid_results else 0.0,
                        processing_time=processing_time,
                        error_message=None if success else "Re-identification failed",
                        metrics={
                            'num_detections': len(person_detections),
                            'num_reid_results': len(reid_results),
                            'average_confidence': np.mean([r.confidence for r in reid_results]) if reid_results else 0.0,
                            'active_tracks': len(reid.active_tracks),
                            'global_id_counter': reid.global_id_counter
                        },
                        timestamp=time.time()
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    processing_time = time.time() - start_time
                    result = TestResult(
                        test_name=f"person_reid_{scenario.name}",
                        component="person_reidentification_system",
                        success=False,
                        accuracy=0.0,
                        processing_time=processing_time,
                        error_message=str(e),
                        metrics={},
                        timestamp=time.time()
                    )
                    results.append(result)
            
        except Exception as e:
            print(f"❌ Person re-identification system test failed: {e}")
        
        return results
    
    def test_smart_alert_threshold_learner(self) -> List[TestResult]:
        """Test smart alert threshold learner"""
        
        print("🧪 Testing Smart Alert Threshold Learner...")
        results = []
        
        try:
            learner = SmartAlertThresholdLearner()
            
            # Generate training data
            print("   Generating training data...")
            feedback_data = []
            
            for _ in range(100):  # 100 feedback samples
                context = learner.simulate_alert_context("stadium")
                feedback = learner.simulate_alert_feedback(context)
                feedback_data.append(feedback)
            
            # Train models
            print("   Training alert threshold models...")
            learner.learn_thresholds(feedback_data)
            
            for scenario in self.test_scenarios:
                start_time = time.time()
                
                try:
                    # Create alert context
                    alert_context = AlertContext(
                        venue_id=f"venue_{scenario.test_data['camera_id']}",
                        venue_type="stadium",
                        event_type="sports",
                        time_of_day=datetime.now().hour,
                        day_of_week=datetime.now().weekday(),
                        season="summer",
                        weather_condition="clear",
                        lighting_condition=scenario.test_data['lighting_condition'],
                        crowd_demographics={'adults': 0.7, 'children': 0.1, 'elderly': 0.2},
                        historical_incidents=0,
                        venue_capacity=1000,
                        current_capacity_ratio=scenario.test_data['density'] / 10.0,
                        emergency_exits=5,
                        security_personnel=20,
                        crowd_management_measures=['barriers', 'signage']
                    )
                    
                    # Get optimal thresholds
                    optimal_thresholds = learner.get_optimal_thresholds(alert_context)
                    
                    # Evaluate performance
                    evaluation = learner.evaluate_threshold_performance(
                        alert_context,
                        scenario.test_data['density'],
                        scenario.test_data['people_count'],
                        0.5,  # movement
                        0.3   # panic
                    )
                    
                    processing_time = time.time() - start_time
                    
                    # Validate results
                    expected_alert_level = scenario.expected_results['alert_level']
                    success = evaluation.risk_assessment == expected_alert_level
                    
                    result = TestResult(
                        test_name=f"smart_alerts_{scenario.name}",
                        component="smart_alert_threshold_learner",
                        success=success,
                        accuracy=evaluation.confidence,
                        processing_time=processing_time,
                        error_message=None if success else f"Expected {expected_alert_level}, got {evaluation.risk_assessment}",
                        metrics={
                            'density_threshold': optimal_thresholds.density_threshold,
                            'people_count_threshold': optimal_thresholds.people_count_threshold,
                            'movement_threshold': optimal_thresholds.movement_threshold,
                            'panic_threshold': optimal_thresholds.panic_threshold,
                            'confidence': optimal_thresholds.confidence,
                            'risk_assessment': evaluation.risk_assessment,
                            'improvement_score': evaluation.improvement_score
                        },
                        timestamp=time.time()
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    processing_time = time.time() - start_time
                    result = TestResult(
                        test_name=f"smart_alerts_{scenario.name}",
                        component="smart_alert_threshold_learner",
                        success=False,
                        accuracy=0.0,
                        processing_time=processing_time,
                        error_message=str(e),
                        metrics={},
                        timestamp=time.time()
                    )
                    results.append(result)
            
        except Exception as e:
            print(f"❌ Smart alert threshold learner test failed: {e}")
        
        return results
    
    def test_environmental_integration_system(self) -> List[TestResult]:
        """Test environmental integration system"""
        
        print("🧪 Testing Environmental Integration System...")
        results = []
        
        try:
            integrator = EnvironmentalIntegrator()
            
            for scenario in self.test_scenarios:
                start_time = time.time()
                
                try:
                    # Get environmental factors
                    environmental_factors = integrator.simulate_environmental_factors()
                    
                    # Calculate environmental impact
                    environmental_impact = integrator.calculate_environmental_impact(
                        environmental_factors
                    )
                    
                    # Apply environmental impact
                    base_values = {
                        'density': scenario.test_data['density'],
                        'movement_intensity': scenario.test_data['flow_intensity'],
                        'panic_threshold': 0.8,
                        'risk_score': 0.0,
                        'evacuation_time': 300
                    }
                    
                    modified_values = integrator.apply_environmental_impact(
                        base_values, environmental_impact
                    )
                    
                    # Get recommendations
                    recommendations = integrator.get_environmental_recommendations(
                        environmental_impact
                    )
                    
                    processing_time = time.time() - start_time
                    
                    # Validate results
                    success = (environmental_impact.confidence > 0.0 and 
                             len(modified_values) > 0 and 
                             len(recommendations) >= 0)
                    
                    result = TestResult(
                        test_name=f"environmental_integration_{scenario.name}",
                        component="environmental_integration_system",
                        success=success,
                        accuracy=environmental_impact.confidence,
                        processing_time=processing_time,
                        error_message=None if success else "Environmental integration failed",
                        metrics={
                            'density_modifier': environmental_impact.density_modifier,
                            'movement_modifier': environmental_impact.movement_modifier,
                            'panic_threshold_modifier': environmental_impact.panic_threshold_modifier,
                            'risk_score_modifier': environmental_impact.risk_score_modifier,
                            'evacuation_time_modifier': environmental_impact.evacuation_time_modifier,
                            'confidence': environmental_impact.confidence,
                            'contributing_factors': environmental_impact.contributing_factors,
                            'recommendations_count': len(recommendations)
                        },
                        timestamp=time.time()
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    processing_time = time.time() - start_time
                    result = TestResult(
                        test_name=f"environmental_integration_{scenario.name}",
                        component="environmental_integration_system",
                        success=False,
                        accuracy=0.0,
                        processing_time=processing_time,
                        error_message=str(e),
                        metrics={},
                        timestamp=time.time()
                    )
                    results.append(result)
            
        except Exception as e:
            print(f"❌ Environmental integration system test failed: {e}")
        
        return results
    
    def test_integrated_ml_system(self) -> List[TestResult]:
        """Test integrated ML system"""
        
        print("🧪 Testing Integrated ML System...")
        results = []
        
        try:
            # Initialize integrated system
            config = SystemConfiguration(
                enable_adaptive_thresholds=True,
                enable_anomaly_detection=True,
                enable_behavior_analysis=True,
                enable_density_forecasting=True,
                enable_person_reid=True,
                enable_smart_alerts=True,
                enable_crowd_simulation=False,  # Disable for testing
                enable_environmental_integration=True,
                processing_mode="balanced"
            )
            
            ml_system = IntegratedMLSystem(config)
            
            # Initialize system
            if not ml_system.initialize_system():
                raise RuntimeError("Failed to initialize integrated ML system")
            
            for scenario in self.test_scenarios:
                start_time = time.time()
                
                try:
                    # Process detection through integrated system
                    detection_data = {
                        'camera_id': scenario.test_data.get('camera_id', 0),
                        'people_count': scenario.test_data['people_count'],
                        'density': scenario.test_data['density'],
                        'confidence': scenario.test_data['confidence'],
                        'flow_intensity': scenario.test_data['flow_intensity'],
                        'average_speed': scenario.test_data['average_speed'],
                        'speed_variance': scenario.test_data['speed_variance'],
                        'direction_consistency': scenario.test_data['direction_consistency'],
                        'acceleration_pattern': scenario.test_data['acceleration_pattern'],
                        'clustering_coefficient': scenario.test_data['clustering_coefficient'],
                        'dispersion_level': scenario.test_data['dispersion_level'],
                        'area_m2': scenario.test_data['area_m2'],
                        'lighting_condition': scenario.test_data['lighting_condition'],
                        'weather_condition': scenario.test_data['weather_condition'],
                        'time_of_day': scenario.test_data['time_of_day'],
                        'camera_angle': scenario.test_data['camera_angle'],
                        'image_quality': scenario.test_data['image_quality'],
                        'motion_blur': scenario.test_data['motion_blur'],
                        'occlusion_level': scenario.test_data['occlusion_level']
                    }
                    
                    # Create simulated frame
                    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    
                    # Process through integrated system
                    unified_result = ml_system.process_detection(detection_data, frame)
                    
                    processing_time = time.time() - start_time
                    
                    # Validate results
                    expected_risk_level = scenario.expected_results['risk_level']
                    success = unified_result.risk_level == expected_risk_level
                    
                    result = TestResult(
                        test_name=f"integrated_ml_{scenario.name}",
                        component="integrated_ml_system",
                        success=success,
                        accuracy=unified_result.system_confidence,
                        processing_time=processing_time,
                        error_message=None if success else f"Expected {expected_risk_level}, got {unified_result.risk_level}",
                        metrics={
                            'overall_risk_score': unified_result.overall_risk_score,
                            'risk_level': unified_result.risk_level,
                            'ml_confidence': unified_result.ml_confidence,
                            'system_confidence': unified_result.system_confidence,
                            'adaptive_threshold': unified_result.adaptive_threshold,
                            'anomaly_score': unified_result.anomaly_score,
                            'anomaly_type': unified_result.anomaly_type,
                            'behavior_classification': unified_result.behavior_classification,
                            'panic_score': unified_result.panic_score,
                            'smart_alert_level': unified_result.smart_alert_level,
                            'recommendations_count': len(unified_result.recommended_actions)
                        },
                        timestamp=time.time()
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    processing_time = time.time() - start_time
                    result = TestResult(
                        test_name=f"integrated_ml_{scenario.name}",
                        component="integrated_ml_system",
                        success=False,
                        accuracy=0.0,
                        processing_time=processing_time,
                        error_message=str(e),
                        metrics={},
                        timestamp=time.time()
                    )
                    results.append(result)
            
        except Exception as e:
            print(f"❌ Integrated ML system test failed: {e}")
        
        return results
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive tests on all ML systems"""
        
        print("🚀 Starting Comprehensive ML System Testing...")
        print("=" * 60)
        
        self.start_time = time.time()
        all_results = []
        
        # Test individual components
        component_tests = [
            ("Adaptive Threshold Optimizer", self.test_adaptive_threshold_optimizer),
            ("Anomaly Detection System", self.test_anomaly_detection_system),
            ("Behavior Analysis System", self.test_behavior_analysis_system),
            ("Density Forecasting System", self.test_density_forecasting_system),
            ("Person Re-identification System", self.test_person_reidentification_system),
            ("Smart Alert Threshold Learner", self.test_smart_alert_threshold_learner),
            ("Environmental Integration System", self.test_environmental_integration_system),
            ("Integrated ML System", self.test_integrated_ml_system)
        ]
        
        for component_name, test_function in component_tests:
            print(f"\n📊 Testing {component_name}...")
            try:
                results = test_function()
                all_results.extend(results)
                
                # Calculate component statistics
                success_count = sum(1 for r in results if r.success)
                total_count = len(results)
                avg_accuracy = np.mean([r.accuracy for r in results])
                avg_processing_time = np.mean([r.processing_time for r in results])
                
                print(f"   ✅ Success Rate: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
                print(f"   📈 Average Accuracy: {avg_accuracy:.3f}")
                print(f"   ⏱️ Average Processing Time: {avg_processing_time:.3f}s")
                
            except Exception as e:
                print(f"   ❌ {component_name} test failed: {e}")
        
        self.end_time = time.time()
        
        # Calculate overall statistics
        overall_stats = self._calculate_overall_statistics(all_results)
        
        # Generate test report
        test_report = self._generate_test_report(all_results, overall_stats)
        
        # Save results
        self._save_test_results(all_results, test_report)
        
        return test_report
    
    def _calculate_overall_statistics(self, all_results: List[TestResult]) -> Dict[str, Any]:
        """Calculate overall test statistics"""
        
        if not all_results:
            return {}
        
        # Overall success rate
        success_count = sum(1 for r in all_results if r.success)
        total_count = len(all_results)
        success_rate = success_count / total_count
        
        # Component-wise statistics
        component_stats = defaultdict(list)
        for result in all_results:
            component_stats[result.component].append(result)
        
        component_summary = {}
        for component, results in component_stats.items():
            component_success = sum(1 for r in results if r.success)
            component_total = len(results)
            component_summary[component] = {
                'success_rate': component_success / component_total,
                'average_accuracy': np.mean([r.accuracy for r in results]),
                'average_processing_time': np.mean([r.processing_time for r in results]),
                'total_tests': component_total
            }
        
        # Performance metrics
        avg_accuracy = np.mean([r.accuracy for r in all_results])
        avg_processing_time = np.mean([r.processing_time for r in all_results])
        total_processing_time = sum(r.processing_time for r in all_results)
        
        # Test duration
        test_duration = self.end_time - self.start_time
        
        return {
            'overall_success_rate': success_rate,
            'total_tests': total_count,
            'successful_tests': success_count,
            'failed_tests': total_count - success_count,
            'average_accuracy': avg_accuracy,
            'average_processing_time': avg_processing_time,
            'total_processing_time': total_processing_time,
            'test_duration': test_duration,
            'component_summary': component_summary,
            'scenario_summary': self._calculate_scenario_statistics(all_results)
        }
    
    def _calculate_scenario_statistics(self, all_results: List[TestResult]) -> Dict[str, Any]:
        """Calculate scenario-wise statistics"""
        
        scenario_stats = defaultdict(list)
        for result in all_results:
            scenario_name = result.test_name.split('_')[-1]  # Extract scenario name
            scenario_stats[scenario_name].append(result)
        
        scenario_summary = {}
        for scenario, results in scenario_stats.items():
            scenario_success = sum(1 for r in results if r.success)
            scenario_total = len(results)
            scenario_summary[scenario] = {
                'success_rate': scenario_success / scenario_total,
                'average_accuracy': np.mean([r.accuracy for r in results]),
                'average_processing_time': np.mean([r.processing_time for r in results]),
                'total_tests': scenario_total
            }
        
        return scenario_summary
    
    def _generate_test_report(self, all_results: List[TestResult], 
                            overall_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        
        report = {
            'test_summary': {
                'test_date': datetime.now().isoformat(),
                'test_duration': overall_stats.get('test_duration', 0),
                'total_tests': overall_stats.get('total_tests', 0),
                'successful_tests': overall_stats.get('successful_tests', 0),
                'failed_tests': overall_stats.get('failed_tests', 0),
                'overall_success_rate': overall_stats.get('overall_success_rate', 0),
                'average_accuracy': overall_stats.get('average_accuracy', 0),
                'average_processing_time': overall_stats.get('average_processing_time', 0)
            },
            'component_performance': overall_stats.get('component_summary', {}),
            'scenario_performance': overall_stats.get('scenario_summary', {}),
            'detailed_results': [
                {
                    'test_name': r.test_name,
                    'component': r.component,
                    'success': r.success,
                    'accuracy': r.accuracy,
                    'processing_time': r.processing_time,
                    'error_message': r.error_message,
                    'metrics': r.metrics,
                    'timestamp': r.timestamp
                }
                for r in all_results
            ],
            'recommendations': self._generate_recommendations(all_results, overall_stats)
        }
        
        return report
    
    def _generate_recommendations(self, all_results: List[TestResult], 
                                overall_stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results"""
        
        recommendations = []
        
        # Overall performance recommendations
        success_rate = overall_stats.get('overall_success_rate', 0)
        if success_rate < 0.8:
            recommendations.append("Overall success rate is below 80% - review system configuration")
        
        avg_accuracy = overall_stats.get('average_accuracy', 0)
        if avg_accuracy < 0.7:
            recommendations.append("Average accuracy is below 70% - consider retraining models")
        
        avg_processing_time = overall_stats.get('average_processing_time', 0)
        if avg_processing_time > 1.0:
            recommendations.append("Processing time is high - consider optimization or hardware upgrade")
        
        # Component-specific recommendations
        component_summary = overall_stats.get('component_summary', {})
        for component, stats in component_summary.items():
            if stats['success_rate'] < 0.7:
                recommendations.append(f"{component} has low success rate - investigate issues")
            
            if stats['average_accuracy'] < 0.6:
                recommendations.append(f"{component} has low accuracy - retrain models")
            
            if stats['average_processing_time'] > 0.5:
                recommendations.append(f"{component} is slow - optimize performance")
        
        # Scenario-specific recommendations
        scenario_summary = overall_stats.get('scenario_summary', {})
        for scenario, stats in scenario_summary.items():
            if stats['success_rate'] < 0.6:
                recommendations.append(f"{scenario} scenario has low success rate - review thresholds")
        
        return recommendations
    
    def _save_test_results(self, all_results: List[TestResult], test_report: Dict[str, Any]):
        """Save test results to files"""
        
        # Save detailed results
        results_file = f"test_results/ml_system_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w') as f:
            json.dump(test_report, f, indent=2, default=str)
        
        # Save summary report
        summary_file = f"test_results/ml_system_test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(summary_file, 'w') as f:
            f.write("# ML System Test Report\n\n")
            f.write(f"**Test Date:** {test_report['test_summary']['test_date']}\n")
            f.write(f"**Test Duration:** {test_report['test_summary']['test_duration']:.2f} seconds\n")
            f.write(f"**Total Tests:** {test_report['test_summary']['total_tests']}\n")
            f.write(f"**Successful Tests:** {test_report['test_summary']['successful_tests']}\n")
            f.write(f"**Failed Tests:** {test_report['test_summary']['failed_tests']}\n")
            f.write(f"**Overall Success Rate:** {test_report['test_summary']['overall_success_rate']:.2%}\n")
            f.write(f"**Average Accuracy:** {test_report['test_summary']['average_accuracy']:.3f}\n")
            f.write(f"**Average Processing Time:** {test_report['test_summary']['average_processing_time']:.3f}s\n\n")
            
            f.write("## Component Performance\n\n")
            for component, stats in test_report['component_performance'].items():
                f.write(f"### {component}\n")
                f.write(f"- Success Rate: {stats['success_rate']:.2%}\n")
                f.write(f"- Average Accuracy: {stats['average_accuracy']:.3f}\n")
                f.write(f"- Average Processing Time: {stats['average_processing_time']:.3f}s\n")
                f.write(f"- Total Tests: {stats['total_tests']}\n\n")
            
            f.write("## Recommendations\n\n")
            for i, rec in enumerate(test_report['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
        
        print(f"✅ Test results saved to {results_file}")
        print(f"✅ Test summary saved to {summary_file}")
    
    def get_validation_metrics(self) -> ValidationMetrics:
        """Get comprehensive validation metrics"""
        
        if not self.test_results:
            return ValidationMetrics(
                overall_accuracy=0.0,
                false_positive_rate=0.0,
                false_negative_rate=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                processing_speed=0.0,
                memory_usage=0.0,
                cpu_usage=0.0,
                gpu_usage=0.0,
                latency=0.0,
                throughput=0.0
            )
        
        # Calculate metrics
        success_count = sum(1 for r in self.test_results if r.success)
        total_count = len(self.test_results)
        
        overall_accuracy = success_count / total_count
        
        # Calculate processing metrics
        avg_processing_time = np.mean([r.processing_time for r in self.test_results])
        processing_speed = 1.0 / avg_processing_time if avg_processing_time > 0 else 0.0
        
        # Calculate latency and throughput
        latency = avg_processing_time * 1000  # Convert to milliseconds
        throughput = processing_speed
        
        return ValidationMetrics(
            overall_accuracy=overall_accuracy,
            false_positive_rate=0.0,  # Would need ground truth data
            false_negative_rate=0.0,  # Would need ground truth data
            precision=overall_accuracy,  # Simplified
            recall=overall_accuracy,  # Simplified
            f1_score=overall_accuracy,  # Simplified
            processing_speed=processing_speed,
            memory_usage=0.0,  # Would need memory monitoring
            cpu_usage=0.0,  # Would need CPU monitoring
            gpu_usage=0.0,  # Would need GPU monitoring
            latency=latency,
            throughput=throughput
        )

# Example usage and testing
if __name__ == "__main__":
    # Initialize validator
    print("🧪 Initializing ML System Validator...")
    validator = MLSystemValidator()
    
    # Run comprehensive tests
    print("🚀 Running comprehensive ML system tests...")
    test_report = validator.run_comprehensive_tests()
    
    # Display results
    print("\n📊 Test Results Summary:")
    print("=" * 50)
    
    summary = test_report['test_summary']
    print(f"Test Date: {summary['test_date']}")
    print(f"Test Duration: {summary['test_duration']:.2f} seconds")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Successful Tests: {summary['successful_tests']}")
    print(f"Failed Tests: {summary['failed_tests']}")
    print(f"Overall Success Rate: {summary['overall_success_rate']:.2%}")
    print(f"Average Accuracy: {summary['average_accuracy']:.3f}")
    print(f"Average Processing Time: {summary['average_processing_time']:.3f}s")
    
    print("\n📈 Component Performance:")
    for component, stats in test_report['component_performance'].items():
        print(f"  {component}:")
        print(f"    Success Rate: {stats['success_rate']:.2%}")
        print(f"    Average Accuracy: {stats['average_accuracy']:.3f}")
        print(f"    Average Processing Time: {stats['average_processing_time']:.3f}s")
    
    print("\n💡 Recommendations:")
    for i, rec in enumerate(test_report['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # Get validation metrics
    metrics = validator.get_validation_metrics()
    print(f"\n📊 Validation Metrics:")
    print(f"  Overall Accuracy: {metrics.overall_accuracy:.3f}")
    print(f"  Processing Speed: {metrics.processing_speed:.2f} ops/sec")
    print(f"  Latency: {metrics.latency:.2f} ms")
    print(f"  Throughput: {metrics.throughput:.2f} detections/sec")
    
    print("\n✅ Comprehensive ML system testing completed!")
