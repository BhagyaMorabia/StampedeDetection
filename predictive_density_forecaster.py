"""
Predictive Crowd Density Forecasting System for STAMPede Detection
Uses time series forecasting and machine learning to predict crowd density 5-15 minutes ahead
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import joblib
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced time series libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.exponential_smoothing import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("⚠️ statsmodels not available - using basic forecasting")

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available - using scikit-learn models")

@dataclass
class ForecastConfig:
    """Configuration for forecasting system"""
    prediction_horizons: List[int] = None  # [5, 10, 15] minutes
    lookback_window: int = 60  # minutes of historical data
    feature_window: int = 10  # minutes for feature extraction
    update_frequency: int = 5  # minutes between model updates
    min_samples: int = 100  # minimum samples for training
    confidence_threshold: float = 0.7  # minimum confidence for predictions
    
    def __post_init__(self):
        if self.prediction_horizons is None:
            self.prediction_horizons = [5, 10, 15]  # minutes

@dataclass
class DensityRecord:
    """Record of crowd density at a specific time"""
    timestamp: float
    people_count: int
    density: float
    area_m2: float
    confidence: float
    environmental_factors: Dict[str, float]
    event_context: Dict[str, Any]

@dataclass
class ForecastResult:
    """Result of density forecasting"""
    timestamp: float
    prediction_horizon: int  # minutes ahead
    predicted_density: float
    predicted_people_count: int
    confidence: float
    prediction_interval: Tuple[float, float]  # lower, upper bounds
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    trend_strength: float
    risk_assessment: str
    recommended_actions: List[str]

class CrowdDensityForecaster:
    """Advanced crowd density forecasting system"""
    
    def __init__(self, config: Optional[ForecastConfig] = None):
        self.config = config or ForecastConfig()
        self.models = {}  # Different models for different horizons
        self.scalers = {}
        self.is_trained = False
        
        # Data storage
        self.density_history = deque(maxlen=self.config.lookback_window * 60)  # Store by seconds
        self.forecast_history = deque(maxlen=1000)
        
        # Performance tracking
        self.forecast_accuracy = {}
        self.model_performance = {}
        self.last_update_time = time.time()
        
        # Feature engineering
        self.feature_names = []
        self._setup_feature_names()
        
        # Create model directory
        os.makedirs("models", exist_ok=True)
    
    def _setup_feature_names(self):
        """Setup feature names for the forecasting model"""
        self.feature_names = [
            # Historical density features
            'density_current', 'density_1min_ago', 'density_5min_ago', 'density_10min_ago',
            'density_trend_5min', 'density_trend_10min', 'density_trend_30min',
            'density_volatility_5min', 'density_volatility_15min',
            'density_moving_avg_5min', 'density_moving_avg_15min', 'density_moving_avg_30min',
            
            # People count features
            'people_current', 'people_1min_ago', 'people_5min_ago', 'people_10min_ago',
            'people_trend_5min', 'people_trend_10min', 'people_trend_30min',
            'people_volatility_5min', 'people_volatility_15min',
            'people_moving_avg_5min', 'people_moving_avg_15min', 'people_moving_avg_30min',
            
            # Temporal features
            'hour_of_day', 'minute_of_hour', 'day_of_week', 'day_of_month', 'month',
            'is_weekend', 'is_holiday', 'is_peak_hour', 'is_night_time',
            
            # Environmental features
            'temperature', 'humidity', 'weather_condition', 'lighting_condition',
            'wind_speed', 'precipitation', 'visibility',
            
            # Event context features
            'event_type', 'event_duration', 'venue_capacity', 'capacity_ratio',
            'event_popularity', 'ticket_price_level', 'special_occasion',
            
            # Derived features
            'density_acceleration', 'people_acceleration', 'crowd_pressure',
            'movement_intensity', 'spatial_distribution', 'clustering_level'
        ]
    
    def extract_features(self, current_time: float, prediction_horizon: int) -> np.ndarray:
        """Extract features for forecasting"""
        features = []
        
        # Get recent density records
        recent_records = self._get_recent_records(current_time, self.config.feature_window)
        
        if len(recent_records) < 5:
            # Insufficient data - return default features
            return np.zeros(len(self.feature_names))
        
        # Historical density features
        current_density = recent_records[-1].density
        features.extend([
            current_density,
            self._get_density_at_time(current_time - 60, recent_records),
            self._get_density_at_time(current_time - 300, recent_records),
            self._get_density_at_time(current_time - 600, recent_records),
            self._calculate_trend(recent_records, 5),
            self._calculate_trend(recent_records, 10),
            self._calculate_trend(recent_records, 30),
            self._calculate_volatility(recent_records, 5),
            self._calculate_volatility(recent_records, 15),
            self._calculate_moving_average(recent_records, 5),
            self._calculate_moving_average(recent_records, 15),
            self._calculate_moving_average(recent_records, 30),
        ])
        
        # People count features
        current_people = recent_records[-1].people_count
        features.extend([
            current_people,
            self._get_people_at_time(current_time - 60, recent_records),
            self._get_people_at_time(current_time - 300, recent_records),
            self._get_people_at_time(current_time - 600, recent_records),
            self._calculate_people_trend(recent_records, 5),
            self._calculate_people_trend(recent_records, 10),
            self._calculate_people_trend(recent_records, 30),
            self._calculate_people_volatility(recent_records, 5),
            self._calculate_people_volatility(recent_records, 15),
            self._calculate_people_moving_average(recent_records, 5),
            self._calculate_people_moving_average(recent_records, 15),
            self._calculate_people_moving_average(recent_records, 30),
        ])
        
        # Temporal features
        dt = datetime.fromtimestamp(current_time)
        features.extend([
            dt.hour,
            dt.minute,
            dt.weekday(),
            dt.day,
            dt.month,
            1 if dt.weekday() >= 5 else 0,  # is_weekend
            0,  # is_holiday (simplified)
            1 if 7 <= dt.hour <= 9 or 17 <= dt.hour <= 19 else 0,  # is_peak_hour
            1 if 22 <= dt.hour or dt.hour <= 6 else 0,  # is_night_time
        ])
        
        # Environmental features (from most recent record)
        env_factors = recent_records[-1].environmental_factors
        features.extend([
            env_factors.get('temperature', 25.0),
            env_factors.get('humidity', 50.0),
            env_factors.get('weather_condition', 0.5),
            env_factors.get('lighting_condition', 0.8),
            env_factors.get('wind_speed', 0.0),
            env_factors.get('precipitation', 0.0),
            env_factors.get('visibility', 1.0),
        ])
        
        # Event context features
        event_context = recent_records[-1].event_context
        features.extend([
            event_context.get('event_type', 0),
            event_context.get('event_duration', 0),
            event_context.get('venue_capacity', 1000),
            event_context.get('capacity_ratio', 0.1),
            event_context.get('event_popularity', 0.5),
            event_context.get('ticket_price_level', 0.5),
            event_context.get('special_occasion', 0),
        ])
        
        # Derived features
        features.extend([
            self._calculate_acceleration(recent_records, 'density'),
            self._calculate_acceleration(recent_records, 'people'),
            current_density * current_people,  # crowd_pressure
            env_factors.get('movement_intensity', 0.5),
            env_factors.get('spatial_distribution', 0.5),
            env_factors.get('clustering_level', 0.5),
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _get_recent_records(self, current_time: float, window_minutes: int) -> List[DensityRecord]:
        """Get recent density records within the specified window"""
        cutoff_time = current_time - (window_minutes * 60)
        return [record for record in self.density_history if record.timestamp >= cutoff_time]
    
    def _get_density_at_time(self, target_time: float, records: List[DensityRecord]) -> float:
        """Get density at a specific time (interpolated if needed)"""
        if not records:
            return 0.0
        
        # Find closest records
        before_record = None
        after_record = None
        
        for record in records:
            if record.timestamp <= target_time:
                before_record = record
            elif record.timestamp > target_time and after_record is None:
                after_record = record
                break
        
        if before_record is None:
            return records[0].density
        if after_record is None:
            return before_record.density
        
        # Linear interpolation
        time_diff = after_record.timestamp - before_record.timestamp
        if time_diff == 0:
            return before_record.density
        
        weight = (target_time - before_record.timestamp) / time_diff
        return before_record.density + weight * (after_record.density - before_record.density)
    
    def _get_people_at_time(self, target_time: float, records: List[DensityRecord]) -> int:
        """Get people count at a specific time (interpolated if needed)"""
        if not records:
            return 0
        
        # Find closest records
        before_record = None
        after_record = None
        
        for record in records:
            if record.timestamp <= target_time:
                before_record = record
            elif record.timestamp > target_time and after_record is None:
                after_record = record
                break
        
        if before_record is None:
            return records[0].people_count
        if after_record is None:
            return before_record.people_count
        
        # Linear interpolation
        time_diff = after_record.timestamp - before_record.timestamp
        if time_diff == 0:
            return before_record.people_count
        
        weight = (target_time - before_record.timestamp) / time_diff
        interpolated = before_record.people_count + weight * (after_record.people_count - before_record.people_count)
        return int(round(interpolated))
    
    def _calculate_trend(self, records: List[DensityRecord], window_minutes: int) -> float:
        """Calculate density trend over specified window"""
        if len(records) < 2:
            return 0.0
        
        window_seconds = window_minutes * 60
        cutoff_time = records[-1].timestamp - window_seconds
        
        window_records = [r for r in records if r.timestamp >= cutoff_time]
        if len(window_records) < 2:
            return 0.0
        
        # Linear regression to find trend
        times = [(r.timestamp - window_records[0].timestamp) / 60 for r in window_records]  # minutes
        densities = [r.density for r in window_records]
        
        if len(times) < 2:
            return 0.0
        
        # Simple linear trend calculation
        n = len(times)
        sum_x = sum(times)
        sum_y = sum(densities)
        sum_xy = sum(t * d for t, d in zip(times, densities))
        sum_x2 = sum(t * t for t in times)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope
    
    def _calculate_people_trend(self, records: List[DensityRecord], window_minutes: int) -> float:
        """Calculate people count trend over specified window"""
        if len(records) < 2:
            return 0.0
        
        window_seconds = window_minutes * 60
        cutoff_time = records[-1].timestamp - window_seconds
        
        window_records = [r for r in records if r.timestamp >= cutoff_time]
        if len(window_records) < 2:
            return 0.0
        
        # Linear regression to find trend
        times = [(r.timestamp - window_records[0].timestamp) / 60 for r in window_records]  # minutes
        people_counts = [r.people_count for r in window_records]
        
        if len(times) < 2:
            return 0.0
        
        # Simple linear trend calculation
        n = len(times)
        sum_x = sum(times)
        sum_y = sum(people_counts)
        sum_xy = sum(t * p for t, p in zip(times, people_counts))
        sum_x2 = sum(t * t for t in times)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope
    
    def _calculate_volatility(self, records: List[DensityRecord], window_minutes: int) -> float:
        """Calculate density volatility over specified window"""
        if len(records) < 2:
            return 0.0
        
        window_seconds = window_minutes * 60
        cutoff_time = records[-1].timestamp - window_seconds
        
        window_records = [r for r in records if r.timestamp >= cutoff_time]
        if len(window_records) < 2:
            return 0.0
        
        densities = [r.density for r in window_records]
        return np.std(densities)
    
    def _calculate_people_volatility(self, records: List[DensityRecord], window_minutes: int) -> float:
        """Calculate people count volatility over specified window"""
        if len(records) < 2:
            return 0.0
        
        window_seconds = window_minutes * 60
        cutoff_time = records[-1].timestamp - window_seconds
        
        window_records = [r for r in records if r.timestamp >= cutoff_time]
        if len(window_records) < 2:
            return 0.0
        
        people_counts = [r.people_count for r in window_records]
        return np.std(people_counts)
    
    def _calculate_moving_average(self, records: List[DensityRecord], window_minutes: int) -> float:
        """Calculate density moving average over specified window"""
        if not records:
            return 0.0
        
        window_seconds = window_minutes * 60
        cutoff_time = records[-1].timestamp - window_seconds
        
        window_records = [r for r in records if r.timestamp >= cutoff_time]
        if not window_records:
            return records[-1].density
        
        densities = [r.density for r in window_records]
        return np.mean(densities)
    
    def _calculate_people_moving_average(self, records: List[DensityRecord], window_minutes: int) -> float:
        """Calculate people count moving average over specified window"""
        if not records:
            return 0.0
        
        window_seconds = window_minutes * 60
        cutoff_time = records[-1].timestamp - window_seconds
        
        window_records = [r for r in records if r.timestamp >= cutoff_time]
        if not window_records:
            return records[-1].people_count
        
        people_counts = [r.people_count for r in window_records]
        return np.mean(people_counts)
    
    def _calculate_acceleration(self, records: List[DensityRecord], metric: str) -> float:
        """Calculate acceleration (second derivative) of density or people count"""
        if len(records) < 3:
            return 0.0
        
        # Get recent values
        recent_records = records[-3:]
        
        if metric == 'density':
            values = [r.density for r in recent_records]
        else:  # people
            values = [r.people_count for r in recent_records]
        
        # Calculate second derivative (acceleration)
        if len(values) >= 3:
            # Simple finite difference approximation
            acceleration = values[2] - 2 * values[1] + values[0]
            return acceleration
        
        return 0.0
    
    def predict_density(self, current_time: float, prediction_horizon: int) -> ForecastResult:
        """Predict crowd density at specified horizon"""
        
        if not self.is_trained or len(self.density_history) < self.config.min_samples:
            # Fallback prediction based on recent trend
            return self._fallback_prediction(current_time, prediction_horizon)
        
        try:
            # Extract features
            features = self.extract_features(current_time, prediction_horizon)
            
            if prediction_horizon not in self.models:
                return self._fallback_prediction(current_time, prediction_horizon)
            
            model = self.models[prediction_horizon]
            scaler = self.scalers[prediction_horizon]
            
            # Scale features
            features_scaled = scaler.transform(features.reshape(1, -1))
            
            # Make prediction
            predicted_density = model.predict(features_scaled)[0]
            
            # Calculate confidence (simplified)
            confidence = min(0.9, max(0.1, 1.0 - abs(predicted_density - self._get_current_density()) / 10.0))
            
            # Calculate prediction interval
            prediction_std = np.sqrt(model.predict(features_scaled.reshape(1, -1))[0] * 0.1)  # Simplified
            lower_bound = max(0, predicted_density - 1.96 * prediction_std)
            upper_bound = predicted_density + 1.96 * prediction_std
            
            # Determine trend
            trend_direction, trend_strength = self._analyze_trend(current_time)
            
            # Risk assessment
            risk_assessment, recommended_actions = self._assess_risk(predicted_density, trend_direction)
            
            # Calculate predicted people count
            current_area = self._get_current_area()
            predicted_people_count = int(predicted_density * current_area)
            
            result = ForecastResult(
                timestamp=current_time,
                prediction_horizon=prediction_horizon,
                predicted_density=predicted_density,
                predicted_people_count=predicted_people_count,
                confidence=confidence,
                prediction_interval=(lower_bound, upper_bound),
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                risk_assessment=risk_assessment,
                recommended_actions=recommended_actions
            )
            
            # Store forecast
            self.forecast_history.append(result)
            
            return result
            
        except Exception as e:
            print(f"⚠️ Forecasting error: {e}")
            return self._fallback_prediction(current_time, prediction_horizon)
    
    def _fallback_prediction(self, current_time: float, prediction_horizon: int) -> ForecastResult:
        """Fallback prediction when model is not available"""
        current_density = self._get_current_density()
        current_people = self._get_current_people_count()
        
        # Simple trend-based prediction
        recent_records = list(self.density_history)[-10:]  # Last 10 records
        if len(recent_records) >= 2:
            trend = recent_records[-1].density - recent_records[0].density
            trend_per_minute = trend / len(recent_records)
            predicted_density = current_density + trend_per_minute * prediction_horizon
        else:
            predicted_density = current_density
        
        predicted_density = max(0, predicted_density)
        predicted_people_count = int(predicted_density * self._get_current_area())
        
        return ForecastResult(
            timestamp=current_time,
            prediction_horizon=prediction_horizon,
            predicted_density=predicted_density,
            predicted_people_count=predicted_people_count,
            confidence=0.3,  # Low confidence for fallback
            prediction_interval=(predicted_density * 0.8, predicted_density * 1.2),
            trend_direction="stable",
            trend_strength=0.0,
            risk_assessment="unknown",
            recommended_actions=["continue_monitoring"]
        )
    
    def _get_current_density(self) -> float:
        """Get current density from most recent record"""
        if self.density_history:
            return self.density_history[-1].density
        return 0.0
    
    def _get_current_people_count(self) -> int:
        """Get current people count from most recent record"""
        if self.density_history:
            return self.density_history[-1].people_count
        return 0
    
    def _get_current_area(self) -> float:
        """Get current area from most recent record"""
        if self.density_history:
            return self.density_history[-1].area_m2
        return 25.0  # Default area
    
    def _analyze_trend(self, current_time: float) -> Tuple[str, float]:
        """Analyze current trend direction and strength"""
        recent_records = list(self.density_history)[-20:]  # Last 20 records
        
        if len(recent_records) < 5:
            return "stable", 0.0
        
        # Calculate trend over last 10 minutes
        trend_10min = self._calculate_trend(recent_records, 10)
        
        if trend_10min > 0.1:
            return "increasing", min(1.0, trend_10min)
        elif trend_10min < -0.1:
            return "decreasing", min(1.0, abs(trend_10min))
        else:
            return "stable", 0.0
    
    def _assess_risk(self, predicted_density: float, trend_direction: str) -> Tuple[str, List[str]]:
        """Assess risk based on predicted density and trend"""
        actions = []
        
        if predicted_density > 8.0:
            risk = "critical"
            actions.extend(["evacuate_immediately", "call_emergency_services", "close_entrances"])
        elif predicted_density > 6.0:
            risk = "high"
            actions.extend(["increase_monitoring", "prepare_evacuation", "limit_entrances"])
        elif predicted_density > 4.0:
            risk = "medium"
            actions.extend(["monitor_closely", "prepare_crowd_control"])
        else:
            risk = "low"
            actions.append("continue_monitoring")
        
        # Adjust based on trend
        if trend_direction == "increasing" and predicted_density > 3.0:
            risk = "medium" if risk == "low" else risk
            actions.append("monitor_trend")
        
        return risk, actions
    
    def train_models(self):
        """Train forecasting models for different horizons"""
        try:
            if len(self.density_history) < self.config.min_samples:
                print(f"⚠️ Insufficient data for training (need {self.config.min_samples}, have {len(self.density_history)})")
                return False
            
            # Prepare training data
            training_data = self._prepare_training_data()
            
            if len(training_data) < 50:
                print("⚠️ Insufficient training samples")
                return False
            
            # Train models for each prediction horizon
            for horizon in self.config.prediction_horizons:
                print(f"🔄 Training model for {horizon}-minute horizon...")
                
                X, y = self._prepare_horizon_data(training_data, horizon)
                
                if len(X) < 20:
                    print(f"⚠️ Insufficient data for {horizon}-minute horizon")
                    continue
                
                # Split data (time series split)
                tscv = TimeSeriesSplit(n_splits=3)
                best_model = None
                best_score = -np.inf
                
                # Try different models
                models = {
                    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                    'Ridge': Ridge(alpha=1.0),
                    'LinearRegression': LinearRegression()
                }
                
                for model_name, model in models.items():
                    try:
                        scores = []
                        for train_idx, val_idx in tscv.split(X):
                            X_train, X_val = X[train_idx], X[val_idx]
                            y_train, y_val = y[train_idx], y[val_idx]
                            
                            # Scale features
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_val_scaled = scaler.transform(X_val)
                            
                            # Train model
                            model.fit(X_train_scaled, y_train)
                            
                            # Evaluate
                            y_pred = model.predict(X_val_scaled)
                            score = r2_score(y_val, y_pred)
                            scores.append(score)
                        
                        avg_score = np.mean(scores)
                        
                        if avg_score > best_score:
                            best_score = avg_score
                            best_model = model
                            best_scaler = scaler
                        
                    except Exception as e:
                        print(f"⚠️ Model {model_name} training failed: {e}")
                        continue
                
                if best_model is not None and best_score > 0.3:
                    # Final training on all data
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    best_model.fit(X_scaled, y)
                    
                    # Store model and scaler
                    self.models[horizon] = best_model
                    self.scalers[horizon] = scaler
                    self.forecast_accuracy[horizon] = best_score
                    
                    print(f"✅ {horizon}-minute model trained - R² Score: {best_score:.3f}")
            
            self.is_trained = len(self.models) > 0
            
            if self.is_trained:
                # Save models
                self._save_models()
                print(f"✅ Forecasting models trained successfully for {len(self.models)} horizons")
            
            return self.is_trained
            
        except Exception as e:
            print(f"⚠️ Model training failed: {e}")
            return False
    
    def _prepare_training_data(self) -> List[Tuple[np.ndarray, float, int]]:
        """Prepare training data from density history"""
        training_data = []
        
        for i in range(len(self.density_history) - max(self.config.prediction_horizons)):
            current_record = self.density_history[i]
            current_time = current_record.timestamp
            
            # Extract features for this time point
            features = self.extract_features(current_time, 5)  # Use 5-min horizon for feature extraction
            
            # Get actual density at different horizons
            for horizon in self.config.prediction_horizons:
                target_time = current_time + (horizon * 60)
                
                # Find actual density at target time
                actual_density = self._get_density_at_time(target_time, list(self.density_history))
                
                if actual_density > 0:  # Only use valid data points
                    training_data.append((features, actual_density, horizon))
        
        return training_data
    
    def _prepare_horizon_data(self, training_data: List[Tuple[np.ndarray, float, int]], 
                             horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for specific prediction horizon"""
        horizon_data = [(features, density) for features, density, h in training_data if h == horizon]
        
        if not horizon_data:
            return np.array([]), np.array([])
        
        X = np.array([features for features, _ in horizon_data])
        y = np.array([density for _, density in horizon_data])
        
        return X, y
    
    def _save_models(self):
        """Save trained models"""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'forecast_accuracy': self.forecast_accuracy,
            'config': self.config,
            'timestamp': time.time()
        }
        
        model_path = "models/density_forecasting_models.pkl"
        joblib.dump(model_data, model_path)
        print(f"✅ Models saved to {model_path}")
    
    def load_models(self, model_path: str = "models/density_forecasting_models.pkl"):
        """Load pre-trained forecasting models"""
        try:
            if os.path.exists(model_path):
                model_data = joblib.load(model_path)
                self.models = model_data['models']
                self.scalers = model_data['scalers']
                self.forecast_accuracy = model_data['forecast_accuracy']
                self.config = model_data['config']
                self.is_trained = True
                print(f"✅ Loaded forecasting models for {len(self.models)} horizons")
                return True
        except Exception as e:
            print(f"⚠️ Failed to load forecasting models: {e}")
        return False
    
    def add_density_record(self, record: DensityRecord):
        """Add new density record to history"""
        self.density_history.append(record)
        
        # Retrain models periodically
        if (time.time() - self.last_update_time) > (self.config.update_frequency * 60):
            self.train_models()
            self.last_update_time = time.time()
    
    def get_forecast_statistics(self) -> Dict[str, Any]:
        """Get forecasting performance statistics"""
        return {
            'is_trained': self.is_trained,
            'forecast_accuracy': self.forecast_accuracy,
            'model_count': len(self.models),
            'prediction_horizons': list(self.models.keys()),
            'density_history_size': len(self.density_history),
            'forecast_history_size': len(self.forecast_history),
            'last_update_time': self.last_update_time,
            'config': {
                'lookback_window': self.config.lookback_window,
                'feature_window': self.config.feature_window,
                'update_frequency': self.config.update_frequency,
                'min_samples': self.config.min_samples
            }
        }
    
    def simulate_density_record(self, base_time: float = None) -> DensityRecord:
        """Simulate density records for testing"""
        if base_time is None:
            base_time = time.time()
        
        # Simulate realistic density patterns
        hour = datetime.fromtimestamp(base_time).hour
        
        # Base density varies by time of day
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Peak hours
            base_density = np.random.uniform(2.0, 4.0)
        elif 10 <= hour <= 16:  # Daytime
            base_density = np.random.uniform(1.0, 3.0)
        elif 20 <= hour <= 22:  # Evening
            base_density = np.random.uniform(1.5, 3.5)
        else:  # Night/early morning
            base_density = np.random.uniform(0.1, 1.0)
        
        # Add some noise and trends
        noise = np.random.normal(0, 0.2)
        trend = np.random.uniform(-0.1, 0.1)
        
        density = max(0, base_density + noise + trend)
        people_count = int(density * 25.0)  # Assuming 25 m² area
        
        return DensityRecord(
            timestamp=base_time,
            people_count=people_count,
            density=density,
            area_m2=25.0,
            confidence=0.8 + 0.2 * np.random.random(),
            environmental_factors={
                'temperature': 20 + 10 * np.random.random(),
                'humidity': 40 + 20 * np.random.random(),
                'weather_condition': np.random.random(),
                'lighting_condition': 0.8 + 0.2 * np.random.random(),
                'wind_speed': np.random.random() * 5,
                'precipitation': np.random.random() * 0.5,
                'visibility': 0.8 + 0.2 * np.random.random(),
                'movement_intensity': np.random.random(),
                'spatial_distribution': np.random.random(),
                'clustering_level': np.random.random(),
            },
            event_context={
                'event_type': np.random.randint(0, 3),
                'event_duration': np.random.randint(60, 300),
                'venue_capacity': 1000,
                'capacity_ratio': density / 10.0,  # Normalize
                'event_popularity': np.random.random(),
                'ticket_price_level': np.random.random(),
                'special_occasion': np.random.randint(0, 2),
            }
        )

# Example usage and testing
if __name__ == "__main__":
    # Initialize forecaster
    forecaster = CrowdDensityForecaster()
    
    # Load existing models if available
    forecaster.load_models()
    
    # Simulate training data
    print("🧪 Simulating training data...")
    base_time = time.time() - 3600  # Start 1 hour ago
    
    for i in range(120):  # 120 data points (2 per minute for 1 hour)
        record = forecaster.simulate_density_record(base_time + i * 30)  # Every 30 seconds
        forecaster.add_density_record(record)
        
        if i % 20 == 0:
            print(f"📊 Generated {i+1} density records")
    
    # Train models
    print("\n🔄 Training forecasting models...")
    forecaster.train_models()
    
    # Test forecasting
    print("\n🔍 Testing density forecasting...")
    current_time = time.time()
    
    for horizon in [5, 10, 15]:
        forecast = forecaster.predict_density(current_time, horizon)
        print(f"🎯 {horizon}-minute forecast:")
        print(f"   Predicted Density: {forecast.predicted_density:.2f} people/m²")
        print(f"   Predicted People: {forecast.predicted_people_count}")
        print(f"   Confidence: {forecast.confidence:.3f}")
        print(f"   Trend: {forecast.trend_direction} (strength: {forecast.trend_strength:.3f})")
        print(f"   Risk: {forecast.risk_assessment}")
        print(f"   Actions: {forecast.recommended_actions}")
        print()
    
    # Get statistics
    stats = forecaster.get_forecast_statistics()
    print(f"📈 Forecasting Statistics:")
    print(f"   Model Trained: {stats['is_trained']}")
    print(f"   Model Count: {stats['model_count']}")
    print(f"   Prediction Horizons: {stats['prediction_horizons']}")
    print(f"   Forecast Accuracy: {stats['forecast_accuracy']}")
    print(f"   Density History: {stats['density_history_size']} records")
    print(f"   Forecast History: {stats['forecast_history_size']} forecasts")
