"""
Environmental Integration System for STAMPede Detection
Integrates weather, time, event type, and other environmental factors into predictions
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

# Try to import weather API libraries
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️ Requests not available - weather API integration disabled")

@dataclass
class WeatherData:
    """Weather information"""
    temperature: float  # Celsius
    humidity: float  # 0-100%
    wind_speed: float  # m/s
    wind_direction: float  # degrees
    precipitation: float  # mm/h
    visibility: float  # km
    pressure: float  # hPa
    uv_index: float  # 0-11
    cloud_cover: float  # 0-100%
    weather_condition: str  # clear, rain, snow, fog, etc.
    timestamp: float

@dataclass
class TimeContext:
    """Time-related context"""
    hour: int  # 0-23
    minute: int  # 0-59
    day_of_week: int  # 0-6 (Monday=0)
    day_of_month: int  # 1-31
    month: int  # 1-12
    year: int
    season: str  # spring, summer, fall, winter
    is_weekend: bool
    is_holiday: bool
    is_peak_hour: bool
    is_night_time: bool
    daylight_hours: float
    sunset_hour: float
    sunrise_hour: float

@dataclass
class EventContext:
    """Event-related context"""
    event_type: str  # sports, concert, festival, exhibition, etc.
    event_duration: int  # minutes
    event_popularity: float  # 0-1 scale
    ticket_price_level: float  # 0-1 scale
    age_demographics: Dict[str, float]  # age group ratios
    expected_attendance: int
    actual_attendance: int
    venue_type: str  # stadium, concert_hall, outdoor, etc.
    capacity_ratio: float  # actual/max capacity
    special_occasion: bool
    alcohol_served: bool
    security_level: str  # low, medium, high

@dataclass
class EnvironmentalFactors:
    """Combined environmental factors"""
    weather: WeatherData
    time_context: TimeContext
    event_context: EventContext
    venue_factors: Dict[str, float]
    social_factors: Dict[str, float]
    economic_factors: Dict[str, float]

@dataclass
class EnvironmentalImpact:
    """Impact of environmental factors on crowd behavior"""
    density_modifier: float  # Multiplier for density calculations
    movement_modifier: float  # Multiplier for movement patterns
    panic_threshold_modifier: float  # Modifier for panic thresholds
    risk_score_modifier: float  # Modifier for risk assessment
    evacuation_time_modifier: float  # Modifier for evacuation time
    confidence: float  # Confidence in the impact assessment
    contributing_factors: List[str]  # Which factors are contributing most

class EnvironmentalIntegrator:
    """Advanced environmental integration system"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.weather_cache = deque(maxlen=100)
        self.environmental_history = deque(maxlen=1000)
        self.historical_patterns = {}
        
        # Weather API configuration
        self.weather_api_url = "http://api.openweathermap.org/data/2.5/weather"
        self.weather_cache_duration = 300  # 5 minutes
        
        # Environmental impact models
        self.impact_models = {}
        self.is_trained = False
        
        # Performance tracking
        self.integration_accuracy = 0.0
        self.prediction_improvement = 0.0
        
        # Create model directory
        os.makedirs("models", exist_ok=True)
    
    def get_weather_data(self, latitude: float, longitude: float) -> Optional[WeatherData]:
        """Get current weather data from API or cache"""
        
        # Check cache first
        cache_key = f"{latitude:.2f},{longitude:.2f}"
        current_time = time.time()
        
        for cached_data in self.weather_cache:
            if (cached_data['key'] == cache_key and 
                current_time - cached_data['timestamp'] < self.weather_cache_duration):
                return cached_data['weather']
        
        # Fetch from API if available
        if REQUESTS_AVAILABLE and self.api_key:
            try:
                params = {
                    'lat': latitude,
                    'lon': longitude,
                    'appid': self.api_key,
                    'units': 'metric'
                }
                
                response = requests.get(self.weather_api_url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    weather = WeatherData(
                        temperature=data['main']['temp'],
                        humidity=data['main']['humidity'],
                        wind_speed=data['wind']['speed'],
                        wind_direction=data['wind'].get('deg', 0),
                        precipitation=data.get('rain', {}).get('1h', 0),
                        visibility=data.get('visibility', 10000) / 1000,  # Convert to km
                        pressure=data['main']['pressure'],
                        uv_index=data.get('uvi', 0),
                        cloud_cover=data['clouds']['all'],
                        weather_condition=data['weather'][0]['main'].lower(),
                        timestamp=current_time
                    )
                    
                    # Cache the result
                    self.weather_cache.append({
                        'key': cache_key,
                        'weather': weather,
                        'timestamp': current_time
                    })
                    
                    return weather
                    
            except Exception as e:
                print(f"⚠️ Weather API error: {e}")
        
        # Fallback to simulated weather
        return self._simulate_weather_data()
    
    def _simulate_weather_data(self) -> WeatherData:
        """Simulate weather data for testing"""
        
        # Simulate realistic weather patterns
        hour = datetime.now().hour
        
        # Temperature varies by time of day
        base_temp = 20 + 10 * np.sin((hour - 6) * np.pi / 12)
        temperature = base_temp + np.random.normal(0, 3)
        
        # Humidity inversely related to temperature
        humidity = max(20, min(90, 80 - temperature * 2 + np.random.normal(0, 10)))
        
        # Wind speed varies
        wind_speed = np.random.exponential(2.0)
        
        # Precipitation probability
        precipitation = np.random.exponential(0.5) if np.random.random() < 0.3 else 0
        
        # Weather conditions
        conditions = ['clear', 'clouds', 'rain', 'snow', 'fog', 'mist']
        weather_condition = np.random.choice(conditions, p=[0.4, 0.3, 0.15, 0.05, 0.05, 0.05])
        
        return WeatherData(
            temperature=temperature,
            humidity=humidity,
            wind_speed=wind_speed,
            wind_direction=np.random.uniform(0, 360),
            precipitation=precipitation,
            visibility=np.random.uniform(1, 20),
            pressure=np.random.uniform(980, 1030),
            uv_index=np.random.uniform(0, 11),
            cloud_cover=np.random.uniform(0, 100),
            weather_condition=weather_condition,
            timestamp=time.time()
        )
    
    def get_time_context(self, timestamp: float = None) -> TimeContext:
        """Get time-related context"""
        
        if timestamp is None:
            timestamp = time.time()
        
        dt = datetime.fromtimestamp(timestamp)
        
        # Calculate season
        month = dt.month
        if month in [12, 1, 2]:
            season = "winter"
        elif month in [3, 4, 5]:
            season = "spring"
        elif month in [6, 7, 8]:
            season = "summer"
        else:
            season = "fall"
        
        # Calculate daylight hours (simplified)
        daylight_hours = 12 + 4 * np.sin(2 * np.pi * (dt.timetuple().tm_yday - 80) / 365)
        
        # Calculate sunset/sunrise (simplified)
        sunrise_hour = 6 + 2 * np.sin(2 * np.pi * (dt.timetuple().tm_yday - 80) / 365)
        sunset_hour = 18 - 2 * np.sin(2 * np.pi * (dt.timetuple().tm_yday - 80) / 365)
        
        return TimeContext(
            hour=dt.hour,
            minute=dt.minute,
            day_of_week=dt.weekday(),
            day_of_month=dt.day,
            month=month,
            year=dt.year,
            season=season,
            is_weekend=dt.weekday() >= 5,
            is_holiday=self._is_holiday(dt),
            is_peak_hour=self._is_peak_hour(dt.hour),
            is_night_time=dt.hour >= 22 or dt.hour <= 6,
            daylight_hours=daylight_hours,
            sunset_hour=sunset_hour,
            sunrise_hour=sunrise_hour
        )
    
    def _is_holiday(self, dt: datetime) -> bool:
        """Check if date is a holiday (simplified)"""
        # Major holidays (simplified)
        holidays = [
            (1, 1),   # New Year's Day
            (7, 4),   # Independence Day (US)
            (12, 25), # Christmas
            (12, 31), # New Year's Eve
        ]
        
        return (dt.month, dt.day) in holidays
    
    def _is_peak_hour(self, hour: int) -> bool:
        """Check if hour is peak time"""
        return hour in [7, 8, 9, 17, 18, 19]
    
    def get_event_context(self, event_data: Dict[str, Any]) -> EventContext:
        """Get event-related context"""
        
        return EventContext(
            event_type=event_data.get('event_type', 'general'),
            event_duration=event_data.get('duration', 120),
            event_popularity=event_data.get('popularity', 0.5),
            ticket_price_level=event_data.get('ticket_price_level', 0.5),
            age_demographics=event_data.get('age_demographics', {
                'children': 0.1,
                'adults': 0.7,
                'elderly': 0.2
            }),
            expected_attendance=event_data.get('expected_attendance', 1000),
            actual_attendance=event_data.get('actual_attendance', 800),
            venue_type=event_data.get('venue_type', 'general'),
            capacity_ratio=event_data.get('capacity_ratio', 0.8),
            special_occasion=event_data.get('special_occasion', False),
            alcohol_served=event_data.get('alcohol_served', False),
            security_level=event_data.get('security_level', 'medium')
        )
    
    def calculate_environmental_impact(self, environmental_factors: EnvironmentalFactors) -> EnvironmentalImpact:
        """Calculate impact of environmental factors on crowd behavior"""
        
        impact_factors = []
        
        # Weather impact
        weather_impact = self._calculate_weather_impact(environmental_factors.weather)
        impact_factors.append(weather_impact)
        
        # Time impact
        time_impact = self._calculate_time_impact(environmental_factors.time_context)
        impact_factors.append(time_impact)
        
        # Event impact
        event_impact = self._calculate_event_impact(environmental_factors.event_context)
        impact_factors.append(event_impact)
        
        # Venue impact
        venue_impact = self._calculate_venue_impact(environmental_factors.venue_factors)
        impact_factors.append(venue_impact)
        
        # Social impact
        social_impact = self._calculate_social_impact(environmental_factors.social_factors)
        impact_factors.append(social_impact)
        
        # Economic impact
        economic_impact = self._calculate_economic_impact(environmental_factors.economic_factors)
        impact_factors.append(economic_impact)
        
        # Combine impacts
        combined_impact = self._combine_impacts(impact_factors)
        
        return combined_impact
    
    def _calculate_weather_impact(self, weather: WeatherData) -> Dict[str, float]:
        """Calculate weather impact on crowd behavior"""
        
        impact = {
            'density_modifier': 1.0,
            'movement_modifier': 1.0,
            'panic_threshold_modifier': 1.0,
            'risk_score_modifier': 1.0,
            'evacuation_time_modifier': 1.0,
            'confidence': 0.8,
            'factors': []
        }
        
        # Temperature impact
        if weather.temperature < 0 or weather.temperature > 35:
            impact['density_modifier'] *= 0.8  # People avoid extreme temperatures
            impact['movement_modifier'] *= 0.9  # Slower movement
            impact['panic_threshold_modifier'] *= 0.8  # Lower panic threshold
            impact['factors'].append('extreme_temperature')
        
        # Precipitation impact
        if weather.precipitation > 5:  # Heavy rain
            impact['density_modifier'] *= 0.7  # People seek shelter
            impact['movement_modifier'] *= 0.8  # Slower movement
            impact['evacuation_time_modifier'] *= 1.3  # Longer evacuation
            impact['factors'].append('heavy_precipitation')
        
        # Wind impact
        if weather.wind_speed > 10:  # Strong wind
            impact['movement_modifier'] *= 0.9  # Affects movement
            impact['evacuation_time_modifier'] *= 1.2  # Longer evacuation
            impact['factors'].append('strong_wind')
        
        # Visibility impact
        if weather.visibility < 1:  # Poor visibility
            impact['panic_threshold_modifier'] *= 0.7  # Lower panic threshold
            impact['risk_score_modifier'] *= 1.3  # Higher risk
            impact['evacuation_time_modifier'] *= 1.4  # Much longer evacuation
            impact['factors'].append('poor_visibility')
        
        # Weather condition impact
        if weather.weather_condition in ['storm', 'thunderstorm']:
            impact['panic_threshold_modifier'] *= 0.6  # Much lower panic threshold
            impact['risk_score_modifier'] *= 1.5  # Much higher risk
            impact['factors'].append('severe_weather')
        
        return impact
    
    def _calculate_time_impact(self, time_context: TimeContext) -> Dict[str, float]:
        """Calculate time-related impact on crowd behavior"""
        
        impact = {
            'density_modifier': 1.0,
            'movement_modifier': 1.0,
            'panic_threshold_modifier': 1.0,
            'risk_score_modifier': 1.0,
            'evacuation_time_modifier': 1.0,
            'confidence': 0.9,
            'factors': []
        }
        
        # Peak hour impact
        if time_context.is_peak_hour:
            impact['density_modifier'] *= 1.3  # Higher density
            impact['panic_threshold_modifier'] *= 0.9  # Slightly lower panic threshold
            impact['factors'].append('peak_hour')
        
        # Night time impact
        if time_context.is_night_time:
            impact['panic_threshold_modifier'] *= 0.8  # Lower panic threshold
            impact['risk_score_modifier'] *= 1.2  # Higher risk
            impact['evacuation_time_modifier'] *= 1.2  # Longer evacuation
            impact['factors'].append('night_time')
        
        # Weekend impact
        if time_context.is_weekend:
            impact['density_modifier'] *= 1.2  # Higher density
            impact['movement_modifier'] *= 1.1  # More active movement
            impact['factors'].append('weekend')
        
        # Holiday impact
        if time_context.is_holiday:
            impact['density_modifier'] *= 1.4  # Much higher density
            impact['panic_threshold_modifier'] *= 0.8  # Lower panic threshold
            impact['risk_score_modifier'] *= 1.3  # Higher risk
            impact['factors'].append('holiday')
        
        # Season impact
        if time_context.season == 'summer':
            impact['density_modifier'] *= 1.1  # Higher outdoor activity
            impact['factors'].append('summer_season')
        elif time_context.season == 'winter':
            impact['density_modifier'] *= 0.9  # Lower outdoor activity
            impact['movement_modifier'] *= 0.95  # Slower movement
            impact['factors'].append('winter_season')
        
        return impact
    
    def _calculate_event_impact(self, event_context: EventContext) -> Dict[str, float]:
        """Calculate event-related impact on crowd behavior"""
        
        impact = {
            'density_modifier': 1.0,
            'movement_modifier': 1.0,
            'panic_threshold_modifier': 1.0,
            'risk_score_modifier': 1.0,
            'evacuation_time_modifier': 1.0,
            'confidence': 0.8,
            'factors': []
        }
        
        # Event type impact
        if event_context.event_type == 'concert':
            impact['density_modifier'] *= 1.2  # Higher density
            impact['movement_modifier'] *= 1.3  # More active movement
            impact['panic_threshold_modifier'] *= 0.9  # Lower panic threshold
            impact['factors'].append('concert_event')
        
        elif event_context.event_type == 'sports':
            impact['density_modifier'] *= 1.3  # Higher density
            impact['movement_modifier'] *= 1.2  # More active movement
            impact['panic_threshold_modifier'] *= 0.8  # Lower panic threshold
            impact['factors'].append('sports_event')
        
        elif event_context.event_type == 'festival':
            impact['density_modifier'] *= 1.4  # Much higher density
            impact['movement_modifier'] *= 1.4  # Much more active movement
            impact['panic_threshold_modifier'] *= 0.7  # Much lower panic threshold
            impact['risk_score_modifier'] *= 1.3  # Higher risk
            impact['factors'].append('festival_event')
        
        # Capacity impact
        if event_context.capacity_ratio > 0.9:
            impact['density_modifier'] *= 1.2  # Overcrowding
            impact['panic_threshold_modifier'] *= 0.7  # Much lower panic threshold
            impact['risk_score_modifier'] *= 1.4  # Much higher risk
            impact['factors'].append('overcrowding')
        
        # Alcohol impact
        if event_context.alcohol_served:
            impact['movement_modifier'] *= 1.2  # More erratic movement
            impact['panic_threshold_modifier'] *= 0.8  # Lower panic threshold
            impact['risk_score_modifier'] *= 1.2  # Higher risk
            impact['factors'].append('alcohol_served')
        
        # Security level impact
        if event_context.security_level == 'low':
            impact['panic_threshold_modifier'] *= 0.9  # Lower panic threshold
            impact['risk_score_modifier'] *= 1.2  # Higher risk
            impact['factors'].append('low_security')
        elif event_context.security_level == 'high':
            impact['panic_threshold_modifier'] *= 1.1  # Higher panic threshold
            impact['risk_score_modifier'] *= 0.9  # Lower risk
            impact['factors'].append('high_security')
        
        # Special occasion impact
        if event_context.special_occasion:
            impact['density_modifier'] *= 1.2  # Higher density
            impact['panic_threshold_modifier'] *= 0.9  # Lower panic threshold
            impact['factors'].append('special_occasion')
        
        return impact
    
    def _calculate_venue_impact(self, venue_factors: Dict[str, float]) -> Dict[str, float]:
        """Calculate venue-related impact on crowd behavior"""
        
        impact = {
            'density_modifier': 1.0,
            'movement_modifier': 1.0,
            'panic_threshold_modifier': 1.0,
            'risk_score_modifier': 1.0,
            'evacuation_time_modifier': 1.0,
            'confidence': 0.7,
            'factors': []
        }
        
        # Exit capacity impact
        exit_capacity = venue_factors.get('exit_capacity_ratio', 0.1)
        if exit_capacity < 0.05:  # Very low exit capacity
            impact['evacuation_time_modifier'] *= 2.0  # Much longer evacuation
            impact['panic_threshold_modifier'] *= 0.7  # Much lower panic threshold
            impact['risk_score_modifier'] *= 1.5  # Much higher risk
            impact['factors'].append('low_exit_capacity')
        
        # Obstacle density impact
        obstacle_density = venue_factors.get('obstacle_density', 0.1)
        if obstacle_density > 0.3:  # High obstacle density
            impact['movement_modifier'] *= 0.8  # Slower movement
            impact['evacuation_time_modifier'] *= 1.3  # Longer evacuation
            impact['factors'].append('high_obstacle_density')
        
        # Lighting quality impact
        lighting_quality = venue_factors.get('lighting_quality', 0.8)
        if lighting_quality < 0.5:  # Poor lighting
            impact['panic_threshold_modifier'] *= 0.8  # Lower panic threshold
            impact['risk_score_modifier'] *= 1.2  # Higher risk
            impact['evacuation_time_modifier'] *= 1.2  # Longer evacuation
            impact['factors'].append('poor_lighting')
        
        # Acoustics impact
        acoustics_quality = venue_factors.get('acoustics_quality', 0.8)
        if acoustics_quality < 0.5:  # Poor acoustics
            impact['panic_threshold_modifier'] *= 0.9  # Slightly lower panic threshold
            impact['factors'].append('poor_acoustics')
        
        return impact
    
    def _calculate_social_impact(self, social_factors: Dict[str, float]) -> Dict[str, float]:
        """Calculate social factors impact on crowd behavior"""
        
        impact = {
            'density_modifier': 1.0,
            'movement_modifier': 1.0,
            'panic_threshold_modifier': 1.0,
            'risk_score_modifier': 1.0,
            'evacuation_time_modifier': 1.0,
            'confidence': 0.6,
            'factors': []
        }
        
        # Age demographics impact
        children_ratio = social_factors.get('children_ratio', 0.1)
        elderly_ratio = social_factors.get('elderly_ratio', 0.2)
        
        if children_ratio > 0.2:  # Many children
            impact['movement_modifier'] *= 0.9  # Slower movement
            impact['panic_threshold_modifier'] *= 0.8  # Lower panic threshold
            impact['evacuation_time_modifier'] *= 1.2  # Longer evacuation
            impact['factors'].append('many_children')
        
        if elderly_ratio > 0.3:  # Many elderly
            impact['movement_modifier'] *= 0.8  # Slower movement
            impact['panic_threshold_modifier'] *= 0.9  # Lower panic threshold
            impact['evacuation_time_modifier'] *= 1.3  # Longer evacuation
            impact['factors'].append('many_elderly')
        
        # Cultural factors
        cultural_diversity = social_factors.get('cultural_diversity', 0.5)
        if cultural_diversity > 0.8:  # High diversity
            impact['panic_threshold_modifier'] *= 0.9  # Slightly lower panic threshold
            impact['factors'].append('high_cultural_diversity')
        
        # Social media influence
        social_media_influence = social_factors.get('social_media_influence', 0.5)
        if social_media_influence > 0.7:  # High social media influence
            impact['movement_modifier'] *= 1.1  # More active movement
            impact['panic_threshold_modifier'] *= 0.9  # Lower panic threshold
            impact['factors'].append('high_social_media_influence')
        
        return impact
    
    def _calculate_economic_impact(self, economic_factors: Dict[str, float]) -> Dict[str, float]:
        """Calculate economic factors impact on crowd behavior"""
        
        impact = {
            'density_modifier': 1.0,
            'movement_modifier': 1.0,
            'panic_threshold_modifier': 1.0,
            'risk_score_modifier': 1.0,
            'evacuation_time_modifier': 1.0,
            'confidence': 0.5,
            'factors': []
        }
        
        # Economic stress impact
        economic_stress = economic_factors.get('economic_stress', 0.5)
        if economic_stress > 0.7:  # High economic stress
            impact['panic_threshold_modifier'] *= 0.9  # Lower panic threshold
            impact['risk_score_modifier'] *= 1.1  # Higher risk
            impact['factors'].append('high_economic_stress')
        
        # Income level impact
        average_income = economic_factors.get('average_income_level', 0.5)
        if average_income < 0.3:  # Low income
            impact['panic_threshold_modifier'] *= 0.9  # Lower panic threshold
            impact['factors'].append('low_income')
        
        # Unemployment rate impact
        unemployment_rate = economic_factors.get('unemployment_rate', 0.1)
        if unemployment_rate > 0.15:  # High unemployment
            impact['panic_threshold_modifier'] *= 0.9  # Lower panic threshold
            impact['risk_score_modifier'] *= 1.1  # Higher risk
            impact['factors'].append('high_unemployment')
        
        return impact
    
    def _combine_impacts(self, impact_factors: List[Dict[str, float]]) -> EnvironmentalImpact:
        """Combine all environmental impacts"""
        
        # Initialize combined impact
        combined = {
            'density_modifier': 1.0,
            'movement_modifier': 1.0,
            'panic_threshold_modifier': 1.0,
            'risk_score_modifier': 1.0,
            'evacuation_time_modifier': 1.0,
            'confidence': 0.0,
            'factors': []
        }
        
        # Weighted combination
        weights = [0.25, 0.25, 0.2, 0.15, 0.1, 0.05]  # Weather, time, event, venue, social, economic
        
        for i, impact in enumerate(impact_factors):
            weight = weights[i] if i < len(weights) else 0.1
            
            combined['density_modifier'] *= impact['density_modifier'] ** weight
            combined['movement_modifier'] *= impact['movement_modifier'] ** weight
            combined['panic_threshold_modifier'] *= impact['panic_threshold_modifier'] ** weight
            combined['risk_score_modifier'] *= impact['risk_score_modifier'] ** weight
            combined['evacuation_time_modifier'] *= impact['evacuation_time_modifier'] ** weight
            
            combined['confidence'] += impact['confidence'] * weight
            combined['factors'].extend(impact['factors'])
        
        # Clamp modifiers to reasonable ranges
        combined['density_modifier'] = max(0.1, min(3.0, combined['density_modifier']))
        combined['movement_modifier'] = max(0.1, min(3.0, combined['movement_modifier']))
        combined['panic_threshold_modifier'] = max(0.1, min(2.0, combined['panic_threshold_modifier']))
        combined['risk_score_modifier'] = max(0.1, min(3.0, combined['risk_score_modifier']))
        combined['evacuation_time_modifier'] = max(0.1, min(3.0, combined['evacuation_time_modifier']))
        
        return EnvironmentalImpact(
            density_modifier=combined['density_modifier'],
            movement_modifier=combined['movement_modifier'],
            panic_threshold_modifier=combined['panic_threshold_modifier'],
            risk_score_modifier=combined['risk_score_modifier'],
            evacuation_time_modifier=combined['evacuation_time_modifier'],
            confidence=combined['confidence'],
            contributing_factors=list(set(combined['factors']))  # Remove duplicates
        )
    
    def apply_environmental_impact(self, base_values: Dict[str, float], 
                                environmental_impact: EnvironmentalImpact) -> Dict[str, float]:
        """Apply environmental impact to base values"""
        
        modified_values = {}
        
        # Apply modifiers
        modified_values['density'] = base_values.get('density', 0) * environmental_impact.density_modifier
        modified_values['movement_intensity'] = base_values.get('movement_intensity', 0) * environmental_impact.movement_modifier
        modified_values['panic_threshold'] = base_values.get('panic_threshold', 0.8) * environmental_impact.panic_threshold_modifier
        modified_values['risk_score'] = base_values.get('risk_score', 0) * environmental_impact.risk_score_modifier
        modified_values['evacuation_time'] = base_values.get('evacuation_time', 300) * environmental_impact.evacuation_time_modifier
        
        # Add confidence information
        modified_values['environmental_confidence'] = environmental_impact.confidence
        modified_values['contributing_factors'] = environmental_impact.contributing_factors
        
        return modified_values
    
    def update_historical_patterns(self, patterns: Dict[str, Any]):
        """Update the system with historical stampede patterns"""
        try:
            self.historical_patterns = patterns
            
            # Update risk weights based on historical data
            if 'venue_risk' in patterns:
                for venue, risk in patterns['venue_risk'].items():
                    self.venue_risk_weights[venue] = risk
            
            if 'event_type_risk' in patterns:
                for event_type, risk in patterns['event_type_risk'].items():
                    self.event_type_risk_weights[event_type] = risk
            
            if 'weather_risk' in patterns:
                for weather, risk in patterns['weather_risk'].items():
                    self.weather_risk_weights[weather] = risk
            
            if 'time_risk' in patterns:
                for time_period, risk in patterns['time_risk'].items():
                    self.time_risk_weights[time_period] = risk
            
            if 'crowd_size_ranges' in patterns:
                for size_range, risk in patterns['crowd_size_ranges'].items():
                    self.crowd_size_risk_weights[size_range] = risk
            
            print("✅ Historical patterns integrated into environmental system")
            return True
            
        except Exception as e:
            print(f"❌ Failed to update historical patterns: {e}")
            return False
    
    def get_environmental_recommendations(self, environmental_impact: EnvironmentalImpact) -> List[str]:
        """Get recommendations based on environmental impact"""
        
        recommendations = []
        
        # High risk recommendations
        if environmental_impact.risk_score_modifier > 1.5:
            recommendations.append("HIGH RISK: Increase security personnel immediately")
            recommendations.append("Consider implementing crowd control measures")
        
        # Weather-related recommendations
        if 'severe_weather' in environmental_impact.contributing_factors:
            recommendations.append("Severe weather detected - prepare emergency protocols")
            recommendations.append("Consider postponing outdoor events")
        
        if 'poor_visibility' in environmental_impact.contributing_factors:
            recommendations.append("Poor visibility conditions - enhance lighting")
            recommendations.append("Increase staff visibility with reflective gear")
        
        # Event-related recommendations
        if 'festival_event' in environmental_impact.contributing_factors:
            recommendations.append("Festival event - implement additional crowd management")
            recommendations.append("Increase medical staff presence")
        
        if 'overcrowding' in environmental_impact.contributing_factors:
            recommendations.append("OVERGROWDING DETECTED - implement capacity controls")
            recommendations.append("Consider opening additional exits")
        
        # Time-related recommendations
        if 'night_time' in environmental_impact.contributing_factors:
            recommendations.append("Night time event - enhance security measures")
            recommendations.append("Improve lighting in all areas")
        
        if 'holiday' in environmental_impact.contributing_factors:
            recommendations.append("Holiday event - expect higher attendance")
            recommendations.append("Implement additional crowd control measures")
        
        # Venue-related recommendations
        if 'low_exit_capacity' in environmental_impact.contributing_factors:
            recommendations.append("Low exit capacity - consider opening additional exits")
            recommendations.append("Implement staggered evacuation procedures")
        
        if 'poor_lighting' in environmental_impact.contributing_factors:
            recommendations.append("Poor lighting conditions - enhance illumination")
            recommendations.append("Install emergency lighting systems")
        
        return recommendations
    
    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get environmental integration statistics"""
        return {
            'integration_accuracy': self.integration_accuracy,
            'prediction_improvement': self.prediction_improvement,
            'weather_cache_size': len(self.weather_cache),
            'environmental_history_size': len(self.environmental_history),
            'api_available': REQUESTS_AVAILABLE,
            'api_key_configured': self.api_key is not None,
            'is_trained': self.is_trained
        }
    
    def simulate_environmental_factors(self) -> EnvironmentalFactors:
        """Simulate environmental factors for testing"""
        
        # Simulate weather
        weather = self._simulate_weather_data()
        
        # Get time context
        time_context = self.get_time_context()
        
        # Simulate event context
        event_context = EventContext(
            event_type=np.random.choice(['sports', 'concert', 'festival', 'exhibition']),
            event_duration=np.random.randint(60, 300),
            event_popularity=np.random.uniform(0.3, 1.0),
            ticket_price_level=np.random.uniform(0.2, 1.0),
            age_demographics={
                'children': np.random.uniform(0.05, 0.25),
                'adults': np.random.uniform(0.6, 0.8),
                'elderly': np.random.uniform(0.1, 0.3)
            },
            expected_attendance=np.random.randint(500, 5000),
            actual_attendance=np.random.randint(400, 4500),
            venue_type=np.random.choice(['stadium', 'concert_hall', 'outdoor', 'shopping_mall']),
            capacity_ratio=np.random.uniform(0.3, 1.0),
            special_occasion=np.random.random() < 0.3,
            alcohol_served=np.random.random() < 0.5,
            security_level=np.random.choice(['low', 'medium', 'high'])
        )
        
        # Simulate venue factors
        venue_factors = {
            'exit_capacity_ratio': np.random.uniform(0.02, 0.15),
            'obstacle_density': np.random.uniform(0.05, 0.4),
            'lighting_quality': np.random.uniform(0.3, 1.0),
            'acoustics_quality': np.random.uniform(0.4, 1.0),
            'ventilation_quality': np.random.uniform(0.5, 1.0)
        }
        
        # Simulate social factors
        social_factors = {
            'children_ratio': np.random.uniform(0.05, 0.25),
            'elderly_ratio': np.random.uniform(0.1, 0.4),
            'cultural_diversity': np.random.uniform(0.2, 0.9),
            'social_media_influence': np.random.uniform(0.3, 0.9),
            'group_size_distribution': np.random.uniform(0.1, 0.8)
        }
        
        # Simulate economic factors
        economic_factors = {
            'economic_stress': np.random.uniform(0.2, 0.8),
            'average_income_level': np.random.uniform(0.2, 0.9),
            'unemployment_rate': np.random.uniform(0.05, 0.25),
            'ticket_price_affordability': np.random.uniform(0.3, 1.0)
        }
        
        return EnvironmentalFactors(
            weather=weather,
            time_context=time_context,
            event_context=event_context,
            venue_factors=venue_factors,
            social_factors=social_factors,
            economic_factors=economic_factors
        )

# Example usage and testing
if __name__ == "__main__":
    # Initialize environmental integrator
    integrator = EnvironmentalIntegrator()
    
    # Simulate environmental factors
    print("🌍 Simulating environmental factors...")
    environmental_factors = integrator.simulate_environmental_factors()
    
    # Calculate environmental impact
    print("🔄 Calculating environmental impact...")
    impact = integrator.calculate_environmental_impact(environmental_factors)
    
    # Display results
    print(f"\n📊 Environmental Impact Analysis:")
    print(f"   Density Modifier: {impact.density_modifier:.3f}")
    print(f"   Movement Modifier: {impact.movement_modifier:.3f}")
    print(f"   Panic Threshold Modifier: {impact.panic_threshold_modifier:.3f}")
    print(f"   Risk Score Modifier: {impact.risk_score_modifier:.3f}")
    print(f"   Evacuation Time Modifier: {impact.evacuation_time_modifier:.3f}")
    print(f"   Confidence: {impact.confidence:.3f}")
    print(f"   Contributing Factors: {impact.contributing_factors}")
    
    # Apply impact to base values
    base_values = {
        'density': 3.5,
        'movement_intensity': 0.6,
        'panic_threshold': 0.8,
        'risk_score': 0.4,
        'evacuation_time': 300
    }
    
    modified_values = integrator.apply_environmental_impact(base_values, impact)
    
    print(f"\n🎯 Modified Values:")
    print(f"   Original Density: {base_values['density']:.2f} → Modified: {modified_values['density']:.2f}")
    print(f"   Original Movement: {base_values['movement_intensity']:.2f} → Modified: {modified_values['movement_intensity']:.2f}")
    print(f"   Original Panic Threshold: {base_values['panic_threshold']:.2f} → Modified: {modified_values['panic_threshold']:.2f}")
    print(f"   Original Risk Score: {base_values['risk_score']:.2f} → Modified: {modified_values['risk_score']:.2f}")
    print(f"   Original Evacuation Time: {base_values['evacuation_time']:.0f}s → Modified: {modified_values['evacuation_time']:.0f}s")
    
    # Get recommendations
    recommendations = integrator.get_environmental_recommendations(impact)
    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    # Get statistics
    stats = integrator.get_integration_statistics()
    print(f"\n📈 Integration Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test weather API (if available)
    if REQUESTS_AVAILABLE:
        print(f"\n🌤️ Testing weather data retrieval...")
        weather = integrator.get_weather_data(40.7128, -74.0060)  # New York coordinates
        if weather:
            print(f"   Temperature: {weather.temperature:.1f}°C")
            print(f"   Humidity: {weather.humidity:.1f}%")
            print(f"   Wind Speed: {weather.wind_speed:.1f} m/s")
            print(f"   Weather Condition: {weather.weather_condition}")
            print(f"   Visibility: {weather.visibility:.1f} km")
        else:
            print("   Using simulated weather data")
    else:
        print("   Weather API not available - using simulated data")
