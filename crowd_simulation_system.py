"""
Crowd Simulation & Prediction System for STAMPede Detection
Physics-based crowd simulation with agent-based modeling for stampede prediction
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

# Try to import physics simulation libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️ Matplotlib not available - simulation visualization disabled")

@dataclass
class Agent:
    """Represents a person in the crowd simulation"""
    id: int
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    target: Optional[Tuple[float, float]]
    radius: float = 0.3  # Personal space radius (meters)
    mass: float = 70.0  # Mass in kg
    max_speed: float = 2.0  # Maximum walking speed (m/s)
    panic_level: float = 0.0  # 0-1 scale
    stress_threshold: float = 0.7
    reaction_time: float = 0.5  # seconds
    age_group: str = "adult"  # child, adult, elderly
    mobility: float = 1.0  # 0-1 scale

@dataclass
class Obstacle:
    """Represents obstacles in the simulation"""
    id: int
    position: Tuple[float, float]
    size: Tuple[float, float]  # width, height
    obstacle_type: str  # wall, barrier, exit, entrance
    capacity: int = 0  # For exits/entrances

@dataclass
class SimulationEnvironment:
    """Environment for crowd simulation"""
    width: float  # meters
    height: float  # meters
    obstacles: List[Obstacle]
    exits: List[Obstacle]
    entrances: List[Obstacle]
    density_zones: List[Tuple[float, float, float]]  # x, y, density_factor
    emergency_zones: List[Tuple[float, float, float]]  # x, y, danger_level

@dataclass
class SimulationResult:
    """Result of crowd simulation"""
    timestamp: float
    agents: List[Agent]
    density_map: np.ndarray
    velocity_field: np.ndarray
    pressure_map: np.ndarray
    bottleneck_locations: List[Tuple[float, float]]
    risk_zones: List[Tuple[float, float, float]]  # x, y, risk_level
    evacuation_time: Optional[float]
    casualties_predicted: int
    simulation_metrics: Dict[str, float]

class CrowdSimulator:
    """Advanced crowd simulation system with physics-based modeling"""
    
    def __init__(self, grid_resolution: int = 50, time_step: float = 0.1):
        self.grid_resolution = grid_resolution
        self.time_step = time_step
        
        # Simulation parameters
        self.repulsion_strength = 1000.0  # Force between agents
        self.friction_coefficient = 0.1
        self.panic_amplification = 2.0
        self.density_threshold = 4.0  # people/m² for panic
        self.max_density = 8.0  # people/m² maximum
        
        # Simulation state
        self.agents: List[Agent] = []
        self.environment: Optional[SimulationEnvironment] = None
        self.simulation_time = 0.0
        self.simulation_results: List[SimulationResult] = []
        
        # Performance tracking
        self.simulation_speed = 0.0  # agents/second
        self.accuracy_metrics = {}
        
        # Visualization
        self.visualization_enabled = MATPLOTLIB_AVAILABLE
        self.fig = None
        self.ax = None
    
    def create_environment(self, width: float, height: float, 
                          obstacles: List[Obstacle] = None) -> SimulationEnvironment:
        """Create simulation environment"""
        
        if obstacles is None:
            obstacles = []
        
        # Separate exits and entrances
        exits = [obs for obs in obstacles if obs.obstacle_type == "exit"]
        entrances = [obs for obs in obstacles if obs.obstacle_type == "entrance"]
        
        # Create density zones (areas with different crowd behavior)
        density_zones = [
            (width * 0.3, height * 0.3, 1.2),  # High density zone
            (width * 0.7, height * 0.7, 0.8),  # Low density zone
        ]
        
        # Create emergency zones (areas prone to incidents)
        emergency_zones = [
            (width * 0.5, height * 0.2, 0.3),  # Moderate risk zone
            (width * 0.2, height * 0.8, 0.5),  # High risk zone
        ]
        
        self.environment = SimulationEnvironment(
            width=width,
            height=height,
            obstacles=obstacles,
            exits=exits,
            entrances=entrances,
            density_zones=density_zones,
            emergency_zones=emergency_zones
        )
        
        return self.environment
    
    def add_agents(self, num_agents: int, spawn_area: Tuple[float, float, float, float] = None):
        """Add agents to the simulation"""
        
        if self.environment is None:
            raise ValueError("Environment must be created first")
        
        if spawn_area is None:
            # Default spawn area (entire environment)
            spawn_area = (0, 0, self.environment.width, self.environment.height)
        
        spawn_x_min, spawn_y_min, spawn_x_max, spawn_y_max = spawn_area
        
        for i in range(num_agents):
            # Random position in spawn area
            x = np.random.uniform(spawn_x_min, spawn_x_max)
            y = np.random.uniform(spawn_y_min, spawn_y_max)
            
            # Random velocity
            vx = np.random.uniform(-0.5, 0.5)
            vy = np.random.uniform(-0.5, 0.5)
            
            # Random properties
            age_group = np.random.choice(['child', 'adult', 'elderly'], p=[0.15, 0.7, 0.15])
            mobility = {'child': 0.8, 'adult': 1.0, 'elderly': 0.6}[age_group]
            max_speed = {'child': 1.5, 'adult': 2.0, 'elderly': 1.2}[age_group]
            
            agent = Agent(
                id=len(self.agents),
                position=(x, y),
                velocity=(vx, vy),
                target=None,
                radius=np.random.uniform(0.25, 0.35),
                mass=np.random.uniform(60, 80),
                max_speed=max_speed,
                panic_level=np.random.uniform(0, 0.3),
                age_group=age_group,
                mobility=mobility
            )
            
            self.agents.append(agent)
    
    def set_agent_targets(self, target_strategy: str = "random_exit"):
        """Set targets for agents based on strategy"""
        
        if not self.environment or not self.environment.exits:
            return
        
        for agent in self.agents:
            if target_strategy == "random_exit":
                # Random exit
                exit_obs = np.random.choice(self.environment.exits)
                agent.target = (exit_obs.position[0], exit_obs.position[1])
            
            elif target_strategy == "nearest_exit":
                # Nearest exit
                distances = []
                for exit_obs in self.environment.exits:
                    dist = np.sqrt((agent.position[0] - exit_obs.position[0])**2 + 
                                 (agent.position[1] - exit_obs.position[1])**2)
                    distances.append(dist)
                
                nearest_exit_idx = np.argmin(distances)
                exit_obs = self.environment.exits[nearest_exit_idx]
                agent.target = (exit_obs.position[0], exit_obs.position[1])
            
            elif target_strategy == "panic_evacuation":
                # Panic evacuation (agents move away from high density areas)
                if agent.panic_level > 0.5:
                    # Move towards nearest exit
                    distances = []
                    for exit_obs in self.environment.exits:
                        dist = np.sqrt((agent.position[0] - exit_obs.position[0])**2 + 
                                     (agent.position[1] - exit_obs.position[1])**2)
                        distances.append(dist)
                    
                    nearest_exit_idx = np.argmin(distances)
                    exit_obs = self.environment.exits[nearest_exit_idx]
                    agent.target = (exit_obs.position[0], exit_obs.position[1])
                else:
                    # Random movement
                    agent.target = (np.random.uniform(0, self.environment.width),
                                  np.random.uniform(0, self.environment.height))
    
    def calculate_forces(self, agent: Agent) -> Tuple[float, float]:
        """Calculate forces acting on an agent"""
        
        fx, fy = 0.0, 0.0
        
        # Target force (desire to reach target)
        if agent.target:
            target_x, target_y = agent.target
            dx = target_x - agent.position[0]
            dy = target_y - agent.position[1]
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance > 0.1:  # Avoid division by zero
                # Normalize direction
                dx /= distance
                dy /= distance
                
                # Calculate desired speed
                desired_speed = agent.max_speed * agent.mobility
                
                # Apply panic amplification
                if agent.panic_level > 0.5:
                    desired_speed *= (1 + agent.panic_level * self.panic_amplification)
                
                fx += dx * desired_speed * 10.0  # Target force strength
                fy += dy * desired_speed * 10.0
        
        # Repulsion forces from other agents
        for other_agent in self.agents:
            if other_agent.id == agent.id:
                continue
            
            dx = agent.position[0] - other_agent.position[0]
            dy = agent.position[1] - other_agent.position[1]
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance < 2.0:  # Only consider nearby agents
                # Normalize direction
                if distance > 0.01:
                    dx /= distance
                    dy /= distance
                
                # Calculate repulsion force
                min_distance = agent.radius + other_agent.radius
                if distance < min_distance:
                    # Strong repulsion when too close
                    force_magnitude = self.repulsion_strength / (distance + 0.1)
                    fx += dx * force_magnitude
                    fy += dy * force_magnitude
        
        # Obstacle avoidance
        for obstacle in self.environment.obstacles:
            if obstacle.obstacle_type in ['wall', 'barrier']:
                # Calculate distance to obstacle
                obs_x, obs_y = obstacle.position
                obs_w, obs_h = obstacle.size
                
                # Find closest point on obstacle
                closest_x = max(obs_x - obs_w/2, min(agent.position[0], obs_x + obs_w/2))
                closest_y = max(obs_y - obs_h/2, min(agent.position[1], obs_y + obs_h/2))
                
                dx = agent.position[0] - closest_x
                dy = agent.position[1] - closest_y
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance < 1.0:  # Avoidance range
                    if distance > 0.01:
                        dx /= distance
                        dy /= distance
                    
                    force_magnitude = self.repulsion_strength * 0.5 / (distance + 0.1)
                    fx += dx * force_magnitude
                    fy += dy * force_magnitude
        
        # Density-based panic
        local_density = self._calculate_local_density(agent.position)
        if local_density > self.density_threshold:
            agent.panic_level = min(1.0, agent.panic_level + 0.1)
            
            # Panic behavior: random movement
            panic_fx = np.random.uniform(-2.0, 2.0)
            panic_fy = np.random.uniform(-2.0, 2.0)
            fx += panic_fx * agent.panic_level
            fy += panic_fy * agent.panic_level
        
        # Apply friction
        fx -= agent.velocity[0] * self.friction_coefficient
        fy -= agent.velocity[1] * self.friction_coefficient
        
        return fx, fy
    
    def _calculate_local_density(self, position: Tuple[float, float], 
                               radius: float = 2.0) -> float:
        """Calculate local density around a position"""
        
        count = 0
        for agent in self.agents:
            dx = agent.position[0] - position[0]
            dy = agent.position[1] - position[1]
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance < radius:
                count += 1
        
        # Convert to density (people per square meter)
        area = np.pi * radius**2
        return count / area
    
    def update_simulation(self):
        """Update simulation by one time step"""
        
        # Update each agent
        for agent in self.agents:
            # Calculate forces
            fx, fy = self.calculate_forces(agent)
            
            # Update velocity (F = ma, so a = F/m)
            ax = fx / agent.mass
            ay = fy / agent.mass
            
            # Update velocity
            vx = agent.velocity[0] + ax * self.time_step
            vy = agent.velocity[1] + ay * self.time_step
            
            # Limit speed
            speed = np.sqrt(vx**2 + vy**2)
            if speed > agent.max_speed * agent.mobility:
                vx = vx / speed * agent.max_speed * agent.mobility
                vy = vy / speed * agent.max_speed * agent.mobility
            
            agent.velocity = (vx, vy)
            
            # Update position
            new_x = agent.position[0] + vx * self.time_step
            new_y = agent.position[1] + vy * self.time_step
            
            # Boundary conditions
            new_x = max(agent.radius, min(self.environment.width - agent.radius, new_x))
            new_y = max(agent.radius, min(self.environment.height - agent.radius, new_y))
            
            agent.position = (new_x, new_y)
        
        # Update simulation time
        self.simulation_time += self.time_step
    
    def run_simulation(self, duration: float, save_interval: float = 1.0) -> List[SimulationResult]:
        """Run simulation for specified duration"""
        
        results = []
        last_save_time = 0.0
        
        print(f"🔄 Running simulation for {duration:.1f} seconds...")
        
        start_time = time.time()
        
        while self.simulation_time < duration:
            self.update_simulation()
            
            # Save results at intervals
            if self.simulation_time - last_save_time >= save_interval:
                result = self._capture_simulation_state()
                results.append(result)
                last_save_time = self.simulation_time
                
                if len(results) % 10 == 0:
                    print(f"   Time: {self.simulation_time:.1f}s, Agents: {len(self.agents)}")
        
        end_time = time.time()
        simulation_speed = len(self.agents) * duration / (end_time - start_time)
        self.simulation_speed = simulation_speed
        
        print(f"✅ Simulation completed - Speed: {simulation_speed:.1f} agents/second")
        
        return results
    
    def _capture_simulation_state(self) -> SimulationResult:
        """Capture current simulation state"""
        
        # Create density map
        density_map = self._create_density_map()
        
        # Create velocity field
        velocity_field = self._create_velocity_field()
        
        # Create pressure map
        pressure_map = self._create_pressure_map()
        
        # Find bottlenecks
        bottleneck_locations = self._find_bottlenecks()
        
        # Find risk zones
        risk_zones = self._find_risk_zones()
        
        # Calculate evacuation time
        evacuation_time = self._estimate_evacuation_time()
        
        # Predict casualties
        casualties_predicted = self._predict_casualties()
        
        # Calculate simulation metrics
        metrics = self._calculate_simulation_metrics()
        
        return SimulationResult(
            timestamp=self.simulation_time,
            agents=self.agents.copy(),
            density_map=density_map,
            velocity_field=velocity_field,
            pressure_map=pressure_map,
            bottleneck_locations=bottleneck_locations,
            risk_zones=risk_zones,
            evacuation_time=evacuation_time,
            casualties_predicted=casualties_predicted,
            simulation_metrics=metrics
        )
    
    def _create_density_map(self) -> np.ndarray:
        """Create density map of the simulation area"""
        
        density_map = np.zeros((self.grid_resolution, self.grid_resolution))
        
        # Calculate grid cell size
        cell_width = self.environment.width / self.grid_resolution
        cell_height = self.environment.height / self.grid_resolution
        
        # Count agents in each cell
        for agent in self.agents:
            grid_x = int(agent.position[0] / cell_width)
            grid_y = int(agent.position[1] / cell_height)
            
            if 0 <= grid_x < self.grid_resolution and 0 <= grid_y < self.grid_resolution:
                density_map[grid_y, grid_x] += 1
        
        # Convert to density (people per square meter)
        cell_area = cell_width * cell_height
        density_map = density_map / cell_area
        
        return density_map
    
    def _create_velocity_field(self) -> np.ndarray:
        """Create velocity field of the simulation area"""
        
        velocity_field = np.zeros((self.grid_resolution, self.grid_resolution, 2))
        
        # Calculate grid cell size
        cell_width = self.environment.width / self.grid_resolution
        cell_height = self.environment.height / self.grid_resolution
        
        # Average velocity in each cell
        cell_counts = np.zeros((self.grid_resolution, self.grid_resolution))
        
        for agent in self.agents:
            grid_x = int(agent.position[0] / cell_width)
            grid_y = int(agent.position[1] / cell_height)
            
            if 0 <= grid_x < self.grid_resolution and 0 <= grid_y < self.grid_resolution:
                velocity_field[grid_y, grid_x, 0] += agent.velocity[0]
                velocity_field[grid_y, grid_x, 1] += agent.velocity[1]
                cell_counts[grid_y, grid_x] += 1
        
        # Average velocities
        for i in range(self.grid_resolution):
            for j in range(self.grid_resolution):
                if cell_counts[i, j] > 0:
                    velocity_field[i, j, 0] /= cell_counts[i, j]
                    velocity_field[i, j, 1] /= cell_counts[i, j]
        
        return velocity_field
    
    def _create_pressure_map(self) -> np.ndarray:
        """Create pressure map based on density and velocity"""
        
        density_map = self._create_density_map()
        velocity_field = self._create_velocity_field()
        
        # Calculate pressure (density * velocity^2)
        velocity_magnitude = np.sqrt(velocity_field[:, :, 0]**2 + velocity_field[:, :, 1]**2)
        pressure_map = density_map * velocity_magnitude**2
        
        return pressure_map
    
    def _find_bottlenecks(self) -> List[Tuple[float, float]]:
        """Find bottleneck locations in the simulation"""
        
        density_map = self._create_density_map()
        bottlenecks = []
        
        # Find high density areas
        threshold = np.percentile(density_map, 90)  # Top 10% density
        
        for i in range(1, self.grid_resolution - 1):
            for j in range(1, self.grid_resolution - 1):
                if density_map[i, j] > threshold:
                    # Check if it's a local maximum
                    local_max = True
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if density_map[i + di, j + dj] > density_map[i, j]:
                                local_max = False
                                break
                        if not local_max:
                            break
                    
                    if local_max:
                        # Convert grid coordinates to world coordinates
                        cell_width = self.environment.width / self.grid_resolution
                        cell_height = self.environment.height / self.grid_resolution
                        x = j * cell_width
                        y = i * cell_height
                        bottlenecks.append((x, y))
        
        return bottlenecks
    
    def _find_risk_zones(self) -> List[Tuple[float, float, float]]:
        """Find risk zones based on density, velocity, and panic"""
        
        density_map = self._create_density_map()
        velocity_field = self._create_velocity_field()
        risk_zones = []
        
        # Calculate risk score for each cell
        for i in range(self.grid_resolution):
            for j in range(self.grid_resolution):
                density = density_map[i, j]
                velocity = np.sqrt(velocity_field[i, j, 0]**2 + velocity_field[i, j, 1]**2)
                
                # Count panicked agents in this area
                panic_count = 0
                total_agents = 0
                
                cell_width = self.environment.width / self.grid_resolution
                cell_height = self.environment.height / self.grid_resolution
                
                for agent in self.agents:
                    grid_x = int(agent.position[0] / cell_width)
                    grid_y = int(agent.position[1] / cell_height)
                    
                    if grid_x == j and grid_y == i:
                        total_agents += 1
                        if agent.panic_level > 0.5:
                            panic_count += 1
                
                panic_ratio = panic_count / max(total_agents, 1)
                
                # Calculate risk score
                risk_score = density * velocity * (1 + panic_ratio)
                
                if risk_score > 2.0:  # Risk threshold
                    x = j * cell_width
                    y = i * cell_height
                    risk_zones.append((x, y, risk_score))
        
        return risk_zones
    
    def _estimate_evacuation_time(self) -> Optional[float]:
        """Estimate evacuation time based on current conditions"""
        
        if not self.environment.exits:
            return None
        
        # Calculate total exit capacity
        total_capacity = sum(exit_obs.capacity for exit_obs in self.environment.exits)
        if total_capacity == 0:
            total_capacity = len(self.environment.exits) * 10  # Default capacity
        
        # Calculate current flow rate
        flow_rate = 0.0
        for exit_obs in self.environment.exits:
            # Count agents near exit
            exit_x, exit_y = exit_obs.position
            agents_near_exit = 0
            
            for agent in self.agents:
                dx = agent.position[0] - exit_x
                dy = agent.position[1] - exit_y
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance < 5.0:  # Within 5 meters of exit
                    agents_near_exit += 1
            
            # Estimate flow rate (agents per second)
            exit_flow_rate = min(agents_near_exit * 0.5, exit_obs.capacity * 0.1)
            flow_rate += exit_flow_rate
        
        if flow_rate > 0:
            evacuation_time = len(self.agents) / flow_rate
            return evacuation_time
        
        return None
    
    def _predict_casualties(self) -> int:
        """Predict potential casualties based on simulation conditions"""
        
        casualties = 0
        
        # Count agents in high-risk situations
        for agent in self.agents:
            local_density = self._calculate_local_density(agent.position)
            
            # High density + high panic = casualty risk
            if local_density > 6.0 and agent.panic_level > 0.7:
                casualties += 1
            elif local_density > 8.0:  # Extreme density
                casualties += 1
        
        return casualties
    
    def _calculate_simulation_metrics(self) -> Dict[str, float]:
        """Calculate various simulation metrics"""
        
        metrics = {}
        
        # Average density
        density_map = self._create_density_map()
        metrics['average_density'] = np.mean(density_map)
        metrics['max_density'] = np.max(density_map)
        
        # Average velocity
        velocities = [np.sqrt(v[0]**2 + v[1]**2) for v in [agent.velocity for agent in self.agents]]
        metrics['average_velocity'] = np.mean(velocities)
        metrics['max_velocity'] = np.max(velocities)
        
        # Panic level
        panic_levels = [agent.panic_level for agent in self.agents]
        metrics['average_panic'] = np.mean(panic_levels)
        metrics['max_panic'] = np.max(panic_levels)
        
        # Pressure
        pressure_map = self._create_pressure_map()
        metrics['average_pressure'] = np.mean(pressure_map)
        metrics['max_pressure'] = np.max(pressure_map)
        
        # Bottlenecks
        bottlenecks = self._find_bottlenecks()
        metrics['bottleneck_count'] = len(bottlenecks)
        
        # Risk zones
        risk_zones = self._find_risk_zones()
        metrics['risk_zone_count'] = len(risk_zones)
        
        return metrics
    
    def visualize_simulation(self, result: SimulationResult, save_path: str = None):
        """Visualize simulation results"""
        
        if not self.visualization_enabled:
            print("⚠️ Visualization not available - matplotlib not installed")
            return
        
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        self.ax.clear()
        
        # Plot agents
        for agent in result.agents:
            color = 'red' if agent.panic_level > 0.5 else 'blue'
            circle = plt.Circle(agent.position, agent.radius, color=color, alpha=0.7)
            self.ax.add_patch(circle)
        
        # Plot obstacles
        for obstacle in self.environment.obstacles:
            if obstacle.obstacle_type in ['wall', 'barrier']:
                rect = plt.Rectangle(
                    (obstacle.position[0] - obstacle.size[0]/2, 
                     obstacle.position[1] - obstacle.size[1]/2),
                    obstacle.size[0], obstacle.size[1],
                    color='black', alpha=0.8
                )
                self.ax.add_patch(rect)
            elif obstacle.obstacle_type == 'exit':
                rect = plt.Rectangle(
                    (obstacle.position[0] - obstacle.size[0]/2, 
                     obstacle.position[1] - obstacle.size[1]/2),
                    obstacle.size[0], obstacle.size[1],
                    color='green', alpha=0.8
                )
                self.ax.add_patch(rect)
        
        # Plot bottlenecks
        for x, y in result.bottleneck_locations:
            self.ax.scatter(x, y, color='orange', s=100, marker='x')
        
        # Plot risk zones
        for x, y, risk in result.risk_zones:
            self.ax.scatter(x, y, color='red', s=risk*50, alpha=0.5)
        
        # Set limits and labels
        self.ax.set_xlim(0, self.environment.width)
        self.ax.set_ylim(0, self.environment.height)
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_title(f'Crowd Simulation - Time: {result.timestamp:.1f}s')
        
        # Add legend
        self.ax.scatter([], [], color='blue', label='Normal Agents')
        self.ax.scatter([], [], color='red', label='Panicked Agents')
        self.ax.scatter([], [], color='green', label='Exits')
        self.ax.scatter([], [], color='orange', label='Bottlenecks')
        self.ax.scatter([], [], color='red', label='Risk Zones')
        self.ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Simulation visualization saved to {save_path}")
        else:
            plt.show()
    
    def get_simulation_statistics(self) -> Dict[str, Any]:
        """Get simulation statistics"""
        return {
            'simulation_time': self.simulation_time,
            'num_agents': len(self.agents),
            'simulation_speed': self.simulation_speed,
            'environment_size': (self.environment.width, self.environment.height) if self.environment else None,
            'num_obstacles': len(self.environment.obstacles) if self.environment else 0,
            'num_exits': len(self.environment.exits) if self.environment else 0,
            'time_step': self.time_step,
            'grid_resolution': self.grid_resolution,
            'visualization_enabled': self.visualization_enabled
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize simulator
    simulator = CrowdSimulator()
    
    # Create environment
    print("🏗️ Creating simulation environment...")
    obstacles = [
        Obstacle(0, (20, 10), (2, 20), "wall"),  # Wall
        Obstacle(1, (40, 30), (20, 2), "wall"),  # Wall
        Obstacle(2, (10, 40), (2, 2), "exit"),    # Exit
        Obstacle(3, (50, 5), (2, 2), "exit"),     # Exit
    ]
    
    environment = simulator.create_environment(60, 50, obstacles)
    
    # Add agents
    print("👥 Adding agents to simulation...")
    simulator.add_agents(100, spawn_area=(5, 5, 55, 45))
    
    # Set agent targets
    simulator.set_agent_targets("panic_evacuation")
    
    # Run simulation
    print("🔄 Running crowd simulation...")
    results = simulator.run_simulation(duration=30.0, save_interval=2.0)
    
    # Analyze results
    print(f"\n📊 Simulation Results:")
    print(f"   Duration: {simulator.simulation_time:.1f} seconds")
    print(f"   Agents: {len(simulator.agents)}")
    print(f"   Simulation Speed: {simulator.simulation_speed:.1f} agents/second")
    
    # Analyze final state
    final_result = results[-1]
    print(f"\n🎯 Final State Analysis:")
    print(f"   Average Density: {final_result.simulation_metrics['average_density']:.2f} people/m²")
    print(f"   Max Density: {final_result.simulation_metrics['max_density']:.2f} people/m²")
    print(f"   Average Velocity: {final_result.simulation_metrics['average_velocity']:.2f} m/s")
    print(f"   Average Panic: {final_result.simulation_metrics['average_panic']:.2f}")
    print(f"   Bottlenecks: {final_result.simulation_metrics['bottleneck_count']}")
    print(f"   Risk Zones: {final_result.simulation_metrics['risk_zone_count']}")
    print(f"   Predicted Casualties: {final_result.casualties_predicted}")
    
    if final_result.evacuation_time:
        print(f"   Estimated Evacuation Time: {final_result.evacuation_time:.1f} seconds")
    
    # Visualize final state
    if simulator.visualization_enabled:
        print("\n🎨 Creating visualization...")
        simulator.visualize_simulation(final_result, "crowd_simulation_result.png")
    
    # Get statistics
    stats = simulator.get_simulation_statistics()
    print(f"\n📈 Simulation Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
