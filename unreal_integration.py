"""
Unreal Engine Integration Module for Deep Tree Echo

Provides interfaces and utilities for working with Unreal Engine concepts:
- 3D coordinate systems
- Actor/Pawn representation
- Level streaming awareness
- Blueprint-compatible data structures
- Gameplay framework integration
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json


class ActorType(Enum):
    """Unreal Engine Actor types"""
    PAWN = "Pawn"
    CHARACTER = "Character"
    PROJECTILE = "Projectile"
    PICKUP = "Pickup"
    TRIGGER = "Trigger"
    ENVIRONMENT = "Environment"


class MovementMode(Enum):
    """Unreal Engine movement modes"""
    WALKING = "Walking"
    FALLING = "Falling"
    FLYING = "Flying"
    SWIMMING = "Swimming"
    CUSTOM = "Custom"


@dataclass
class UEVector:
    """Unreal Engine compatible vector (right-handed Z-up coordinate system)"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)
    
    def magnitude(self) -> float:
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalized(self) -> 'UEVector':
        mag = self.magnitude()
        if mag > 0:
            return UEVector(self.x/mag, self.y/mag, self.z/mag)
        return UEVector()
    
    def dot(self, other: 'UEVector') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: 'UEVector') -> 'UEVector':
        return UEVector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )


@dataclass
class UERotator:
    """Unreal Engine rotator (pitch, yaw, roll in degrees)"""
    pitch: float = 0.0  # Rotation around Y axis
    yaw: float = 0.0    # Rotation around Z axis
    roll: float = 0.0   # Rotation around X axis
    
    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.pitch, self.yaw, self.roll)
    
    def to_forward_vector(self) -> UEVector:
        """Convert rotation to forward direction vector"""
        pitch_rad = np.radians(self.pitch)
        yaw_rad = np.radians(self.yaw)
        
        return UEVector(
            np.cos(pitch_rad) * np.cos(yaw_rad),
            np.cos(pitch_rad) * np.sin(yaw_rad),
            np.sin(pitch_rad)
        )
    
    def to_right_vector(self) -> UEVector:
        """Convert rotation to right direction vector"""
        yaw_rad = np.radians(self.yaw)
        
        return UEVector(
            -np.sin(yaw_rad),
            np.cos(yaw_rad),
            0.0
        )
    
    def to_up_vector(self) -> UEVector:
        """Convert rotation to up direction vector"""
        # For simple case, up is relative to roll
        forward = self.to_forward_vector()
        right = self.to_right_vector()
        
        return right.cross(forward)


@dataclass
class UETransform:
    """Unreal Engine transform (location, rotation, scale)"""
    location: UEVector = field(default_factory=UEVector)
    rotation: UERotator = field(default_factory=UERotator)
    scale: UEVector = field(default_factory=lambda: UEVector(1.0, 1.0, 1.0))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'location': self.location.to_tuple(),
            'rotation': self.rotation.to_tuple(),
            'scale': self.scale.to_tuple()
        }


@dataclass
class UEActor:
    """Unreal Engine Actor representation"""
    name: str
    actor_type: ActorType
    transform: UETransform = field(default_factory=UETransform)
    velocity: UEVector = field(default_factory=UEVector)
    is_active: bool = True
    tags: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def get_distance_to(self, other: 'UEActor') -> float:
        """Calculate distance to another actor"""
        dx = self.transform.location.x - other.transform.location.x
        dy = self.transform.location.y - other.transform.location.y
        dz = self.transform.location.z - other.transform.location.z
        return np.sqrt(dx**2 + dy**2 + dz**2)
    
    def get_forward_vector(self) -> UEVector:
        """Get the forward direction vector"""
        return self.transform.rotation.to_forward_vector()
    
    def is_in_front_of(self, other: 'UEActor', angle_threshold: float = 90.0) -> bool:
        """Check if another actor is in front of this actor"""
        # Vector from this actor to other
        to_other = UEVector(
            other.transform.location.x - self.transform.location.x,
            other.transform.location.y - self.transform.location.y,
            other.transform.location.z - self.transform.location.z
        ).normalized()
        
        # Forward vector
        forward = self.get_forward_vector()
        
        # Calculate angle using dot product
        dot = forward.dot(to_other)
        angle = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
        
        return angle < angle_threshold


@dataclass
class UECharacter(UEActor):
    """Unreal Engine Character (specialized Pawn with movement)"""
    movement_mode: MovementMode = MovementMode.WALKING
    max_walk_speed: float = 600.0  # Unreal units per second
    jump_z_velocity: float = 420.0
    is_crouched: bool = False
    controller_rotation: UERotator = field(default_factory=UERotator)
    
    def __post_init__(self):
        self.actor_type = ActorType.CHARACTER


class UnrealEnvironment:
    """
    Simulated Unreal Engine environment for training
    """
    
    def __init__(self, level_name: str = "Training_Level"):
        self.logger = logging.getLogger(__name__)
        self.level_name = level_name
        
        # Actor management
        self.actors: Dict[str, UEActor] = {}
        self.player_character: Optional[UECharacter] = None
        
        # Environment properties
        self.gravity_z: float = -980.0  # Unreal units per second squared
        self.world_bounds: Tuple[float, float, float, float, float, float] = (
            -10000, -10000, -1000, 10000, 10000, 5000  # min_x, min_y, min_z, max_x, max_y, max_z
        )
        
        # Gameplay tracking
        self.delta_time: float = 0.016  # ~60 FPS
        self.time_elapsed: float = 0.0
        
        self.logger.info(f"Initialized Unreal Environment: {level_name}")
    
    def spawn_player_character(self, name: str = "Echo_Player",
                              location: Tuple[float, float, float] = (0, 0, 100)) -> UECharacter:
        """Spawn the player character"""
        transform = UETransform(
            location=UEVector(*location),
            rotation=UERotator(0, 0, 0),
            scale=UEVector(1, 1, 1)
        )
        
        self.player_character = UECharacter(
            name=name,
            transform=transform,
            tags=["Player", "Controlled"]
        )
        
        self.actors[name] = self.player_character
        self.logger.info(f"Spawned player character: {name} at {location}")
        
        return self.player_character
    
    def spawn_actor(self, name: str, actor_type: ActorType,
                   location: Tuple[float, float, float],
                   rotation: Tuple[float, float, float] = (0, 0, 0)) -> UEActor:
        """Spawn a generic actor"""
        transform = UETransform(
            location=UEVector(*location),
            rotation=UERotator(*rotation)
        )
        
        actor = UEActor(
            name=name,
            actor_type=actor_type,
            transform=transform
        )
        
        self.actors[name] = actor
        self.logger.info(f"Spawned {actor_type.value}: {name}")
        
        return actor
    
    def spawn_target(self, name: str, location: Tuple[float, float, float],
                    radius: float = 50.0) -> UEActor:
        """Spawn a target for practice"""
        actor = self.spawn_actor(name, ActorType.PICKUP, location)
        actor.properties['radius'] = radius
        actor.properties['is_target'] = True
        actor.tags.append("Target")
        
        return actor
    
    def move_character(self, character: UECharacter, 
                      input_vector: Tuple[float, float],
                      delta_time: float = None) -> bool:
        """Move character based on input"""
        if delta_time is None:
            delta_time = self.delta_time
        
        try:
            # Get movement direction from input
            forward_input, right_input = input_vector
            
            # Get character's rotation
            forward = character.transform.rotation.to_forward_vector()
            right = character.transform.rotation.to_right_vector()
            
            # Calculate movement direction
            movement = UEVector(
                forward.x * forward_input + right.x * right_input,
                forward.y * forward_input + right.y * right_input,
                0.0  # No vertical movement from input
            )
            
            # Normalize and scale by speed
            if movement.magnitude() > 0:
                movement = movement.normalized()
                speed = character.max_walk_speed * delta_time
                
                # Update location
                character.transform.location.x += movement.x * speed
                character.transform.location.y += movement.y * speed
                
                # Update velocity
                character.velocity = UEVector(
                    movement.x * character.max_walk_speed,
                    movement.y * character.max_walk_speed,
                    character.velocity.z
                )
            
            # Apply gravity if falling
            if character.movement_mode == MovementMode.FALLING:
                character.velocity.z += self.gravity_z * delta_time
                character.transform.location.z += character.velocity.z * delta_time
                
                # Check ground collision (simplified)
                if character.transform.location.z <= 0:
                    character.transform.location.z = 0
                    character.velocity.z = 0
                    character.movement_mode = MovementMode.WALKING
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error moving character: {e}")
            return False
    
    def rotate_character(self, character: UECharacter,
                        yaw_delta: float, pitch_delta: float,
                        delta_time: float = None) -> bool:
        """Rotate character controller"""
        if delta_time is None:
            delta_time = self.delta_time
        
        try:
            # Update controller rotation (view direction)
            character.controller_rotation.yaw += yaw_delta * delta_time
            character.controller_rotation.pitch = np.clip(
                character.controller_rotation.pitch + pitch_delta * delta_time,
                -89.0, 89.0  # Prevent looking too far up/down
            )
            
            # For characters, body rotation follows controller yaw
            character.transform.rotation.yaw = character.controller_rotation.yaw
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error rotating character: {e}")
            return False
    
    def check_line_of_sight(self, from_actor: UEActor, to_actor: UEActor,
                           max_distance: float = 10000.0) -> bool:
        """Check if there's line of sight between actors"""
        distance = from_actor.get_distance_to(to_actor)
        
        if distance > max_distance:
            return False
        
        # Simplified LOS check - could add obstacle detection
        # For now, just check if in front and within range
        return from_actor.is_in_front_of(to_actor)
    
    def get_actors_in_radius(self, center: Tuple[float, float, float],
                            radius: float, 
                            actor_type: Optional[ActorType] = None) -> List[UEActor]:
        """Get all actors within radius of a point"""
        center_vec = UEVector(*center)
        actors_in_radius = []
        
        for actor in self.actors.values():
            # Calculate distance
            distance = np.sqrt(
                (actor.transform.location.x - center_vec.x)**2 +
                (actor.transform.location.y - center_vec.y)**2 +
                (actor.transform.location.z - center_vec.z)**2
            )
            
            if distance <= radius:
                if actor_type is None or actor.actor_type == actor_type:
                    actors_in_radius.append(actor)
        
        return actors_in_radius
    
    def create_training_scenario_1p(self, num_targets: int = 10,
                                   spawn_radius: float = 1000.0) -> Dict[str, Any]:
        """Create a first-person training scenario"""
        self.logger.info(f"Creating 1P training scenario with {num_targets} targets")
        
        # Clear existing targets
        self.actors = {k: v for k, v in self.actors.items() if "Target" not in v.tags}
        
        # Spawn targets in a circle around player
        targets = []
        for i in range(num_targets):
            angle = (i / num_targets) * 2 * np.pi
            x = spawn_radius * np.cos(angle)
            y = spawn_radius * np.sin(angle)
            z = np.random.uniform(50, 500)  # Random height
            
            target = self.spawn_target(f"Target_{i}", (x, y, z))
            targets.append({
                'name': target.name,
                'position': (x, y, z),
                'distance': np.sqrt(x**2 + y**2 + z**2)
            })
        
        return {
            'type': '1P_training',
            'num_targets': num_targets,
            'targets': targets,
            'player_position': self.player_character.transform.location.to_tuple() if self.player_character else (0, 0, 0),
            'time_limit': 60.0
        }
    
    def create_training_scenario_3p(self, num_waypoints: int = 8,
                                   area_size: float = 2000.0) -> Dict[str, Any]:
        """Create a third-person training scenario with navigation"""
        self.logger.info(f"Creating 3P training scenario with {num_waypoints} waypoints")
        
        # Generate navigation waypoints
        navigation_points = []
        for i in range(num_waypoints):
            x = np.random.uniform(-area_size/2, area_size/2)
            y = np.random.uniform(-area_size/2, area_size/2)
            z = 0  # Ground level
            
            # Spawn marker
            marker = self.spawn_actor(
                f"Waypoint_{i}",
                ActorType.TRIGGER,
                (x, y, z)
            )
            marker.tags.append("Waypoint")
            marker.properties['order'] = i
            
            # Random camera angle preference
            camera_angle = np.random.uniform(30, 60)
            
            navigation_points.append({
                'name': marker.name,
                'position': (x, y, z),
                'camera_angle': camera_angle,
                'order': i
            })
        
        return {
            'type': '3P_training',
            'num_waypoints': num_waypoints,
            'navigation_points': navigation_points,
            'area_size': area_size,
            'player_position': self.player_character.transform.location.to_tuple() if self.player_character else (0, 0, 0)
        }
    
    def tick(self, delta_time: float = None) -> bool:
        """Update environment (called each frame)"""
        if delta_time is None:
            delta_time = self.delta_time
        
        self.time_elapsed += delta_time
        
        # Update all actors (physics, AI, etc.)
        for actor in self.actors.values():
            if actor.is_active:
                # Apply velocity to location
                actor.transform.location.x += actor.velocity.x * delta_time
                actor.transform.location.y += actor.velocity.y * delta_time
                actor.transform.location.z += actor.velocity.z * delta_time
        
        return True
    
    def get_environment_state(self) -> Dict[str, Any]:
        """Get current environment state"""
        return {
            'level_name': self.level_name,
            'time_elapsed': self.time_elapsed,
            'num_actors': len(self.actors),
            'player_location': self.player_character.transform.location.to_tuple() if self.player_character else None,
            'player_rotation': self.player_character.transform.rotation.to_tuple() if self.player_character else None,
            'active_targets': len([a for a in self.actors.values() if "Target" in a.tags])
        }
    
    def export_to_json(self, filepath: str) -> bool:
        """Export environment to JSON (Blueprint-compatible format)"""
        try:
            export_data = {
                'level_name': self.level_name,
                'actors': []
            }
            
            for actor in self.actors.values():
                actor_data = {
                    'name': actor.name,
                    'type': actor.actor_type.value,
                    'transform': actor.transform.to_dict(),
                    'tags': actor.tags,
                    'properties': actor.properties
                }
                export_data['actors'].append(actor_data)
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Exported environment to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting environment: {e}")
            return False


class UnrealEngineOptimizer:
    """
    Optimization utilities for Unreal Engine performance
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.performance_metrics: List[Dict[str, float]] = []
    
    def optimize_for_reflexes(self, environment: UnrealEnvironment,
                             target_fps: int = 120) -> Dict[str, Any]:
        """Optimize environment for lightning-fast reflexes"""
        self.logger.info(f"Optimizing for {target_fps} FPS reflexes")
        
        # Calculate optimal delta time
        optimal_delta = 1.0 / target_fps
        environment.delta_time = optimal_delta
        
        # Reduce physics complexity for non-critical actors
        non_player_actors = [a for a in environment.actors.values() 
                            if "Player" not in a.tags]
        
        optimizations = {
            'target_fps': target_fps,
            'delta_time': optimal_delta,
            'actors_optimized': len(non_player_actors),
            'recommendations': []
        }
        
        # Add performance recommendations
        if len(environment.actors) > 100:
            optimizations['recommendations'].append(
                "Consider using object pooling for targets"
            )
        
        optimizations['recommendations'].append(
            f"Use {target_fps} FPS cap for consistent input latency"
        )
        
        optimizations['recommendations'].append(
            "Enable 'Reduce Input Latency' in Unreal settings"
        )
        
        return optimizations
    
    def calculate_input_latency(self, frame_time: float, 
                               render_time: float = 0.0) -> float:
        """Calculate total input latency"""
        # Simplified calculation
        # Real UE5 would include: input sampling, game thread, render thread, GPU
        total_latency = frame_time + render_time
        return total_latency * 1000  # Convert to ms
    
    def get_performance_recommendations(self, avg_fps: float) -> List[str]:
        """Get performance optimization recommendations"""
        recommendations = []
        
        if avg_fps < 60:
            recommendations.append("Enable Temporal Super Resolution (TSR)")
            recommendations.append("Reduce shadow quality")
            recommendations.append("Use Level of Detail (LOD) system")
        
        if avg_fps < 120:
            recommendations.append("Disable motion blur for competitive play")
            recommendations.append("Lower post-processing quality")
        
        recommendations.append("Use StatFPS command to monitor performance")
        recommendations.append("Enable NVIDIA Reflex if available")
        
        return recommendations
