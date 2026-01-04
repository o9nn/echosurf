"""
Gamer-Girl Persona Module for Deep Tree Echo

This module implements a gaming-focused persona with:
- Advanced 1st person (1P) and 3rd person (3P) coordination
- Unreal Engine-compatible spatial awareness
- Lightning-fast reflex system
- Strategic mastery for tactical decision-making
- Avatar embodied cognition
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import time


class GamePerspective(Enum):
    """Gaming perspective types"""
    FIRST_PERSON = "1P"
    THIRD_PERSON = "3P"
    TOP_DOWN = "TD"
    ISOMETRIC = "ISO"


class ReflexMode(Enum):
    """Reflex response modes"""
    LIGHTNING = "lightning"  # <50ms response
    COMPETITIVE = "competitive"  # 50-100ms response
    CASUAL = "casual"  # 100-200ms response
    STRATEGIC = "strategic"  # >200ms, prioritize planning


@dataclass
class GamingSkills:
    """Gaming skill proficiency levels (0.0 to 1.0)"""
    aim_precision: float = 0.5
    reaction_time: float = 0.5
    spatial_awareness: float = 0.5
    tactical_planning: float = 0.5
    resource_management: float = 0.5
    movement_control: float = 0.5
    situational_awareness: float = 0.5
    coordination_1p: float = 0.5
    coordination_3p: float = 0.5
    avatar_embodiment: float = 0.5
    
    def to_dict(self) -> Dict[str, float]:
        """Convert skills to dictionary"""
        return {
            'aim_precision': self.aim_precision,
            'reaction_time': self.reaction_time,
            'spatial_awareness': self.spatial_awareness,
            'tactical_planning': self.tactical_planning,
            'resource_management': self.resource_management,
            'movement_control': self.movement_control,
            'situational_awareness': self.situational_awareness,
            'coordination_1p': self.coordination_1p,
            'coordination_3p': self.coordination_3p,
            'avatar_embodiment': self.avatar_embodiment
        }


@dataclass
class AvatarState:
    """Current avatar state in game environment"""
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    health: float = 1.0
    stamina: float = 1.0
    active_abilities: List[str] = field(default_factory=list)
    inventory: Dict[str, int] = field(default_factory=dict)
    perspective: GamePerspective = GamePerspective.FIRST_PERSON
    
    # Unreal Engine compatible fields
    forward_vector: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    right_vector: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    up_vector: Tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass
class TacticalDecision:
    """Represents a tactical decision point"""
    timestamp: float = field(default_factory=time.time)
    situation: str = ""
    options: List[str] = field(default_factory=list)
    chosen_action: str = ""
    confidence: float = 0.0
    outcome: Optional[str] = None
    success: bool = False


class GamerPersona:
    """
    Gamer-Girl Persona with advanced gaming capabilities
    """
    
    def __init__(self, name: str = "Echo-Gamer", reflex_mode: ReflexMode = ReflexMode.LIGHTNING):
        self.logger = logging.getLogger(__name__)
        self.name = name
        self.reflex_mode = reflex_mode
        
        # Initialize gaming skills
        self.skills = GamingSkills(
            aim_precision=0.8,
            reaction_time=0.9,
            spatial_awareness=0.85,
            tactical_planning=0.75,
            coordination_1p=0.85,
            coordination_3p=0.82,
            avatar_embodiment=0.88
        )
        
        # Avatar state
        self.avatar = AvatarState()
        
        # Performance tracking
        self.reflex_history: List[float] = []
        self.decision_history: List[TacticalDecision] = []
        self.training_sessions: List[Dict[str, Any]] = []
        
        # Strategic mastery
        self.tactical_patterns: Dict[str, List[str]] = {}
        self.learned_strategies: List[Dict[str, Any]] = []
        
        # Embodied cognition
        self.motor_patterns: Dict[str, np.ndarray] = {}
        self.muscle_memory: Dict[str, List[Tuple[float, float]]] = {}
        
        # Performance metrics
        self.avg_reaction_time: float = 0.0
        self.peak_performance_time: Optional[float] = None
        
        self.logger.info(f"Initialized {self.name} with {reflex_mode.value} reflexes")
    
    def calculate_target_reflex_time(self) -> float:
        """Calculate target response time based on reflex mode"""
        if self.reflex_mode == ReflexMode.LIGHTNING:
            return 0.045  # 45ms
        elif self.reflex_mode == ReflexMode.COMPETITIVE:
            return 0.075  # 75ms
        elif self.reflex_mode == ReflexMode.CASUAL:
            return 0.150  # 150ms
        else:  # STRATEGIC
            return 0.250  # 250ms
    
    def train_coordination_1p(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train first-person coordination skills
        
        Args:
            scenario: Training scenario with targets, obstacles, etc.
            
        Returns:
            Training results with performance metrics
        """
        start_time = time.time()
        
        # Set perspective
        self.avatar.perspective = GamePerspective.FIRST_PERSON
        
        # Extract scenario elements
        targets = scenario.get('targets', [])
        obstacles = scenario.get('obstacles', [])
        time_limit = scenario.get('time_limit', 10.0)
        
        results = {
            'hits': 0,
            'misses': 0,
            'avg_reaction_time': 0.0,
            'accuracy': 0.0,
            'completion_time': 0.0,
            'coordination_score': 0.0
        }
        
        reaction_times = []
        
        # Simulate 1P engagement with targets
        for i, target in enumerate(targets):
            target_time = time.time()
            
            # Calculate aim adjustment based on target position
            target_pos = target.get('position', (0, 0, 0))
            aim_error = self._calculate_aim_error_1p(target_pos)
            
            # Apply reflex response
            reaction_time = self._simulate_reflex_response()
            reaction_times.append(reaction_time)
            
            # Check if hit or miss based on accuracy and timing
            hit_chance = self.skills.aim_precision * (1.0 - aim_error) * self.skills.coordination_1p
            
            if np.random.random() < hit_chance and reaction_time < time_limit:
                results['hits'] += 1
            else:
                results['misses'] += 1
            
            # Update muscle memory
            self._update_muscle_memory('1p_target', target_pos[:2])
        
        # Calculate final metrics
        total_attempts = results['hits'] + results['misses']
        if total_attempts > 0:
            results['accuracy'] = results['hits'] / total_attempts
            results['avg_reaction_time'] = np.mean(reaction_times)
            results['coordination_score'] = (
                results['accuracy'] * 0.6 + 
                (1.0 - min(1.0, results['avg_reaction_time'] / self.calculate_target_reflex_time())) * 0.4
            )
        
        results['completion_time'] = time.time() - start_time
        
        # Update skills based on performance
        self._adapt_skills_1p(results)
        
        # Store training session
        self.training_sessions.append({
            'type': '1P_coordination',
            'timestamp': start_time,
            'results': results,
            'scenario': scenario
        })
        
        return results
    
    def train_coordination_3p(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train third-person coordination skills
        
        Args:
            scenario: Training scenario with spatial challenges
            
        Returns:
            Training results with performance metrics
        """
        start_time = time.time()
        
        # Set perspective
        self.avatar.perspective = GamePerspective.THIRD_PERSON
        
        # Extract scenario elements
        navigation_points = scenario.get('navigation_points', [])
        camera_angles = scenario.get('camera_angles', [])
        spatial_challenges = scenario.get('spatial_challenges', [])
        
        results = {
            'navigation_accuracy': 0.0,
            'spatial_awareness_score': 0.0,
            'camera_control_score': 0.0,
            'completion_time': 0.0,
            'coordination_score': 0.0
        }
        
        # Simulate 3P navigation and spatial awareness
        navigation_errors = []
        
        for point in navigation_points:
            target_pos = point.get('position', (0, 0, 0))
            
            # Calculate navigation error with 3P perspective
            nav_error = self._calculate_navigation_error_3p(target_pos)
            navigation_errors.append(nav_error)
            
            # Simulate camera adjustment
            camera_angle = point.get('camera_angle', 0)
            camera_score = self._evaluate_camera_control(camera_angle)
            
            # Update spatial awareness
            self._update_spatial_map_3p(target_pos)
        
        # Calculate metrics
        if navigation_errors:
            results['navigation_accuracy'] = 1.0 - min(1.0, np.mean(navigation_errors))
            results['spatial_awareness_score'] = self.skills.spatial_awareness * results['navigation_accuracy']
            results['camera_control_score'] = self.skills.coordination_3p
            results['coordination_score'] = (
                results['navigation_accuracy'] * 0.4 +
                results['spatial_awareness_score'] * 0.3 +
                results['camera_control_score'] * 0.3
            )
        
        results['completion_time'] = time.time() - start_time
        
        # Update skills based on performance
        self._adapt_skills_3p(results)
        
        # Store training session
        self.training_sessions.append({
            'type': '3P_coordination',
            'timestamp': start_time,
            'results': results,
            'scenario': scenario
        })
        
        return results
    
    def make_tactical_decision(self, situation: str, options: List[str], 
                             context: Dict[str, Any]) -> TacticalDecision:
        """
        Make a strategic tactical decision based on situation
        
        Args:
            situation: Description of current tactical situation
            options: Available actions
            context: Additional context (enemy positions, resources, etc.)
            
        Returns:
            TacticalDecision with chosen action and confidence
        """
        decision = TacticalDecision(
            situation=situation,
            options=options
        )
        
        # Analyze situation using strategic mastery
        threat_level = context.get('threat_level', 0.5)
        resources = context.get('resources', {})
        allies = context.get('allies', [])
        
        # Calculate option scores
        option_scores = {}
        for option in options:
            score = self._evaluate_tactical_option(
                option, situation, threat_level, resources, allies
            )
            option_scores[option] = score
        
        # Choose best option
        if option_scores:
            decision.chosen_action = max(option_scores, key=option_scores.get)
            decision.confidence = option_scores[decision.chosen_action]
        
        # Record decision
        self.decision_history.append(decision)
        
        # Update tactical patterns
        if situation not in self.tactical_patterns:
            self.tactical_patterns[situation] = []
        self.tactical_patterns[situation].append(decision.chosen_action)
        
        return decision
    
    def update_avatar_embodiment(self, sensory_input: Dict[str, Any]) -> bool:
        """
        Update avatar state based on sensory input (embodied cognition)
        
        Args:
            sensory_input: Raw sensory data from game environment
            
        Returns:
            Success status
        """
        try:
            # Update position
            if 'position' in sensory_input:
                self.avatar.position = tuple(sensory_input['position'])
            
            # Update rotation/orientation
            if 'rotation' in sensory_input:
                self.avatar.rotation = tuple(sensory_input['rotation'])
            
            # Update velocity
            if 'velocity' in sensory_input:
                self.avatar.velocity = tuple(sensory_input['velocity'])
            
            # Update health and stamina
            self.avatar.health = sensory_input.get('health', self.avatar.health)
            self.avatar.stamina = sensory_input.get('stamina', self.avatar.stamina)
            
            # Update Unreal Engine vectors
            self._update_ue_vectors()
            
            # Adapt skills based on embodiment
            embodiment_quality = self._calculate_embodiment_quality()
            self.skills.avatar_embodiment = (
                0.9 * self.skills.avatar_embodiment + 0.1 * embodiment_quality
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating avatar embodiment: {e}")
            return False
    
    def _calculate_aim_error_1p(self, target_pos: Tuple[float, float, float]) -> float:
        """Calculate aiming error in first-person perspective"""
        # Simplified error calculation based on distance and angle
        distance = np.linalg.norm(np.array(target_pos) - np.array(self.avatar.position))
        
        # Error increases with distance, decreases with skill
        base_error = distance * 0.1
        skill_factor = 1.0 - self.skills.aim_precision
        
        return base_error * skill_factor
    
    def _calculate_navigation_error_3p(self, target_pos: Tuple[float, float, float]) -> float:
        """Calculate navigation error in third-person perspective"""
        # Account for camera angle and spatial awareness
        distance = np.linalg.norm(np.array(target_pos) - np.array(self.avatar.position))
        
        base_error = distance * 0.15  # 3P has slightly higher base error
        skill_factor = 1.0 - (self.skills.spatial_awareness * self.skills.coordination_3p)
        
        return base_error * skill_factor
    
    def _simulate_reflex_response(self) -> float:
        """Simulate reflex response time"""
        target_time = self.calculate_target_reflex_time()
        
        # Add variance based on reaction time skill
        variance = (1.0 - self.skills.reaction_time) * 0.05
        actual_time = target_time + np.random.normal(0, variance)
        
        # Record reflex
        self.reflex_history.append(actual_time)
        
        # Update average
        if len(self.reflex_history) > 100:
            self.reflex_history = self.reflex_history[-100:]
        self.avg_reaction_time = np.mean(self.reflex_history)
        
        return actual_time
    
    def _update_muscle_memory(self, action: str, position: Tuple[float, float]):
        """Update muscle memory patterns for repeated actions"""
        if action not in self.muscle_memory:
            self.muscle_memory[action] = []
        
        self.muscle_memory[action].append(position)
        
        # Keep last 1000 patterns
        if len(self.muscle_memory[action]) > 1000:
            self.muscle_memory[action] = self.muscle_memory[action][-1000:]
    
    def _update_spatial_map_3p(self, position: Tuple[float, float, float]):
        """Update internal spatial map for 3P navigation"""
        # Store in motor patterns as spatial memory
        if '3p_spatial' not in self.motor_patterns:
            self.motor_patterns['3p_spatial'] = np.zeros((100, 3))
        
        # Add new position to spatial memory (simplified)
        idx = len(self.training_sessions) % 100
        self.motor_patterns['3p_spatial'][idx] = position
    
    def _evaluate_camera_control(self, camera_angle: float) -> float:
        """Evaluate camera control quality"""
        # Optimal camera angles are typically between 30-60 degrees in 3P games
        optimal_angle = 45.0
        angle_error = abs(camera_angle - optimal_angle) / 180.0
        
        return (1.0 - angle_error) * self.skills.coordination_3p
    
    def _evaluate_tactical_option(self, option: str, situation: str, 
                                 threat_level: float, resources: Dict, 
                                 allies: List) -> float:
        """Evaluate tactical option quality"""
        base_score = 0.5
        
        # Check past success with this option in similar situations
        if situation in self.tactical_patterns:
            pattern_frequency = self.tactical_patterns[situation].count(option)
            pattern_score = pattern_frequency / len(self.tactical_patterns[situation])
            base_score = 0.3 * base_score + 0.7 * pattern_score
        
        # Adjust based on threat level
        if 'aggressive' in option.lower() and threat_level > 0.7:
            base_score *= 1.2
        elif 'defensive' in option.lower() and threat_level < 0.3:
            base_score *= 0.8
        
        # Resource considerations
        if 'resources' in option.lower() and resources:
            base_score *= 1.1
        
        # Team coordination
        if 'team' in option.lower() and allies:
            base_score *= 1.15
        
        # Apply tactical planning skill
        return min(1.0, base_score * self.skills.tactical_planning)
    
    def _adapt_skills_1p(self, results: Dict[str, Any]):
        """Adapt 1P skills based on training results"""
        learning_rate = 0.05
        
        if results.get('accuracy', 0) > 0.8:
            self.skills.aim_precision = min(1.0, self.skills.aim_precision + learning_rate)
            self.skills.coordination_1p = min(1.0, self.skills.coordination_1p + learning_rate)
        
        if results.get('avg_reaction_time', 1.0) < self.calculate_target_reflex_time():
            self.skills.reaction_time = min(1.0, self.skills.reaction_time + learning_rate)
    
    def _adapt_skills_3p(self, results: Dict[str, Any]):
        """Adapt 3P skills based on training results"""
        learning_rate = 0.05
        
        if results.get('navigation_accuracy', 0) > 0.8:
            self.skills.spatial_awareness = min(1.0, self.skills.spatial_awareness + learning_rate)
            self.skills.coordination_3p = min(1.0, self.skills.coordination_3p + learning_rate)
        
        if results.get('coordination_score', 0) > 0.85:
            self.skills.movement_control = min(1.0, self.skills.movement_control + learning_rate)
    
    def _update_ue_vectors(self):
        """Update Unreal Engine compatible direction vectors"""
        # Convert rotation to direction vectors
        pitch, yaw, roll = self.avatar.rotation
        
        # Forward vector (using yaw)
        self.avatar.forward_vector = (
            np.cos(np.radians(yaw)),
            np.sin(np.radians(yaw)),
            0.0
        )
        
        # Right vector (perpendicular to forward)
        self.avatar.right_vector = (
            -np.sin(np.radians(yaw)),
            np.cos(np.radians(yaw)),
            0.0
        )
        
        # Up vector (standard)
        self.avatar.up_vector = (0.0, 0.0, 1.0)
    
    def _calculate_embodiment_quality(self) -> float:
        """Calculate how well the persona is embodied in the avatar"""
        # Based on consistency of avatar state updates
        quality = 0.5
        
        # Check if position is being updated
        if self.avatar.position != (0.0, 0.0, 0.0):
            quality += 0.2
        
        # Check if rotation is being updated
        if self.avatar.rotation != (0.0, 0.0, 0.0):
            quality += 0.2
        
        # Check velocity tracking
        if self.avatar.velocity != (0.0, 0.0, 0.0):
            quality += 0.1
        
        return min(1.0, quality)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            'name': self.name,
            'reflex_mode': self.reflex_mode.value,
            'skills': self.skills.to_dict(),
            'avg_reaction_time': self.avg_reaction_time,
            'target_reaction_time': self.calculate_target_reflex_time(),
            'training_sessions': len(self.training_sessions),
            'decisions_made': len(self.decision_history),
            'avatar_embodiment': self.skills.avatar_embodiment,
            'perspective': self.avatar.perspective.value,
            'tactical_patterns_learned': len(self.tactical_patterns)
        }
