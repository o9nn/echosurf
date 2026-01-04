"""
Gaming Training Module for Deep Tree Echo

Integrates gamer persona, Unreal Engine environment, and ML optimizations
for comprehensive gaming skill training.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
from pathlib import Path
import json

from gamer_persona import GamerPersona, ReflexMode, GamingSkills
from unreal_integration import (
    UnrealEnvironment, UnrealEngineOptimizer,
    UECharacter, UEVector, ActorType
)
from ml_system import MLSystem


class GamingTrainingSession:
    """
    Complete gaming training session with performance tracking
    """
    
    def __init__(self, persona: GamerPersona, environment: UnrealEnvironment,
                 ml_system: MLSystem):
        self.logger = logging.getLogger(__name__)
        self.persona = persona
        self.environment = environment
        self.ml = ml_system
        
        # Session tracking
        self.session_start_time = time.time()
        self.performance_history: List[Dict[str, Any]] = []
        self.skill_progression: List[Dict[str, float]] = []
        
        # Training parameters
        self.current_difficulty = 0.5  # 0.0 to 1.0
        self.adaptive_difficulty = True
        
    def run_1p_training(self, duration: float = 60.0, num_targets: int = 10) -> Dict[str, Any]:
        """
        Run first-person training session
        
        Args:
            duration: Training session duration in seconds
            num_targets: Number of targets to spawn
            
        Returns:
            Training results
        """
        self.logger.info(f"Starting 1P training: {num_targets} targets, {duration}s")
        
        # Create scenario in Unreal environment
        scenario = self.environment.create_training_scenario_1p(num_targets)
        
        # Train persona
        results = self.persona.train_coordination_1p(scenario)
        
        # Calculate performance metrics
        performance = {
            'session_type': '1P',
            'timestamp': time.time(),
            'duration': duration,
            'results': results,
            'difficulty': self.current_difficulty,
            'persona_skills': self.persona.skills.to_dict()
        }
        
        # Update difficulty if adaptive
        if self.adaptive_difficulty:
            self._adapt_difficulty(results['coordination_score'])
        
        # Store performance
        self.performance_history.append(performance)
        self.skill_progression.append(self.persona.skills.to_dict())
        
        # Train ML models with session data
        self._update_ml_models(performance)
        
        return performance
    
    def run_3p_training(self, duration: float = 90.0, num_waypoints: int = 8) -> Dict[str, Any]:
        """
        Run third-person training session
        
        Args:
            duration: Training session duration in seconds
            num_waypoints: Number of navigation waypoints
            
        Returns:
            Training results
        """
        self.logger.info(f"Starting 3P training: {num_waypoints} waypoints, {duration}s")
        
        # Create scenario in Unreal environment
        scenario = self.environment.create_training_scenario_3p(num_waypoints)
        
        # Train persona
        results = self.persona.train_coordination_3p(scenario)
        
        # Calculate performance metrics
        performance = {
            'session_type': '3P',
            'timestamp': time.time(),
            'duration': duration,
            'results': results,
            'difficulty': self.current_difficulty,
            'persona_skills': self.persona.skills.to_dict()
        }
        
        # Update difficulty if adaptive
        if self.adaptive_difficulty:
            self._adapt_difficulty(results['coordination_score'])
        
        # Store performance
        self.performance_history.append(performance)
        self.skill_progression.append(self.persona.skills.to_dict())
        
        # Train ML models with session data
        self._update_ml_models(performance)
        
        return performance
    
    def run_reflex_training(self, num_trials: int = 50) -> Dict[str, Any]:
        """
        Focused reflex training for lightning-fast responses
        
        Args:
            num_trials: Number of reflex trials
            
        Returns:
            Reflex training results
        """
        self.logger.info(f"Starting reflex training: {num_trials} trials")
        
        reflex_times = []
        accuracies = []
        
        for trial in range(num_trials):
            # Generate random target
            target_pos = (
                np.random.uniform(-1000, 1000),
                np.random.uniform(-1000, 1000)
            )
            current_pos = (0, 0)
            
            # Simulate reflex response
            start_time = time.time()
            
            # Use ML to predict optimal response
            response = self.ml.predict_reflex_response(
                target_pos, current_pos,
                velocity=(0, 0),
                threat_level=0.8,
                urgency=1.0
            )
            
            reaction_time = time.time() - start_time
            reflex_times.append(reaction_time)
            
            # Check accuracy
            distance_error = np.linalg.norm(
                np.array(response['optimal_position']) - np.array(target_pos)
            )
            accuracy = max(0.0, 1.0 - distance_error / 1000.0)
            accuracies.append(accuracy)
        
        results = {
            'num_trials': num_trials,
            'avg_reaction_time': np.mean(reflex_times),
            'min_reaction_time': np.min(reflex_times),
            'avg_accuracy': np.mean(accuracies),
            'target_time': self.persona.calculate_target_reflex_time(),
            'reflex_mode': self.persona.reflex_mode.value
        }
        
        # Update persona reflex skill
        if results['avg_reaction_time'] < self.persona.calculate_target_reflex_time():
            self.persona.skills.reaction_time = min(1.0, self.persona.skills.reaction_time + 0.1)
        
        self.logger.info(f"Reflex training complete: {results['avg_reaction_time']*1000:.1f}ms avg")
        
        return results
    
    def run_tactical_training(self, num_scenarios: int = 20) -> Dict[str, Any]:
        """
        Strategic tactical decision-making training
        
        Args:
            num_scenarios: Number of tactical scenarios
            
        Returns:
            Tactical training results
        """
        self.logger.info(f"Starting tactical training: {num_scenarios} scenarios")
        
        decisions_made = []
        successful_decisions = 0
        
        for scenario_num in range(num_scenarios):
            # Generate tactical scenario
            situation = self._generate_tactical_scenario()
            
            # Make decision
            decision = self.persona.make_tactical_decision(
                situation['description'],
                situation['options'],
                situation['context']
            )
            
            # Simulate outcome (simplified)
            success = np.random.random() < (self.persona.skills.tactical_planning * 0.9)
            decision.success = success
            decision.outcome = "success" if success else "failure"
            
            if success:
                successful_decisions += 1
            
            decisions_made.append(decision)
        
        results = {
            'num_scenarios': num_scenarios,
            'successful_decisions': successful_decisions,
            'success_rate': successful_decisions / num_scenarios,
            'tactical_skill': self.persona.skills.tactical_planning,
            'patterns_learned': len(self.persona.tactical_patterns)
        }
        
        # Update tactical skill
        if results['success_rate'] > 0.7:
            self.persona.skills.tactical_planning = min(1.0, 
                self.persona.skills.tactical_planning + 0.05)
        
        self.logger.info(f"Tactical training complete: {results['success_rate']*100:.1f}% success")
        
        return results
    
    def _generate_tactical_scenario(self) -> Dict[str, Any]:
        """Generate a random tactical scenario"""
        scenarios = [
            {
                'description': "Enemy team pushing objective",
                'options': ["Aggressive counter", "Defensive hold", "Flank maneuver", "Tactical retreat"],
                'context': {'threat_level': 0.8, 'resources': {'ammo': 0.6}, 'allies': [1, 2]}
            },
            {
                'description': "Low resources, multiple enemies nearby",
                'options': ["Search for resources", "Engage enemies", "Hide and wait", "Call for backup"],
                'context': {'threat_level': 0.7, 'resources': {'ammo': 0.2, 'health': 0.3}, 'allies': []}
            },
            {
                'description': "Opportunity to capture objective",
                'options': ["Solo capture", "Wait for team", "Scout area first", "Secure perimeter"],
                'context': {'threat_level': 0.4, 'resources': {'ammo': 0.8}, 'allies': [1]}
            },
            {
                'description': "Team needs support at different locations",
                'options': ["Support location A", "Support location B", "Split team", "Coordinate timing"],
                'context': {'threat_level': 0.6, 'resources': {'ammo': 0.7}, 'allies': [1, 2, 3]}
            }
        ]
        
        return scenarios[np.random.randint(0, len(scenarios))]
    
    def _adapt_difficulty(self, performance_score: float):
        """Adapt difficulty based on performance"""
        if performance_score > 0.85:
            self.current_difficulty = min(1.0, self.current_difficulty + 0.1)
            self.logger.info(f"Increased difficulty to {self.current_difficulty:.2f}")
        elif performance_score < 0.5:
            self.current_difficulty = max(0.1, self.current_difficulty - 0.1)
            self.logger.info(f"Decreased difficulty to {self.current_difficulty:.2f}")
    
    def _update_ml_models(self, performance: Dict[str, Any]):
        """Update ML models with session performance data"""
        try:
            # Extract training data from performance
            if performance['session_type'] == '1P':
                # Update reflex model with 1P data
                if len(self.ml.reflex_training_data) > 100:
                    self.ml.train_reflex_model(self.ml.reflex_training_data[-100:])
            
            elif performance['session_type'] == '3P':
                # Update movement and spatial models
                pass
            
        except Exception as e:
            self.logger.error(f"Error updating ML models: {e}")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary"""
        session_duration = time.time() - self.session_start_time
        
        return {
            'session_duration': session_duration,
            'total_sessions': len(self.performance_history),
            'current_difficulty': self.current_difficulty,
            'persona_summary': self.persona.get_performance_summary(),
            'skill_improvement': self._calculate_skill_improvement(),
            'recent_performance': self.performance_history[-5:] if self.performance_history else []
        }
    
    def _calculate_skill_improvement(self) -> Dict[str, float]:
        """Calculate skill improvement over the session"""
        if len(self.skill_progression) < 2:
            return {}
        
        initial_skills = self.skill_progression[0]
        current_skills = self.skill_progression[-1]
        
        improvement = {}
        for skill_name in initial_skills:
            improvement[skill_name] = current_skills[skill_name] - initial_skills[skill_name]
        
        return improvement


class GamingTrainingProgram:
    """
    Complete training program with multiple sessions and progression tracking
    """
    
    def __init__(self, save_dir: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        
        # Initialize save directory
        if save_dir is None:
            save_dir = Path.home() / '.deep_tree_echo' / 'gaming_training'
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.persona = GamerPersona(name="Echo-Gamer-Girl", reflex_mode=ReflexMode.LIGHTNING)
        self.environment = UnrealEnvironment(level_name="Training_Arena")
        self.ml_system = MLSystem()
        self.optimizer = UnrealEngineOptimizer()
        
        # Spawn player
        self.player = self.environment.spawn_player_character("Echo_Avatar")
        
        # Training history
        self.sessions: List[GamingTrainingSession] = []
        self.program_start_time = time.time()
        
        self.logger.info("Gaming Training Program initialized")
    
    def run_comprehensive_training(self, duration_minutes: int = 30) -> Dict[str, Any]:
        """
        Run comprehensive training covering all aspects
        
        Args:
            duration_minutes: Total training duration in minutes
            
        Returns:
            Complete training results
        """
        self.logger.info(f"Starting comprehensive training: {duration_minutes} minutes")
        
        end_time = time.time() + (duration_minutes * 60)
        session_num = 0
        
        while time.time() < end_time:
            session_num += 1
            
            # Create new session
            session = GamingTrainingSession(
                self.persona, self.environment, self.ml_system
            )
            
            # Alternate between training types
            if session_num % 4 == 1:
                self.logger.info(f"Session {session_num}: 1P Coordination")
                session.run_1p_training(duration=60.0)
            elif session_num % 4 == 2:
                self.logger.info(f"Session {session_num}: 3P Coordination")
                session.run_3p_training(duration=90.0)
            elif session_num % 4 == 3:
                self.logger.info(f"Session {session_num}: Reflex Training")
                session.run_reflex_training(num_trials=50)
            else:
                self.logger.info(f"Session {session_num}: Tactical Training")
                session.run_tactical_training(num_scenarios=15)
            
            self.sessions.append(session)
            
            # Brief rest between sessions
            time.sleep(1)
        
        # Generate final report
        results = self._generate_final_report()
        
        # Save results
        self._save_training_results(results)
        
        return results
    
    def optimize_for_gaming_performance(self) -> Dict[str, Any]:
        """Optimize all systems for peak gaming performance"""
        self.logger.info("Optimizing for gaming performance...")
        
        # Optimize environment
        env_optimizations = self.optimizer.optimize_for_reflexes(
            self.environment, target_fps=120
        )
        
        # Optimize ML models
        ml_optimizations = self.ml_system.optimize_for_performance()
        
        # Set persona to lightning mode
        self.persona.reflex_mode = ReflexMode.LIGHTNING
        
        results = {
            'environment': env_optimizations,
            'ml_system': ml_optimizations,
            'persona': {
                'reflex_mode': self.persona.reflex_mode.value,
                'target_reaction_time': self.persona.calculate_target_reflex_time()
            }
        }
        
        self.logger.info("Optimization complete")
        return results
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final training report"""
        total_duration = time.time() - self.program_start_time
        
        report = {
            'program_duration_minutes': total_duration / 60,
            'total_sessions': len(self.sessions),
            'persona_final_state': self.persona.get_performance_summary(),
            'environment_state': self.environment.get_environment_state(),
            'skills': self.persona.skills.to_dict(),
            'performance_metrics': {
                'avg_reaction_time': self.persona.avg_reaction_time,
                'training_sessions_completed': len(self.persona.training_sessions),
                'tactical_patterns_learned': len(self.persona.tactical_patterns),
                'decisions_made': len(self.persona.decision_history)
            },
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate training recommendations"""
        recommendations = []
        
        skills = self.persona.skills
        
        if skills.aim_precision < 0.7:
            recommendations.append("Focus on aim training drills")
        
        if skills.reaction_time < 0.8:
            recommendations.append("Practice reflex training more frequently")
        
        if skills.tactical_planning < 0.7:
            recommendations.append("Study tactical scenarios and strategies")
        
        if skills.coordination_1p < skills.coordination_3p:
            recommendations.append("Increase 1P coordination training")
        elif skills.coordination_3p < skills.coordination_1p:
            recommendations.append("Increase 3P coordination training")
        
        if not recommendations:
            recommendations.append("Excellent performance! Continue current training regimen")
        
        return recommendations
    
    def _save_training_results(self, results: Dict[str, Any]):
        """Save training results to file"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = self.save_dir / f"training_results_{timestamp}.json"
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Training results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving training results: {e}")


# Convenience function for quick training
def quick_train_gamer_persona(duration_minutes: int = 30) -> Dict[str, Any]:
    """
    Quick start function for gaming training
    
    Args:
        duration_minutes: Training duration
        
    Returns:
        Training results
    """
    program = GamingTrainingProgram()
    
    # Optimize first
    program.optimize_for_gaming_performance()
    
    # Run training
    results = program.run_comprehensive_training(duration_minutes)
    
    return results
