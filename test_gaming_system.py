"""
Tests for gaming persona and training system
"""

import unittest
import numpy as np
from gamer_persona import (
    GamerPersona, ReflexMode, GamingSkills,
    GamePerspective, AvatarState
)
from unreal_integration import (
    UnrealEnvironment, UEVector, UERotator,
    ActorType, MovementMode
)
from gaming_training import GamingTrainingSession, quick_train_gamer_persona
from ml_system import MLSystem


class TestGamerPersona(unittest.TestCase):
    """Test gamer persona functionality"""
    
    def setUp(self):
        """Set up test persona"""
        self.persona = GamerPersona(name="Test_Gamer", reflex_mode=ReflexMode.LIGHTNING)
    
    def test_persona_initialization(self):
        """Test persona is initialized correctly"""
        self.assertEqual(self.persona.name, "Test_Gamer")
        self.assertEqual(self.persona.reflex_mode, ReflexMode.LIGHTNING)
        self.assertIsInstance(self.persona.skills, GamingSkills)
        self.assertIsInstance(self.persona.avatar, AvatarState)
    
    def test_reflex_target_time(self):
        """Test reflex target time calculation"""
        lightning_time = self.persona.calculate_target_reflex_time()
        self.assertLess(lightning_time, 0.05)  # Less than 50ms
        
        self.persona.reflex_mode = ReflexMode.COMPETITIVE
        comp_time = self.persona.calculate_target_reflex_time()
        self.assertLess(comp_time, 0.10)  # Less than 100ms
        self.assertGreater(comp_time, lightning_time)
    
    def test_1p_coordination_training(self):
        """Test first-person coordination training"""
        scenario = {
            'targets': [
                {'position': (100, 0, 0), 'class': 'enemy'},
                {'position': (0, 100, 0), 'class': 'enemy'},
                {'position': (-100, 0, 0), 'class': 'enemy'}
            ],
            'obstacles': [],
            'time_limit': 10.0
        }
        
        results = self.persona.train_coordination_1p(scenario)
        
        self.assertIn('hits', results)
        self.assertIn('misses', results)
        self.assertIn('accuracy', results)
        self.assertIn('coordination_score', results)
        self.assertGreaterEqual(results['accuracy'], 0.0)
        self.assertLessEqual(results['accuracy'], 1.0)
    
    def test_3p_coordination_training(self):
        """Test third-person coordination training"""
        scenario = {
            'navigation_points': [
                {'position': (100, 100, 0), 'camera_angle': 45},
                {'position': (200, 0, 0), 'camera_angle': 60},
                {'position': (0, 200, 0), 'camera_angle': 30}
            ],
            'camera_angles': [45, 60, 30],
            'spatial_challenges': []
        }
        
        results = self.persona.train_coordination_3p(scenario)
        
        self.assertIn('navigation_accuracy', results)
        self.assertIn('spatial_awareness_score', results)
        self.assertIn('coordination_score', results)
        self.assertGreaterEqual(results['coordination_score'], 0.0)
        self.assertLessEqual(results['coordination_score'], 1.0)
    
    def test_tactical_decision_making(self):
        """Test tactical decision making"""
        situation = "Enemy approaching from multiple directions"
        options = ["Aggressive push", "Defensive position", "Tactical retreat", "Flank maneuver"]
        context = {
            'threat_level': 0.8,
            'resources': {'ammo': 0.5, 'health': 0.7},
            'allies': [1, 2]
        }
        
        decision = self.persona.make_tactical_decision(situation, options, context)
        
        self.assertIn(decision.chosen_action, options)
        self.assertGreaterEqual(decision.confidence, 0.0)
        self.assertLessEqual(decision.confidence, 1.0)
    
    def test_avatar_embodiment(self):
        """Test avatar embodiment updates"""
        sensory_input = {
            'position': [100.0, 200.0, 50.0],
            'rotation': [0.0, 90.0, 0.0],
            'velocity': [10.0, 5.0, 0.0],
            'health': 0.8,
            'stamina': 0.6
        }
        
        success = self.persona.update_avatar_embodiment(sensory_input)
        
        self.assertTrue(success)
        self.assertEqual(self.persona.avatar.position, (100.0, 200.0, 50.0))
        self.assertEqual(self.persona.avatar.rotation, (0.0, 90.0, 0.0))
        self.assertEqual(self.persona.avatar.health, 0.8)


class TestUnrealIntegration(unittest.TestCase):
    """Test Unreal Engine integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.environment = UnrealEnvironment(level_name="Test_Level")
    
    def test_environment_initialization(self):
        """Test environment initializes correctly"""
        self.assertEqual(self.environment.level_name, "Test_Level")
        self.assertEqual(len(self.environment.actors), 0)
        self.assertIsNone(self.environment.player_character)
    
    def test_spawn_player_character(self):
        """Test spawning player character"""
        player = self.environment.spawn_player_character("Test_Player", (0, 0, 100))
        
        self.assertIsNotNone(player)
        self.assertEqual(player.name, "Test_Player")
        self.assertEqual(player.transform.location.to_tuple(), (0, 0, 100))
        self.assertEqual(len(self.environment.actors), 1)
    
    def test_spawn_targets(self):
        """Test spawning targets"""
        target1 = self.environment.spawn_target("Target_1", (100, 0, 0))
        target2 = self.environment.spawn_target("Target_2", (0, 100, 0))
        
        self.assertEqual(len(self.environment.actors), 2)
        self.assertIn("Target", target1.tags)
        self.assertTrue(target1.properties['is_target'])
    
    def test_character_movement(self):
        """Test character movement"""
        player = self.environment.spawn_player_character("Test_Player")
        initial_pos = player.transform.location.to_tuple()
        
        # Move forward
        self.environment.move_character(player, (1.0, 0.0), delta_time=0.1)
        
        final_pos = player.transform.location.to_tuple()
        
        # Position should have changed
        self.assertNotEqual(initial_pos, final_pos)
    
    def test_character_rotation(self):
        """Test character rotation"""
        player = self.environment.spawn_player_character("Test_Player")
        initial_yaw = player.transform.rotation.yaw
        
        # Rotate right
        self.environment.rotate_character(player, 45.0, 0.0, delta_time=1.0)
        
        final_yaw = player.transform.rotation.yaw
        
        # Yaw should have increased
        self.assertGreater(final_yaw, initial_yaw)
    
    def test_1p_training_scenario(self):
        """Test 1P training scenario creation"""
        self.environment.spawn_player_character("Player")
        scenario = self.environment.create_training_scenario_1p(num_targets=5)
        
        self.assertEqual(scenario['type'], '1P_training')
        self.assertEqual(scenario['num_targets'], 5)
        self.assertEqual(len(scenario['targets']), 5)
    
    def test_3p_training_scenario(self):
        """Test 3P training scenario creation"""
        self.environment.spawn_player_character("Player")
        scenario = self.environment.create_training_scenario_3p(num_waypoints=4)
        
        self.assertEqual(scenario['type'], '3P_training')
        self.assertEqual(scenario['num_waypoints'], 4)
        self.assertEqual(len(scenario['navigation_points']), 4)
    
    def test_ue_vector_operations(self):
        """Test UE vector operations"""
        v1 = UEVector(1.0, 0.0, 0.0)
        v2 = UEVector(0.0, 1.0, 0.0)
        
        # Test magnitude
        self.assertEqual(v1.magnitude(), 1.0)
        
        # Test dot product
        self.assertEqual(v1.dot(v2), 0.0)
        
        # Test cross product
        cross = v1.cross(v2)
        self.assertEqual(cross.to_tuple(), (0.0, 0.0, 1.0))
    
    def test_ue_rotator_to_vectors(self):
        """Test rotator to direction vector conversion"""
        rotator = UERotator(0.0, 90.0, 0.0)
        
        forward = rotator.to_forward_vector()
        right = rotator.to_right_vector()
        
        # At 90 degrees yaw, forward should point in Y direction
        self.assertAlmostEqual(forward.y, 1.0, places=5)
        self.assertAlmostEqual(forward.x, 0.0, places=5)


class TestMLSystemGaming(unittest.TestCase):
    """Test ML system gaming features"""
    
    def setUp(self):
        """Set up test ML system"""
        self.ml = MLSystem()
    
    def test_reflex_model_exists(self):
        """Test reflex model is created"""
        self.assertIsNotNone(self.ml.reflex_model)
    
    def test_tactical_model_exists(self):
        """Test tactical model is created"""
        self.assertIsNotNone(self.ml.tactical_model)
    
    def test_reflex_prediction(self):
        """Test reflex response prediction"""
        target_pos = (100.0, 200.0)
        current_pos = (0.0, 0.0)
        
        response = self.ml.predict_reflex_response(
            target_pos, current_pos,
            velocity=(0, 0),
            threat_level=0.8,
            urgency=1.0
        )
        
        self.assertIn('optimal_position', response)
        self.assertIn('confidence', response)
        self.assertIn('estimated_reaction_time', response)
        self.assertIn('should_execute', response)
    
    def test_tactical_prediction(self):
        """Test tactical action prediction"""
        # Create random situation features
        situation_features = np.random.random(16)
        
        prediction = self.ml.predict_tactical_action(situation_features)
        
        self.assertIn('action_index', prediction)
        self.assertIn('confidence', prediction)
        self.assertIn('action_scores', prediction)
        self.assertGreaterEqual(prediction['action_index'], 0)
        self.assertLess(prediction['action_index'], 8)
    
    def test_performance_optimization(self):
        """Test performance optimization"""
        results = self.ml.optimize_for_performance()
        
        self.assertIn('average_inference_time_ms', results)
        self.assertIn('recommendations', results)
        
        # Should be fast enough for gaming (less than target)
        # Note: Actual performance depends on hardware; this is a reasonable check
        self.assertLess(results['average_inference_time_ms'], 50)  # Less than 50ms is acceptable
        
        # Log actual performance for information
        print(f"\n  Actual ML inference time: {results['average_inference_time_ms']:.2f}ms")
        print(f"  Target: <{self.ml.TARGET_INFERENCE_MS}ms for competitive gaming")


class TestGamingTraining(unittest.TestCase):
    """Test gaming training system"""
    
    def setUp(self):
        """Set up test training session"""
        self.persona = GamerPersona(reflex_mode=ReflexMode.COMPETITIVE)
        self.environment = UnrealEnvironment()
        self.ml = MLSystem()
        self.environment.spawn_player_character("Test_Player")
        
        self.session = GamingTrainingSession(
            self.persona, self.environment, self.ml
        )
    
    def test_session_initialization(self):
        """Test training session initializes correctly"""
        self.assertIsNotNone(self.session.persona)
        self.assertIsNotNone(self.session.environment)
        self.assertIsNotNone(self.session.ml)
        self.assertEqual(len(self.session.performance_history), 0)
    
    def test_1p_training_session(self):
        """Test running 1P training session"""
        performance = self.session.run_1p_training(duration=10.0, num_targets=3)
        
        self.assertEqual(performance['session_type'], '1P')
        self.assertIn('results', performance)
        self.assertIn('difficulty', performance)
        self.assertEqual(len(self.session.performance_history), 1)
    
    def test_3p_training_session(self):
        """Test running 3P training session"""
        performance = self.session.run_3p_training(duration=10.0, num_waypoints=3)
        
        self.assertEqual(performance['session_type'], '3P')
        self.assertIn('results', performance)
        self.assertEqual(len(self.session.performance_history), 1)
    
    def test_reflex_training_session(self):
        """Test reflex training"""
        results = self.session.run_reflex_training(num_trials=10)
        
        self.assertIn('avg_reaction_time', results)
        self.assertIn('avg_accuracy', results)
        self.assertIn('num_trials', results)
        self.assertEqual(results['num_trials'], 10)
    
    def test_tactical_training_session(self):
        """Test tactical training"""
        results = self.session.run_tactical_training(num_scenarios=5)
        
        self.assertIn('success_rate', results)
        self.assertIn('num_scenarios', results)
        self.assertEqual(results['num_scenarios'], 5)
        self.assertGreaterEqual(results['success_rate'], 0.0)
        self.assertLessEqual(results['success_rate'], 1.0)
    
    def test_session_summary(self):
        """Test session summary generation"""
        # Run some training first
        self.session.run_reflex_training(num_trials=5)
        
        summary = self.session.get_session_summary()
        
        self.assertIn('session_duration', summary)
        self.assertIn('total_sessions', summary)
        self.assertIn('persona_summary', summary)
        self.assertGreater(summary['total_sessions'], 0)


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
