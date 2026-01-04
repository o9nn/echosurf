#!/usr/bin/env python3
"""
Example: Quick start with Echo Gamer-Girl Persona

This demonstrates the gaming training capabilities of Deep Tree Echo.
"""

import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Main demonstration function"""
    
    print("=" * 70)
    print("   Echo Gamer-Girl Persona Training Demo")
    print("=" * 70)
    print()
    
    try:
        # Import gaming modules
        logger.info("Loading gaming modules...")
        from gamer_persona import GamerPersona, ReflexMode
        from unreal_integration import UnrealEnvironment, UnrealEngineOptimizer
        from gaming_training import GamingTrainingSession
        from ml_system import MLSystem
        
        print("✓ Modules loaded successfully\n")
        
        # 1. Create Gamer Persona
        print("1. Initializing Gamer-Girl Persona...")
        persona = GamerPersona(name="Echo-Gamer-Girl", reflex_mode=ReflexMode.LIGHTNING)
        print(f"   Name: {persona.name}")
        print(f"   Reflex Mode: {persona.reflex_mode.value}")
        print(f"   Target Reaction Time: {persona.calculate_target_reflex_time()*1000:.1f}ms")
        print()
        
        # 2. Create Unreal Environment
        print("2. Setting up Unreal Engine Training Arena...")
        environment = UnrealEnvironment(level_name="Training_Arena")
        player = environment.spawn_player_character("Echo_Avatar", location=(0, 0, 100))
        print(f"   Level: {environment.level_name}")
        print(f"   Player: {player.name} spawned at {player.transform.location.to_tuple()}")
        print()
        
        # 3. Initialize ML System
        print("3. Initializing ML Performance Optimizer...")
        ml_system = MLSystem()
        print("   ✓ Visual model ready")
        print("   ✓ Reflex model ready")
        print("   ✓ Tactical model ready")
        print()
        
        # 4. Optimize for Performance
        print("4. Optimizing for Gaming Performance...")
        optimizer = UnrealEngineOptimizer()
        optimizations = optimizer.optimize_for_reflexes(environment, target_fps=120)
        print(f"   Target FPS: {optimizations['target_fps']}")
        print(f"   Delta Time: {optimizations['delta_time']*1000:.2f}ms")
        print(f"   Optimized Actors: {optimizations['actors_optimized']}")
        print()
        
        # 5. Create Training Session
        print("5. Creating Training Session...")
        session = GamingTrainingSession(persona, environment, ml_system)
        print("   ✓ Session initialized")
        print()
        
        # 6. Run Quick Reflex Training
        print("6. Running Quick Reflex Training (10 trials)...")
        reflex_results = session.run_reflex_training(num_trials=10)
        print(f"   Average Reaction Time: {reflex_results['avg_reaction_time']*1000:.1f}ms")
        print(f"   Minimum Reaction Time: {reflex_results['min_reaction_time']*1000:.1f}ms")
        print(f"   Average Accuracy: {reflex_results['avg_accuracy']*100:.1f}%")
        print(f"   Target Time: {reflex_results['target_time']*1000:.1f}ms")
        print()
        
        # 7. Run 1P Training
        print("7. Running 1P Coordination Training...")
        scenario_1p = environment.create_training_scenario_1p(num_targets=5)
        results_1p = session.run_1p_training(duration=10.0, num_targets=5)
        print(f"   Hits: {results_1p['results']['hits']}")
        print(f"   Misses: {results_1p['results']['misses']}")
        print(f"   Accuracy: {results_1p['results']['accuracy']*100:.1f}%")
        print(f"   Coordination Score: {results_1p['results']['coordination_score']*100:.1f}%")
        print()
        
        # 8. Run Tactical Training
        print("8. Running Tactical Decision Training...")
        tactical_results = session.run_tactical_training(num_scenarios=5)
        print(f"   Scenarios Completed: {tactical_results['num_scenarios']}")
        print(f"   Successful Decisions: {tactical_results['successful_decisions']}")
        print(f"   Success Rate: {tactical_results['success_rate']*100:.1f}%")
        print(f"   Patterns Learned: {tactical_results['patterns_learned']}")
        print()
        
        # 9. Display Final Performance
        print("9. Final Performance Summary:")
        summary = persona.get_performance_summary()
        print(f"   Persona: {summary['name']}")
        print(f"   Reflex Mode: {summary['reflex_mode']}")
        print(f"   Training Sessions: {summary['training_sessions']}")
        print(f"   Decisions Made: {summary['decisions_made']}")
        print()
        print("   Skills:")
        for skill, value in summary['skills'].items():
            bar_length = int(value * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"     {skill:20s} {bar} {value*100:5.1f}%")
        print()
        
        # 10. ML Performance Check
        print("10. ML Performance Metrics:")
        ml_perf = ml_system.optimize_for_performance()
        print(f"   Average Inference Time: {ml_perf['average_inference_time_ms']:.2f}ms")
        print(f"   Reflex Model Optimized: {'✓' if ml_perf['reflex_model_optimized'] else '✗'}")
        print(f"   Tactical Model Optimized: {'✓' if ml_perf['tactical_model_optimized'] else '✗'}")
        print()
        
        # Summary
        print("=" * 70)
        print("   Training Complete! Echo Gamer-Girl is ready for action! 🎮⚡")
        print("=" * 70)
        print()
        print("Next Steps:")
        print("  • Run comprehensive training: echo.train_gaming_skills(duration_minutes=30)")
        print("  • Export training data: environment.export_to_json('training_data.json')")
        print("  • Continue skill development with focused training sessions")
        print()
        
        return 0
        
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        print("\n❌ Error: Required dependencies not installed")
        print("   Please install: pip install -r requirements.txt")
        return 1
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
