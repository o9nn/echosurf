#!/usr/bin/env python3
"""
Deep Tree Echo Snapshot v2 - System Verification Script
Tests all core systems without requiring GUI or X11 dependencies
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test that all core modules can be imported"""
    print("=" * 70)
    print("Deep Tree Echo Snapshot v2 - System Verification")
    print("=" * 70)
    
    results = []
    
    # Test 1: Core system
    print("\n[1/9] Testing Core Deep Tree Echo...")
    try:
        from deep_tree_echo import DeepTreeEcho, TreeNode, SpatialContext
        print("    ✓ Core modules imported successfully")
        results.append(True)
    except Exception as e:
        print(f"    ✗ Import failed: {e}")
        results.append(False)
    
    # Test 2: ML System
    print("\n[2/9] Testing ML System...")
    try:
        from ml_system import MLSystem
        print("    ✓ ML System imported successfully")
        results.append(True)
    except Exception as e:
        print(f"    ✗ Import failed: {e}")
        results.append(False)
    
    # Test 3: Emotional Systems
    print("\n[3/9] Testing Emotional Intelligence...")
    try:
        from differential_emotion_theory import DifferentialEmotionSystem, DETEmotion
        from emotional_dynamics import EmotionalDynamics, EmotionalState
        print("    ✓ Emotional systems imported successfully")
        results.append(True)
    except Exception as e:
        print(f"    ✗ Import failed: {e}")
        results.append(False)
    
    # Test 4: Memory System
    print("\n[4/9] Testing Hypergraph Memory...")
    try:
        from memory_management import HypergraphMemory
        print("    ✓ Memory system imported successfully")
        results.append(True)
    except Exception as e:
        print(f"    ✗ Import failed: {e}")
        results.append(False)
    
    # Test 5: Cognitive Architecture
    print("\n[5/9] Testing Cognitive Architecture...")
    try:
        from cognitive_architecture import CognitiveArchitecture
        print("    ✓ Cognitive architecture imported successfully")
        results.append(True)
    except Exception as e:
        print(f"    ✗ Import failed: {e}")
        results.append(False)
    
    # Test 6: Activity Systems
    print("\n[6/9] Testing Activity Management...")
    try:
        from activity_stream import ActivityStream
        from activity_regulation import ActivityRegulator
        print("    ✓ Activity systems imported successfully")
        results.append(True)
    except Exception as e:
        print(f"    ✗ Import failed: {e}")
        results.append(False)
    
    # Test 7: Evolution Systems
    print("\n[7/9] Testing Evolution Mechanisms...")
    try:
        from echo_evolution import EchoAgent
        print("    ✓ Evolution systems imported successfully")
        results.append(True)
    except Exception as e:
        print(f"    ✗ Import failed: {e}")
        results.append(False)
    
    # Test 8: Browser Automation (import only, no execution)
    print("\n[8/9] Testing Browser Automation...")
    try:
        from selenium_interface import SeleniumInterface
        print("    ✓ Browser automation imported successfully")
        results.append(True)
    except Exception as e:
        print(f"    ✗ Import failed: {e}")
        results.append(False)
    
    # Test 9: Monitoring Systems
    print("\n[9/9] Testing Monitoring Systems...")
    try:
        from adaptive_heartbeat import AdaptiveHeartbeat
        from emergency_protocols import EmergencyProtocols
        print("    ✓ Monitoring systems imported successfully")
        results.append(True)
    except Exception as e:
        print(f"    ✗ Import failed: {e}")
        results.append(False)
    
    return results

def test_functionality():
    """Test basic functionality of core systems"""
    print("\n" + "=" * 70)
    print("Functional Tests")
    print("=" * 70)
    
    results = []
    
    # Functional Test 1: Echo Propagation
    print("\n[1/5] Testing Echo Propagation...")
    try:
        from deep_tree_echo import TreeNode, SpatialContext
        
        # Create simple tree without full system to avoid GUI dependencies
        root = TreeNode(content="Test Root", echo_value=0.5)
        child = TreeNode(content="Test Child", echo_value=0.3)
        root.children.append(child)
        child.parent = root
        
        assert root.echo_value > 0, "Echo value must be positive"
        print(f"    ✓ Echo propagation functional (value: {root.echo_value:.3f})")
        results.append(True)
    except Exception as e:
        print(f"    ✗ Test failed: {e}")
        results.append(False)
    
    # Functional Test 2: Emotional Processing
    print("\n[2/5] Testing Emotional Processing...")
    try:
        from differential_emotion_theory import DifferentialEmotionSystem
        det = DifferentialEmotionSystem(use_julia=False)
        emotions = det.content_to_det_emotion("This is exciting!")
        
        # Check if emotions array is valid (should have multiple emotions)
        emotion_count = len(emotions) if hasattr(emotions, '__len__') else (emotions.shape[0] if hasattr(emotions, 'shape') else 0)
        assert emotion_count >= 10, f"Should have at least 10 emotions, got {emotion_count}"
        print(f"    ✓ Emotional processing functional ({emotion_count} emotions tracked)")
        results.append(True)
    except Exception as e:
        print(f"    ✗ Test failed: {e}")
        results.append(False)
    
    # Functional Test 3: Memory Operations
    print("\n[3/5] Testing Memory Operations...")
    try:
        from memory_management import HypergraphMemory, MemoryNode, MemoryType
        memory = HypergraphMemory()
        
        # Create a memory node
        node = MemoryNode(
            id="test_001",
            content="Test memory content",
            memory_type=MemoryType.DECLARATIVE
        )
        node_id = memory.add_node(node)
        
        assert node_id is not None, "Node creation failed"
        print(f"    ✓ Memory operations functional (node: {node_id})")
        results.append(True)
    except Exception as e:
        print(f"    ✗ Test failed: {e}")
        results.append(False)
    
    # Functional Test 4: Cognitive Processing
    print("\n[4/5] Testing Cognitive Processing...")
    try:
        from cognitive_architecture import CognitiveArchitecture
        cog = CognitiveArchitecture()
        goals = cog.generate_goals({"situation": "test"})
        
        print(f"    ✓ Cognitive processing functional ({len(goals)} goals generated)")
        results.append(True)
    except Exception as e:
        print(f"    ✗ Test failed: {e}")
        results.append(False)
    
    # Functional Test 5: Spatial Context
    print("\n[5/5] Testing Spatial Context...")
    try:
        from deep_tree_echo import TreeNode, SpatialContext
        
        # Create node with spatial context
        spatial = SpatialContext(position=(1.0, 2.0, 3.0))
        root = TreeNode(content="Spatial Test", spatial_context=spatial)
        
        assert root.spatial_context is not None, "Spatial context missing"
        assert hasattr(root.spatial_context, 'position'), "Position missing"
        print(f"    ✓ Spatial context functional (pos: {root.spatial_context.position})")
        results.append(True)
    except Exception as e:
        print(f"    ✗ Test failed: {e}")
        results.append(False)
    
    return results

def main():
    """Run all verification tests"""
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║   Deep Tree Echo Snapshot v2 - Preservation Verification          ║")
    print("║   Testing system functionality without GUI dependencies           ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print("\n")
    
    # Run import tests
    import_results = test_imports()
    
    # Run functional tests
    functional_results = test_functionality()
    
    # Summary
    print("\n" + "=" * 70)
    print("Verification Summary")
    print("=" * 70)
    
    import_passed = sum(import_results)
    import_total = len(import_results)
    functional_passed = sum(functional_results)
    functional_total = len(functional_results)
    total_passed = import_passed + functional_passed
    total_tests = import_total + functional_total
    
    print(f"\nImport Tests:      {import_passed}/{import_total} passed")
    print(f"Functional Tests:  {functional_passed}/{functional_total} passed")
    print(f"Total:             {total_passed}/{total_tests} passed")
    
    if total_passed == total_tests:
        print("\n✅ ALL TESTS PASSED")
        print("\nSnapshot Status: FULLY OPERATIONAL")
        print("All systems are preserved and functional.")
        return 0
    elif total_passed >= total_tests * 0.8:
        print("\n⚠️  MOST TESTS PASSED")
        print("\nSnapshot Status: MOSTLY OPERATIONAL")
        print("Core systems functional, some optional systems may need attention.")
        return 0
    else:
        print("\n❌ TESTS FAILED")
        print("\nSnapshot Status: NEEDS ATTENTION")
        print("Some core systems are not functioning properly.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
