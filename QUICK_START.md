# Quick Start Guide - Deep Tree Echo Snapshot v2
## Getting Started with the Mid-2025 Time Capsule

---

## 🚀 5-Minute Quick Start

### 1. Clone and Enter Repository

```bash
git clone https://github.com/orgitcog/echosurface-evolving.git
cd echosurface-evolving
```

### 2. Install Dependencies

```bash
# Install Python dependencies
pip3 install -r requirements.txt

# Install additional dependencies
pip3 install networkx aiohttp
```

### 3. Verify Installation

```bash
# Run verification script
python3 verify_snapshot.py
```

**Expected:** All tests should pass, status "FULLY OPERATIONAL" or "MOSTLY OPERATIONAL"  
(Note: Some optional features may show warnings in certain environments)

### 4. Explore the Documentation

Start with these files in order:
1. **SNAPSHOT_v2.md** - Overview of what's preserved
2. **EVOLUTION_TIMELINE.md** - The journey to this point
3. **FEATURES_2024-2025.md** - Key achievements
4. **README.md** - Full documentation

### 5. Try a Simple Example

```python
# Create and run this file: test_echo.py
from deep_tree_echo import DeepTreeEcho

# Initialize (without Julia for simplicity)
echo = DeepTreeEcho(use_julia=False)

# Create a tree
root = echo.create_tree("Hello from Deep Tree Echo!")
child1 = echo.add_child(root, "First child node")
child2 = echo.add_child(root, "Second child node")

# Propagate echoes
echo.propagate_echoes()

# Show results
print(f"Root echo value: {root.echo_value:.3f}")
print(f"Root spatial position: {root.spatial_context.position}")
print(f"Root has DET emotions: {root.det_state is not None}")
print(f"Tree depth: {len(root.children)}")
```

Run it:
```bash
python3 test_echo.py
```

---

## 📚 Exploring Different Systems

### Emotional Intelligence

```python
from differential_emotion_theory import DifferentialEmotionSystem, DETEmotion

# Create emotion system
det = DifferentialEmotionSystem(use_julia=False)

# Analyze text for emotions
emotions = det.content_to_det_emotion("This is exciting and wonderful!")

# Show joy and interest levels
print(f"Joy: {emotions[DETEmotion.JOY.value]:.2f}")
print(f"Interest: {emotions[DETEmotion.INTEREST.value]:.2f}")
```

### Memory System

```python
from memory_management import HypergraphMemory, MemoryNode, MemoryType

# Create memory system
memory = HypergraphMemory()

# Store a memory
node = MemoryNode(
    id="memory_001",
    content="Deep Tree Echo is an autonomous AI system",
    memory_type=MemoryType.DECLARATIVE,
    salience=0.8
)
memory.add_node(node)

# Retrieve memories by type
declarative_memories = memory.get_by_type(MemoryType.DECLARATIVE)
print(f"Found {len(declarative_memories)} declarative memories")
```

### Cognitive Architecture

```python
from cognitive_architecture import CognitiveArchitecture

# Initialize cognitive system
cog = CognitiveArchitecture()

# Generate goals
context = {"situation": "exploration", "complexity": "medium"}
goals = cog.generate_goals(context)

print(f"Generated {len(goals)} goals:")
for goal in goals:
    print(f"  - {goal}")
```

### Spatial Awareness

```python
from deep_tree_echo import DeepTreeEcho, SpatialContext

# Create system with spatial awareness
echo = DeepTreeEcho(use_julia=False)
echo.spatial_awareness_enabled = True

# Create node with specific position
root = echo.create_tree("3D Object")
root.spatial_context.position = (5.0, 3.0, 10.0)
root.spatial_context.orientation = (0.0, 45.0, 0.0)
root.spatial_context.field_of_view = 120.0

print(f"Position: {root.spatial_context.position}")
print(f"Orientation: {root.spatial_context.orientation}")
print(f"FOV: {root.spatial_context.field_of_view}°")
```

---

## 🎯 Common Tasks

### View Activity Logs

```bash
# Recent activity
tail -20 activity_logs/activity_stream.log

# All activity
cat activity_logs/activity_stream.log
```

### Check System Health

```python
from adaptive_heartbeat import AdaptiveHeartbeat

heartbeat = AdaptiveHeartbeat()
status = heartbeat.get_status()
print(f"System status: {status}")
```

### Export Memory State

```python
from memory_management import HypergraphMemory

memory = HypergraphMemory()
memory.save()  # Saves to echo_memory/ directory
```

---

## 🔍 Understanding the Architecture

### System Layers

```
┌─────────────────────────────────────┐
│     User Interface Layer            │
│  (GUI Dashboard, Web Dashboard)     │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│     Cognitive Layer                 │
│  (Goals, Personality, Emotions)     │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│     Memory Layer                    │
│  (Hypergraph, Multi-type Memory)    │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│     Processing Layer                │
│  (ML, Echo Propagation, Patterns)   │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│     Sensory-Motor Layer             │
│  (Vision, Mouse, Keyboard, Browser) │
└─────────────────────────────────────┘
```

### Key Files to Explore

| File | Purpose | Complexity |
|------|---------|------------|
| `deep_tree_echo.py` | Core system | ⭐⭐⭐ |
| `ml_system.py` | ML models | ⭐⭐ |
| `differential_emotion_theory.py` | Emotions | ⭐⭐⭐ |
| `memory_management.py` | Hypergraph memory | ⭐⭐⭐⭐ |
| `cognitive_architecture.py` | Cognition | ⭐⭐⭐ |
| `selenium_interface.py` | Browser control | ⭐⭐ |
| `activity_stream.py` | Activity tracking | ⭐ |
| `echo_evolution.py` | Self-evolution | ⭐⭐⭐⭐ |

---

## 🐛 Troubleshooting Quick Fixes

### Problem: Import Errors

```bash
# Reinstall all dependencies
pip3 install --force-reinstall -r requirements.txt
pip3 install networkx aiohttp
```

### Problem: TensorFlow Warnings

These are normal! The system works without GPU:
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
```

### Problem: X11/Display Errors

For headless environments:
```bash
# Use virtual display
export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x24 &

# Or skip GUI features
# Focus on core systems without graphical components
```

### Problem: Module Not Found

```bash
# Check if module is installed
pip3 show <module-name>

# Install if missing
pip3 install <module-name>
```

---

## 📖 Learning Paths

### Path 1: Beginner (Understanding Basics)

1. Read SNAPSHOT_v2.md overview
2. Run verify_snapshot.py
3. Try the 5-minute example above
4. Explore emotional system example
5. Read Deep-Tree-Echo-Persona.md

**Time:** 30 minutes

### Path 2: Developer (Hands-on Coding)

1. Complete Path 1
2. Read FEATURES_2024-2025.md
3. Experiment with all system examples above
4. Modify examples to add new features
5. Read source code of key files
6. Create your own integration

**Time:** 2-3 hours

### Path 3: Researcher (Deep Understanding)

1. Complete Path 1 & 2
2. Read EVOLUTION_TIMELINE.md
3. Study all main system files
4. Read research papers referenced
5. Analyze integration patterns
6. Design experiments
7. Contribute improvements

**Time:** Several days

---

## 🎓 Educational Exercises

### Exercise 1: Emotion-Spatial Coupling

**Goal:** Understand how emotions affect spatial perception

```python
from deep_tree_echo import DeepTreeEcho
from differential_emotion_theory import DETEmotion

echo = DeepTreeEcho(use_julia=False)
root = echo.create_tree("Emotional Space Test")

# Set high joy
root.det_state.det_emotions[DETEmotion.JOY.value] = 0.9

# Check field of view (should expand with joy)
initial_fov = root.spatial_context.field_of_view
print(f"Initial FOV: {initial_fov}")

# TODO: Implement emotion-to-space update
# echo.update_spatial_from_emotion(root)
# print(f"Updated FOV: {root.spatial_context.field_of_view}")
```

### Exercise 2: Memory Pattern Recognition

**Goal:** Store and retrieve related memories

```python
from memory_management import HypergraphMemory, MemoryNode, MemoryType, MemoryEdge

memory = HypergraphMemory()

# Create concept nodes
concepts = []
for i, concept in enumerate(["AI", "Machine Learning", "Neural Networks"]):
    node = MemoryNode(
        id=f"concept_{i}",
        content=concept,
        memory_type=MemoryType.SEMANTIC
    )
    memory.add_node(node)
    concepts.append(node.id)

# Create relationships
edge1 = MemoryEdge(concepts[1], concepts[0], "is_part_of")
edge2 = MemoryEdge(concepts[2], concepts[1], "implements")
memory.add_edge(edge1)
memory.add_edge(edge2)

# Explore connections
# TODO: Implement graph traversal
```

### Exercise 3: Echo Propagation Analysis

**Goal:** Understand how echo values propagate

```python
from deep_tree_echo import DeepTreeEcho

echo = DeepTreeEcho(use_julia=False)
root = echo.create_tree("Root")

# Create tree structure
for i in range(3):
    child = echo.add_child(root, f"Child {i}")
    for j in range(2):
        grandchild = echo.add_child(child, f"Grandchild {i}.{j}")

# Propagate and analyze
echo.propagate_echoes()

def print_tree(node, indent=0):
    print("  " * indent + f"{node.content}: {node.echo_value:.3f}")
    for child in node.children:
        print_tree(child, indent + 1)

print_tree(root)
```

---

## 🔗 Next Steps

After exploring this snapshot:

1. **Experiment:** Modify examples and see what happens
2. **Integrate:** Combine different systems in novel ways
3. **Extend:** Add new features or capabilities
4. **Document:** Write about your discoveries
5. **Share:** Contribute back to the community
6. **Research:** Dive into specific areas of interest
7. **Build:** Create applications using Deep Tree Echo

---

## 📞 Getting Help

### Resources
- **Documentation:** All .md files in repository
- **Code Examples:** test_*.py files
- **Source Code:** All .py files with inline comments
- **Verification:** verify_snapshot.py for system health

### Community
- **GitHub Issues:** Report bugs or ask questions
- **Discussions:** Share ideas and insights
- **Pull Requests:** Contribute improvements

---

## ✨ Remember

Deep Tree Echo is not just code—it's a **living cognitive system** that combines:
- 🧠 Intelligence (ML, patterns, reasoning)
- 💚 Emotion (DET, emotional dynamics)
- 🎮 Embodiment (3D spatial awareness)
- 🌐 Action (browser automation, I/O)
- 🔄 Evolution (self-improvement)
- 💾 Memory (hypergraph knowledge)

**Explore with curiosity, experiment with courage, and enjoy the journey!**

---

*Quick Start Guide for Deep Tree Echo Snapshot v2*  
*Last Updated: January 2, 2026*  
*Status: OPERATIONAL*

🎉 **Welcome to the Deep Tree Echo community!** 🎉
