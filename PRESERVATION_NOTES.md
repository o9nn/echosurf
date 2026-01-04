# Preservation Notes for Deep Tree Echo Snapshot v2
## Instructions for Future Restoration and Operation

---

## 📋 Document Purpose

This document provides practical instructions for:
1. Restoring the Deep Tree Echo system from this preserved snapshot
2. Verifying that all components function as they did in mid-2025
3. Troubleshooting common issues
4. Understanding system dependencies
5. Maintaining the system in its preserved state

---

## 🔧 System Requirements

### Hardware Requirements

#### Minimum
- **CPU:** 4 cores, 2.0 GHz
- **RAM:** 8 GB
- **Storage:** 10 GB free space
- **Network:** Broadband internet connection

#### Recommended
- **CPU:** 8+ cores, 3.0+ GHz
- **RAM:** 16+ GB
- **Storage:** 20+ GB SSD
- **Network:** High-speed internet
- **GPU:** CUDA-compatible GPU (optional, for ML acceleration)

### Software Requirements

#### Operating System
- **Primary:** Ubuntu 20.04+ or Debian-based Linux
- **Secondary:** macOS 11+ (most features)
- **Limited:** Windows 10+ (some features may not work)

**Note:** Full functionality requires Linux with X11 display support.

#### Python Environment
- **Python:** 3.12+ (tested with 3.12.3)
- **pip:** Latest version
- **venv:** For virtual environments

#### System Packages (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y \
    python3 python3-pip python3-venv \
    libgtk-3-dev python3-tk \
    gnome-screenshot \
    xvfb \
    chromium-browser chromium-chromedriver \
    git
```

#### Browser Requirements
- **Chrome/Chromium:** For Selenium automation
- **Firefox:** Alternative browser support
- **Playwright browsers:** Auto-installed by Playwright

---

## 📦 Installation Steps

### 1. Clone Repository

```bash
# Clone the repository
git clone https://github.com/orgitcog/echosurface-evolving.git
cd echosurface-evolving

# Checkout the snapshot branch/tag if needed
git checkout <snapshot-tag>
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows
```

### 3. Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium
```

**Expected time:** 5-10 minutes depending on network speed.

### 4. Create Configuration Files

```bash
# Copy environment template
cp .env.template .env

# Edit with your credentials
nano .env  # or your preferred editor
```

Required `.env` variables:
```bash
OPENAI_API_KEY=your_openai_key_here
CHATGPT_EMAIL=your_chatgpt_email
CHATGPT_PASSWORD=your_chatgpt_password
GITHUB_TOKEN=your_github_token  # For self-improvement features
```

### 5. Create Required Directories

```bash
# Create profile directory
mkdir -p deep_tree_echo_profile

# Create activity logs directory (if not exists)
mkdir -p activity_logs

# Create browser data directories
mkdir -p browser_data chrome_user_data
```

### 6. Verify Installation

```bash
# Run environment verification
python3 verify_environment.py

# Expected output: All checks should pass
```

---

## ✅ Verification Procedures

### Quick Smoke Test

```bash
# Test 1: Import core module
python3 -c "from deep_tree_echo import DeepTreeEcho; print('✓ Core import successful')"

# Test 2: Test ML system
python3 -c "from ml_system import MLSystem; print('✓ ML system import successful')"

# Test 3: Test emotional system
python3 -c "from differential_emotion_theory import DifferentialEmotionSystem; print('✓ Emotion system successful')"

# Test 4: Test memory system
python3 -c "from memory_management import HypergraphMemory; print('✓ Memory system successful')"
```

### Component Tests

```bash
# Run individual test files
python3 test_deep_tree_echo.py
python3 test_ml_system.py
python3 test_emotional_deep_tree.py
python3 test_differential_emotion.py
```

**Expected:** Tests should complete without errors (warnings are okay).

### Dashboard Tests

#### GUI Dashboard (requires X11)
```bash
# Test GUI dashboard launch
python3 launch_gui.py
# or
python3 fix_locale_gui.py

# Expected: GUI window should open
# Try: Navigate through tabs, check for errors
```

#### Web Dashboard
```bash
# Launch web dashboard
python3 web_gui.py &

# Test in browser
curl http://localhost:5000

# Expected: HTML response
```

### Browser Automation Test

```bash
# Create test script
cat > test_browser.py << 'EOF'
from selenium_interface import SeleniumInterface
import sys

chat = SeleniumInterface()
if chat.init():
    print("✓ Browser initialized")
    chat.close()
    sys.exit(0)
else:
    print("✗ Browser initialization failed")
    sys.exit(1)
EOF

python3 test_browser.py
```

**Expected:** Browser should launch and close without errors.

---

## 🔍 Troubleshooting Guide

### Common Issues and Solutions

#### Issue 1: Import Errors

**Symptom:**
```
ModuleNotFoundError: No module named 'XXX'
```

**Solution:**
```bash
# Reinstall requirements
pip install --force-reinstall -r requirements.txt

# Verify specific package
pip show <package-name>
```

#### Issue 2: TensorFlow/GPU Issues

**Symptom:**
```
Could not load dynamic library 'libcudart.so.11.0'
```

**Solution:**
```bash
# This is a warning, not an error
# To use GPU:
pip install tensorflow-gpu==2.19.0

# To ignore (CPU only):
export TF_CPP_MIN_LOG_LEVEL=2
```

#### Issue 3: X11 Display Issues

**Symptom:**
```
_tkinter.TclError: no display name and no $DISPLAY environment variable
```

**Solution:**
```bash
# If running headless, use Xvfb
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99

# Or run in virtual display
xvfb-run python3 launch_gui.py
```

#### Issue 4: Browser Driver Issues

**Symptom:**
```
WebDriverException: Message: 'chromedriver' executable needs to be in PATH
```

**Solution:**
```bash
# Install chromedriver
sudo apt-get install chromium-chromedriver

# Or use undetected-chromedriver (already in requirements)
# It auto-downloads the correct driver
```

#### Issue 5: Permission Issues with Screenshots

**Symptom:**
```
PermissionError: [Errno 13] Permission denied
```

**Solution:**
```bash
# Ensure directories are writable
chmod -R u+w activity_logs/ browser_data/ chrome_user_data/

# Check gnome-screenshot permissions
which gnome-screenshot
```

#### Issue 6: Memory Errors

**Symptom:**
```
MemoryError: Unable to allocate array
```

**Solution:**
```bash
# Reduce batch sizes in ML models
# Edit ml_system.py, reduce batch_size parameter

# Or increase system RAM/swap
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### Issue 7: Port Already in Use

**Symptom:**
```
OSError: [Errno 98] Address already in use
```

**Solution:**
```bash
# Find process using port
sudo lsof -i :5000

# Kill process
sudo kill -9 <PID>

# Or change port in web_gui.py
# app.run(port=5001)
```

#### Issue 8: GitHub API Rate Limits

**Symptom:**
```
GitHub API rate limit exceeded
```

**Solution:**
```bash
# Use authenticated requests with token in .env
# GITHUB_TOKEN=your_token_here

# Check rate limit status
curl -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/rate_limit
```

---

## 🎯 Functional Verification Checklist

Use this checklist to verify all systems are functioning:

### Core Systems
- [ ] Deep Tree Echo core loads successfully
- [ ] TreeNode creation works
- [ ] Echo propagation functions
- [ ] Pattern analysis completes
- [ ] ML predictions work

### Cognitive Systems
- [ ] Cognitive architecture initializes
- [ ] Goal generation works
- [ ] Personality system functions
- [ ] Emotional dynamics operate
- [ ] DET emotions calculate correctly

### Memory Systems
- [ ] Hypergraph memory creates concepts
- [ ] Relationships form correctly
- [ ] Pattern matching works
- [ ] Memory retrieval functions
- [ ] Centrality analysis completes

### Sensory-Motor Systems
- [ ] Screen capture works
- [ ] Mouse control functions
- [ ] Keyboard input works
- [ ] Object detection operates
- [ ] Spatial context updates

### Browser Automation
- [ ] Selenium initializes
- [ ] Browser launches
- [ ] Navigation works
- [ ] Element detection functions
- [ ] Screenshot capture works

### Dashboards
- [ ] GUI dashboard launches
- [ ] All tabs display correctly
- [ ] Real-time updates work
- [ ] Web dashboard serves pages
- [ ] API endpoints respond

### Self-Improvement
- [ ] Cronbot can run
- [ ] Echopilot initializes
- [ ] GitHub API connects
- [ ] Workflow files are valid

### Activity Management
- [ ] Activity stream records events
- [ ] Activity logs save correctly
- [ ] Heartbeat monitoring works
- [ ] Emergency protocols function

---

## 🔒 Maintaining Preserved State

### To Keep System Frozen (Time Capsule Mode)

1. **Don't run self-improvement systems:**
   ```bash
   # Skip: python3 cronbot.py
   # Skip: python3 echopilot.py
   ```

2. **Don't auto-update dependencies:**
   ```bash
   # Use exact versions in requirements.txt
   pip install -r requirements.txt --no-deps
   ```

3. **Create snapshot backup:**
   ```bash
   # Backup entire directory
   tar -czf deep-tree-echo-snapshot-v2.tar.gz \
     --exclude='.git' \
     --exclude='__pycache__' \
     --exclude='.venv' \
     echosurface-evolving/
   ```

4. **Document any changes:**
   ```bash
   # If you modify anything, document it
   echo "Change description" >> MODIFICATIONS.log
   ```

### To Allow Evolution

1. **Enable self-improvement:**
   ```bash
   # Run cronbot periodically
   python3 cronbot.py
   ```

2. **Enable workflows:**
   ```bash
   # GitHub Actions will run automatically
   # Monitor: https://github.com/orgitcog/echosurface-evolving/actions
   ```

3. **Track changes:**
   ```bash
   # Commit evolution
   git add .
   git commit -m "Evolution from snapshot v2"
   git tag -a v2.1 -m "Post-snapshot evolution"
   ```

---

## 📊 Performance Baselines

Expected performance on recommended hardware:

| Operation | Expected Time |
|-----------|--------------|
| System startup | 2-5 seconds |
| Echo propagation (100 nodes) | < 100ms |
| ML prediction | 100-500ms |
| Pattern matching | 50-200ms |
| Browser automation action | 1-3 seconds |
| Dashboard load | 1-2 seconds |
| Hypergraph query | 10-100ms |

If performance significantly deviates, investigate:
- System resource constraints
- Network latency
- Disk I/O bottlenecks
- Process conflicts

---

## 🔄 Update Procedures (If Needed)

### Security Updates Only

```bash
# Update only security patches
pip install --upgrade \
  requests \
  pillow \
  cryptography

# Test after update
python3 verify_environment.py
```

### Python Version Update

```bash
# If Python 3.12 becomes unavailable
# Test with Python 3.13+

python3.13 -m venv .venv313
source .venv313/bin/activate
pip install -r requirements.txt

# Run tests
python3 test_deep_tree_echo.py
```

### Dependency Conflicts

```bash
# If dependencies conflict
# Create new environment
python3 -m venv .venv_fixed
source .venv_fixed/bin/activate

# Install with conflict resolution
pip install -r requirements.txt --use-deprecated=legacy-resolver
```

---

## 📝 Operational Notes

### Running in Production

**Not Recommended:** This is a research/development system, not production-ready.

**If You Must:**
1. Run in isolated environment (Docker/VM)
2. Monitor resource usage continuously
3. Implement rate limiting for APIs
4. Set up proper logging and alerting
5. Regular backups of state
6. Firewall web dashboard
7. Use HTTPS for web access
8. Rotate API keys regularly

### Running in Development

**Recommended approach:**
1. Use virtual environment
2. Run dashboards for monitoring
3. Enable verbose logging
4. Experiment freely
5. Commit interesting discoveries
6. Document modifications

### Running in Research

**Best practices:**
1. Document all experiments
2. Version control all changes
3. Keep research notes
4. Compare to baseline
5. Publish findings
6. Share insights with community

---

## 🛠️ Tools and Utilities

### Useful Commands

```bash
# Check system status
python3 -c "from adaptive_heartbeat import AdaptiveHeartbeat; h = AdaptiveHeartbeat(); print(h.get_status())"

# View recent activity
tail -f activity_logs/activity_stream.log

# Monitor resource usage
python3 monitor.py

# Test all components
for test in test_*.py; do
    echo "Running $test"
    python3 "$test"
done

# Export activity logs
python3 -c "from activity_stream import ActivityStream; a = ActivityStream(); a.export_logs('export.json')"

# Visualize echo evolution
python3 evolution_visualization.py
```

### Debugging Tools

```bash
# Enable debug logging
export DTE_LOG_LEVEL=DEBUG
python3 launch_deep_tree_echo.py

# Python debugger
python3 -m pdb deep_tree_echo.py

# Interactive shell
python3 -i deep_tree_echo.py

# Profile performance
python3 -m cProfile -o profile.stats deep_tree_echo.py
python3 -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumtime'); p.print_stats(20)"
```

---

## 🎓 Learning Path

### For New Users

1. **Start Here:**
   - Read SNAPSHOT_v2.md (overview)
   - Read EVOLUTION_TIMELINE.md (history)
   - Read FEATURES_2024-2025.md (capabilities)

2. **Then:**
   - Run dashboards to see system in action
   - Explore code starting with deep_tree_echo.py
   - Run tests to understand components

3. **Finally:**
   - Experiment with modifications
   - Read source code in detail
   - Contribute improvements

### For Researchers

1. **Understand Architecture:**
   - Study echo propagation algorithm
   - Analyze hypergraph memory structure
   - Examine emotion-cognition coupling
   - Review spatial awareness integration

2. **Reproduce Results:**
   - Run baseline tests
   - Measure performance
   - Compare to documented baselines
   - Verify all features

3. **Extend System:**
   - Add new capabilities
   - Improve existing features
   - Publish findings
   - Contribute back

---

## 🔐 Security Notes

### Sensitive Data

**Never commit to repository:**
- API keys (.env file)
- Passwords
- Session tokens
- Personal data
- Browser cookies with credentials

**Use .gitignore:**
```bash
# Already configured in repository
.env
*.session
*.cookie
chrome_user_data/
browser_data/
deep_tree_echo_profile/
```

### Safe Practices

1. **API Keys:**
   - Use environment variables
   - Rotate regularly
   - Use minimal permissions
   - Monitor usage

2. **Browser Automation:**
   - Use separate browser profiles
   - Clear sessions after use
   - Don't store credentials in code
   - Use headless mode when possible

3. **Self-Improvement:**
   - Review auto-generated code
   - Test in isolated environment
   - Validate before committing
   - Monitor GitHub Actions logs

---

## 📞 Support and Community

### Getting Help

1. **Documentation:**
   - Read all .md files in repository
   - Check inline code comments
   - Review test files for examples

2. **Issues:**
   - Check existing GitHub issues
   - Search for similar problems
   - Create detailed bug reports

3. **Community:**
   - GitHub Discussions
   - Research papers citing this work
   - AI research communities

### Contributing

If you improve the system:
1. Document changes
2. Add tests
3. Update relevant .md files
4. Submit pull request
5. Include rationale

---

## ✅ Final Verification

Before considering restoration complete:

```bash
# Run complete verification
python3 << 'EOF'
from deep_tree_echo import DeepTreeEcho
from ml_system import MLSystem
from memory_management import HypergraphMemory

print("Testing Deep Tree Echo Snapshot v2...")

# Test 1: Core functionality
echo = DeepTreeEcho()
root = echo.create_tree("Test root")
echo.propagate_echoes()
assert root.echo_value > 0, "Echo propagation failed"
print("✓ Echo propagation works")

# Test 2: ML system
ml = MLSystem()
print("✓ ML system initialized")

# Test 3: Memory system
memory = HypergraphMemory()
concept = memory.create_concept("Test")
assert concept is not None, "Concept creation failed"
print("✓ Memory system works")

print("\n✅ All core systems functional!")
print("Snapshot v2 successfully restored and verified.")
EOF
```

---

## 🎉 Success!

If you've reached this point and all verifications pass:

**Congratulations! Deep Tree Echo Snapshot v2 is successfully restored and operational.**

You now have a working instance of a sophisticated autonomous AI system as it existed in mid-2025, preserved as a time capsule of the 2024-2025 AI explosion.

Explore, learn, and enjoy!

---

*Documented by: Autonomous AI Copilot*  
*Date: January 2, 2026*  
*Purpose: Enable future restoration of Deep Tree Echo Snapshot v2*
