#!/usr/bin/env python3
"""
Mario RL Training - Version Compatibility Checker
Run this script to verify all dependencies are correctly installed
"""

import sys
import pkg_resources

# Expected versions for compatibility
EXPECTED_VERSIONS = {
    'gym': '0.21.0',
    'gym-super-mario-bros': '7.4.0',
    'nes-py': '8.2.1',
    'stable-baselines3': '1.8.0',
    'torch': '2.0.1',
    'torchvision': '0.15.2',
    'opencv-python': '4.8.1.78',
    'matplotlib': '3.7.2',
    'numpy': '1.24.3',
    'tqdm': '4.65.0',
    'tensorboard': '2.13.0'
}

def check_python_version():
    """Check Python version compatibility"""
    python_version = sys.version_info
    print(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major == 3 and 8 <= python_version.minor <= 11:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python version should be 3.8-3.11")
        return False

def check_package_versions():
    """Check if all required packages are installed with correct versions"""
    print("\n📦 Checking package versions...")
    
    all_good = True
    installed_packages = {pkg.project_name: pkg.version for pkg in pkg_resources.working_set}
    
    for package, expected_version in EXPECTED_VERSIONS.items():
        if package in installed_packages:
            installed_version = installed_packages[package]
            if installed_version == expected_version:
                print(f"✅ {package}: {installed_version}")
            else:
                print(f"⚠️  {package}: {installed_version} (expected {expected_version})")
                all_good = False
        else:
            print(f"❌ {package}: NOT INSTALLED")
            all_good = False
    
    return all_good

def test_imports():
    """Test importing all critical modules"""
    print("\n🧪 Testing imports...")
    
    test_imports = [
        ('gym', None),
        ('gymnasium', None),
        ('gym_super_mario_bros', None),
        ('nes_py.wrappers', 'JoypadSpace'),
        ('stable_baselines3', 'PPO'),
        ('torch', None),
        ('cv2', None),
        ('matplotlib.pyplot', None),
        ('numpy', None),
        ('shimmy', None)
    ]
    
    all_imported = True
    
    for module_name, import_class in test_imports:
        try:
            if import_class is None:
                exec(f"import {module_name}")
            else:
                exec(f"from {module_name} import {import_class}")
            print(f"✅ Successfully imported {module_name}")
        except ImportError as e:
            print(f"❌ Failed to import {module_name}: {e}")
            all_imported = False
    
    return all_imported

def test_mario_environment():
    """Test creating a basic Mario environment"""
    print("\n🎮 Testing Mario environment creation...")
    
    try:
        import gymnasium as gym
        import gym_super_mario_bros
        from nes_py.wrappers import JoypadSpace
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
        from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
        
        # Create environment
        env = gym_super_mario_bros.make('SuperMarioBros-v0')
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        # Wrap with compatibility layer for gymnasium
        env = GymV21CompatibilityV0(env=env)
        
        # Test reset
        obs, info = env.reset()
        print(f"✅ Environment created successfully")
        print(f"📊 Observation shape: {obs.shape if hasattr(obs, 'shape') else 'Unknown'}")
        print(f"📊 Action space: {env.action_space}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Failed to create Mario environment: {e}")
        print("💡 Try the old gym version instead...")
        
        # Fallback to old gym
        try:
            import gym
            import gym_super_mario_bros
            from nes_py.wrappers import JoypadSpace
            from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
            
            env = gym_super_mario_bros.make('SuperMarioBros-v0')
            env = JoypadSpace(env, SIMPLE_MOVEMENT)
            obs = env.reset()
            print(f"✅ Environment created successfully (old gym API)")
            print(f"📊 Observation shape: {obs.shape if hasattr(obs, 'shape') else 'Unknown'}")
            env.close()
            return True
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")
            return False

def main():
    """Run all compatibility checks"""
    print("🍄 Mario RL Training - Compatibility Check 🍄")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Package Versions", check_package_versions),
        ("Import Tests", test_imports),
        ("Mario Environment", test_mario_environment)
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {check_name} failed with error: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📋 SUMMARY")
    print("=" * 50)
    
    if all(results):
        print("🎉 ALL CHECKS PASSED! Your environment is ready for Mario RL training!")
        print("\n🚀 You can now run: python mario.py")
    else:
        print("❌ Some checks failed. Please fix the issues above before training.")
        print("\n💡 To fix issues:")
        print("   1. Make sure you're in the correct conda environment")
        print("   2. Reinstall packages with: pip install -r requirements.txt")
        print("   3. Or run the setup script: bash setup_mario_env.sh")
    
    print(f"\n📊 Results: {sum(results)}/{len(results)} checks passed")

if __name__ == "__main__":
    main()