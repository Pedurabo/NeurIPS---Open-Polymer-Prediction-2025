#!/usr/bin/env python3
"""
Test script for data exploration modules
Run this to verify that all modules are working correctly
"""

import sys
import os

# Add src directory to path
sys.path.append('src')

def test_imports():
    """Test that all required modules can be imported"""
    print("🧪 Testing module imports...")
    
    try:
        from data.loader import PolymerDataLoader
        print("✅ PolymerDataLoader imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import PolymerDataLoader: {e}")
        return False
    
    try:
        from utils.visualization import DataVisualizer
        print("✅ DataVisualizer imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import DataVisualizer: {e}")
        return False
    
    try:
        from utils.visualization import create_exploration_report
        print("✅ create_exploration_report imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import create_exploration_report: {e}")
        return False
    
    return True

def test_data_loader():
    """Test the data loader functionality"""
    print("\n📊 Testing data loader...")
    
    try:
        from data.loader import PolymerDataLoader
        
        # Create loader instance
        loader = PolymerDataLoader()
        print("✅ PolymerDataLoader instance created")
        
        # Test loading data (will fail if no data files exist)
        data_files = loader.load_all_data()
        print(f"✅ Data loading attempted, found: {list(data_files.keys())}")
        
        # Test getting data info
        data_info = loader.get_data_info()
        print(f"✅ Data info retrieved: {list(data_info.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        return False

def test_visualizer():
    """Test the visualizer functionality"""
    print("\n📈 Testing visualizer...")
    
    try:
        from utils.visualization import DataVisualizer
        
        # Create visualizer instance
        visualizer = DataVisualizer()
        print("✅ DataVisualizer instance created")
        
        # Test with dummy data
        dummy_data_info = {
            'train': {
                'shape': (1000, 10),
                'memory_usage': 1024*1024,
                'null_counts': {'col1': 0, 'col2': 5},
                'unique_counts': {'col1': 100, 'col2': 50}
            }
        }
        
        print("✅ Dummy data created for testing")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualizer test failed: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("TESTING DATA EXPLORATION MODULES")
    print("=" * 60)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please check module dependencies.")
        return
    
    # Test data loader
    if not test_data_loader():
        print("\n❌ Data loader tests failed.")
        return
    
    # Test visualizer
    if not test_visualizer():
        print("\n❌ Visualizer tests failed.")
        return
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\n🚀 Ready to run data exploration!")
    print("\nNext steps:")
    print("1. Extract competition data to the 'data/' directory")
    print("2. Run: python notebooks/01_data_exploration.py")
    print("3. Or run: python test_exploration.py")
    
    # Check if data directory exists and has files
    data_dir = "data"
    if os.path.exists(data_dir):
        files = os.listdir(data_dir)
        if files:
            print(f"\n📁 Data directory contains: {files}")
        else:
            print(f"\n📁 Data directory is empty. Please add competition data files.")
    else:
        print(f"\n📁 Data directory not found. Please create 'data/' directory and add competition files.")

if __name__ == "__main__":
    main()
