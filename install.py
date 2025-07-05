#!/usr/bin/env python3
"""
Installation script for Flux Kontext Diff Merge ComfyUI node
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def check_comfyui_structure():
    """Check if we're in the right directory structure"""
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    
    # Check if we're in custom_nodes directory
    if os.path.basename(parent_dir) == "custom_nodes":
        print("✅ Detected ComfyUI custom_nodes directory structure")
        return True
    elif os.path.basename(grandparent_dir) == "ComfyUI":
        print("✅ Detected ComfyUI directory structure")
        return True
    else:
        print("⚠️  Warning: This doesn't appear to be in a ComfyUI custom_nodes directory")
        print("   Make sure you've placed this in: ComfyUI/custom_nodes/flux-kontext-diff-merge/")
        return False

def main():
    print("🚀 Flux Kontext Diff Merge Installation")
    print("=" * 40)
    
    # Check directory structure
    check_comfyui_structure()
    
    # Install requirements
    if install_requirements():
        print("\n🎉 Installation completed successfully!")
        print("\n📝 Next steps:")
        print("1. Restart ComfyUI")
        print("2. Look for 'Flux Kontext Diff Merge' in the image/postprocessing category")
        print("3. Check the README.md for usage instructions")
    else:
        print("\n❌ Installation failed. Please install requirements manually:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main() 