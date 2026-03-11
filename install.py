#!/usr/bin/env python3
"""Installation script for Flux Kontext Diff Merge ComfyUI node."""

from pathlib import Path
import os
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"


def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)]
        )
        print("Requirements installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False


def check_comfyui_structure():
    """Check whether the node is installed inside a ComfyUI layout."""
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    grandparent_dir = os.path.dirname(parent_dir)

    if os.path.basename(parent_dir) == "custom_nodes":
        print("Detected ComfyUI custom_nodes directory structure.")
        return True
    if os.path.basename(grandparent_dir) == "ComfyUI":
        print("Detected ComfyUI directory structure.")
        return True

    print("Warning: this does not appear to be in a ComfyUI custom_nodes directory.")
    print("Place it in: ComfyUI/custom_nodes/flux-kontext-diff-merge/")
    return False


def main():
    print("Flux Kontext Diff Merge Installation")
    print("=" * 40)

    check_comfyui_structure()

    if install_requirements():
        print("\nInstallation completed successfully.")
        print("\nNext steps:")
        print("1. Restart ComfyUI")
        print("2. Look for 'Flux Kontext Diff Merge' in the image/postprocessing category")
        print("3. Check the README.md for usage instructions")
    else:
        print("\nInstallation failed. Please install requirements manually:")
        print(f"   {sys.executable} -m pip install -r {REQUIREMENTS_FILE}")


if __name__ == "__main__":
    main() 