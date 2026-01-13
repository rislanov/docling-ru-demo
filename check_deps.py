#!/usr/bin/env python3
"""
Script to check dependency installation.
"""

import sys

def check_dependencies():
    """Checks installation of all required dependencies."""
    dependencies = {
        'torch': 'PyTorch',
        'docling': 'Docling',
    }
    
    all_installed = True
    print("Checking dependencies...\n")
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name} is installed")
        except ImportError:
            print(f"✗ {name} is not installed")
            all_installed = False
    
    print()
    
    if all_installed:
        print("✓ All dependencies are installed!")
        
        # Check MPS support for Apple Silicon
        try:
            import torch
            if torch.backends.mps.is_available():
                print("✓ Apple Silicon GPU (MPS) support is available")
            elif torch.cuda.is_available():
                print("✓ NVIDIA GPU (CUDA) support is available")
            else:
                print("✓ CPU will be used")
        except Exception as e:
            print(f"Warning: unable to check GPU availability: {e}")
        
        return 0
    else:
        print("✗ Some dependencies are not installed.")
        print("\nInstall them using:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == '__main__':
    sys.exit(check_dependencies())
