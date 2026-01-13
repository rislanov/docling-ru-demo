#!/usr/bin/env python3
"""
Скрипт для проверки установки зависимостей.
"""

import sys

def check_dependencies():
    """Проверяет установку всех необходимых зависимостей."""
    dependencies = {
        'torch': 'PyTorch',
        'docling': 'Docling',
    }
    
    all_installed = True
    print("Проверка зависимостей...\n")
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name} установлен")
        except ImportError:
            print(f"✗ {name} не установлен")
            all_installed = False
    
    print()
    
    if all_installed:
        print("✓ Все зависимости установлены!")
        
        # Проверка поддержки MPS для Apple Silicon
        try:
            import torch
            if torch.backends.mps.is_available():
                print("✓ Поддержка Apple Silicon GPU (MPS) доступна")
            elif torch.cuda.is_available():
                print("✓ Поддержка NVIDIA GPU (CUDA) доступна")
            else:
                print("✓ Будет использоваться CPU")
        except Exception as e:
            print(f"Предупреждение: не удалось проверить доступность GPU: {e}")
        
        return 0
    else:
        print("✗ Некоторые зависимости не установлены.")
        print("\nУстановите их с помощью:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == '__main__':
    sys.exit(check_dependencies())
