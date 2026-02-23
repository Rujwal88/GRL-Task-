import sys
import platform
import importlib

def check_package(name, expected_version=None):
    try:
        mod = importlib.import_module(name)
        print(f"Imported {name}")
        if expected_version:
            import warnings
            warnings.filterwarnings("ignore")
            version = getattr(mod, "__version__", None)
            if version:
                print(f"{name} version: {version}")
    except ImportError as e:
        print(f"Failed to import {name}: {e}")
    except Exception as e:
        print(f"Error checking {name}: {e}")

print("Validating environment (Non-Torch packages)...")
print(f"Python: {platform.python_version()}")

# Core imports excluding torch/torchaudio for this check
packages = [
    "numpy", "scipy", "numba", "librosa", 
    "fastapi", "transformers", "accelerate", "gradio"
]

for p in packages:
    expected = None
    check_package(p, expected)

print("\nChecking Torch separately...")
try:
    import torch
    print(f"Torch version: {torch.__version__}")
except Exception as e:
    print(f"Torch failed: {e}")

print("Validation check complete.")
