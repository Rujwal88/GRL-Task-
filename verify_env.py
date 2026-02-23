import sys
import platform
import importlib

def check_package(name, expected_version=None):
    try:
        mod = importlib.import_module(name)
        print(f"Imported {name}")
        if expected_version:
            import warnings
            # Suppress warnings for clean output
            warnings.filterwarnings("ignore")
            
            # Check version attribute (some packages use different attributes)
            version = getattr(mod, "__version__", None)
            if version:
                print(f"{name} version: {version}")
                # Strict check for torch/torchaudio as requested
                base_version = version.split('+')[0]
                if expected_version and base_version != expected_version:
                    print(f"Mismatch! Expected {expected_version}, got {version}")
                    sys.exit(1)
            else:
                print(f"Could not find version for {name}")
    except ImportError as e:
        print(f"Failed to import {name}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error checking {name}: {e}")
        sys.exit(1)

print("Validating environment...")
print(f"Python: {platform.python_version()}")

if not (sys.version_info.major == 3 and sys.version_info.minor == 11):
    print("Python version failure!")
    sys.exit(1)

# Core imports
packages = [
    "numpy", "scipy", "torch", "numba", "librosa", 
    "fastapi", "transformers"
]

for p in packages:
    expected = None
    if p == "torch": expected = "2.10.0" 
    check_package(p, expected)

import torchaudio
print(f"Torchaudio version: {torchaudio.__version__}")
if torchaudio.__version__.split('+')[0] != "2.10.0":
    print("Torchaudio version mismatch!")
    sys.exit(1)

print("SUCCESS: Environment validated.")
