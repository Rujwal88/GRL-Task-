import sys
import importlib
import pkg_resources

# --- Configuration ---
REQUIRED_PYTHON = (3, 11)
REQUIRED_PACKAGES = {
    "numpy": "2.3.5",
    "scipy": "1.17.0",
    "pandas": "3.0.0",
    "numba": "0.63.1",
    "llvmlite": "0.46.0",
    "torch": "2.10.0",
    "torchaudio": "2.10.0",
    "transformers": "4.57.3",
    "tokenizers": "0.22.2",
    "fastapi": "0.129.0",
    "pydantic": "2.12.5",
    "starlette": "0.52.1",
    "gradio": "6.5.1",
    "huggingface_hub": "0.36.2",
    "onnxruntime": "1.24.1",
    "pip": "26.0.1",
}

CRITICAL_PAIRS = [
    ("numba", "llvmlite"),
    ("transformers", "tokenizers"),
    ("fastapi", "starlette"),
    ("torch", "torchaudio")
]

def check_python_version():
    """Validates Python version is exactly 3.11.x"""
    current = sys.version_info[:2]
    if current != REQUIRED_PYTHON:
        print(f"❌ FAIL: Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]} required, found {current[0]}.{current[1]}")
        sys.exit(1)
    print(f"✅ PASS: Python {sys.version.split()[0]} verified")

def check_package_versions():
    """Validates installed package versions against requirements"""
    failures = []
    installed_packages = {p.project_name.lower(): p.version for p in pkg_resources.working_set}
    
    print("\nPackage Version Verification:")
    for pkg, version in REQUIRED_PACKAGES.items():
        base_pkg = pkg.replace('_', '-').lower() # normalize
        
        # Exact match or normalized match checking
        installed_ver = None
        for k in installed_packages:
            if k.replace('_', '-') == base_pkg:
                installed_ver = installed_packages[k]
                break
        
        if not installed_ver:
            print(f"❌ MISSING: {pkg}")
            failures.append(pkg)
            continue
            
        if installed_ver != version:
            print(f"❌ MISMATCH: {pkg} required {version}, found {installed_ver}")
            failures.append(pkg)
        else:
            print(f"✅ PASS: {pkg} {version}")

    if failures:
        print(f"\n❌ CRITICAL: {len(failures)} version mismatches found.")
        sys.exit(1)

def check_imports():
    """Validates critical imports work (DLL/binary check)"""
    print("\nImport/Binary Compatibility Verification:")
    modules = [
        "numpy", 
        "torch", 
        "onnxruntime", 
        "pydantic", 
        "transformers", 
        "gradio",
        "numba"
    ]
    
    for mod in modules:
        try:
            importlib.import_module(mod)
            print(f"✅ PASS: Import {mod}")
        except ImportError as e:
            print(f"❌ FAIL: Import {mod} failed: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ FAIL: Import {mod} crashed: {e}")
            sys.exit(1)

    # Specific functionality checks
    try:
        import numpy
        ver = numpy.__version__
        if not ver.startswith("2."):
             print(f"❌ FAIL: NumPy 2.x required, loaded {ver}")
             sys.exit(1)
        print("✅ PASS: NumPy 2.x runtime verified")
    except Exception as e:
        print(f"❌ FAIL: NumPy check error: {e}")
        sys.exit(1)
        
    try:
         import torch
         print(f"✅ PASS: Torch {torch.__version__} (CUDA: {torch.version.cuda if torch.cuda.is_available() else 'cpu'})")
    except Exception as e:
         print(f"❌ FAIL: Torch check error: {e}")

if __name__ == "__main__":
    print("Starting Environment Verification...\n")
    check_python_version()
    check_package_versions()
    check_imports()
    print("\n✅✅✅ Environment Verification COMPLETE. Ready for Production. ✅✅✅")
