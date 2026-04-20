import os
import sys
import warnings

warnings.filterwarnings("ignore")

def setup_environment():
    try:
        import torch
        _lib = os.path.join(os.path.dirname(torch.__file__), 'lib')

        if hasattr(os, 'add_dll_directory') and os.path.exists(_lib):
            os.add_dll_directory(_lib)
            print(f"[INFO] DLL dir added: {_lib}")

    except Exception as e:
        print(f"[WARNING] DLL fix skipped: {e}")

    try:
        import SharedArray
    except ImportError:
        import types
        sys.modules['SharedArray'] = types.ModuleType('SharedArray')
        print("[INFO] SharedArray stubbed (Windows)")

def check_torch():
    import torch

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name} | VRAM: {p.total_memory/1e9:.1f} GB")