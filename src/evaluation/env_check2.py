import os, sys
sys.path.insert(0, r"C:\Users\Sajan\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none")
sys.path.insert(0, r"C:\Users\Sajan\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\Lib\site-packages")
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("cuda version", torch.version.cuda)
    print("vram GB", torch.cuda.get_device_properties(0).total_memory / 1e9)
