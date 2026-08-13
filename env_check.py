import sys
sys.path.insert(0, r'C:\Users\Sajan\Desktop\Project XYZ')
sys.path.insert(0, r'C:\Users\Sajan\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\Lib\site-packages')
import torch
print('PyTorch version:', torch.__version__)
print('Torch file:', torch.__file__)

from tokenizers import Tokenizer
print('Tokenizer OK')

from model import ModelConfig, SmallEnglishLLM
print('Model import OK')

import argparse
print('Argparse OK')