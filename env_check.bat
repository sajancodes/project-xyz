@echo off
cd "C:\Users\Sajan\Desktop\Project XYZ"
call .venv\Scripts\activate
python --version
python -c "import sys; print(sys.path[:3])"
python -c "import torch; print(torch.__version__)"
pause