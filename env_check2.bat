@echo off
cd "C:\Users\Sajan\Desktop\Project XYZ"
call .venv\Scripts\activate
python -c "import torch; print(torch.__version__)"
python -c "import sys; print(sys.path[:3])"
pause