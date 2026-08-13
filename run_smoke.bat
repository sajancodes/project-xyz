@echo off
cd "C:\Users\Sajan\Desktop\Project XYZ"
call .venv\Scripts\activate
python src\training\train_semantic_v2d.py --smoke --steps 5 > smoke_output.txt 2>&1
echo.
echo === SMOKE TEST OUTPUT ===
type smoke_output.txt
pause