@echo off
cd /d %~dp0

start "FastAPI" cmd /k ".venv\Scripts\uvicorn.exe src.api.main:app --host 127.0.0.1 --port 8000 --reload"

timeout /t 2 /nobreak >nul

start "Streamlit" cmd /k ".venv\Scripts\streamlit.exe run src/web/app.py"



