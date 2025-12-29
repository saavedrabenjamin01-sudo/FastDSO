@echo off
title Portal Predictivo Distribucion - Flask
REM Pon este .bat en la MISMA carpeta que app.py
cd /d "%~dp0"

echo ===========================================================
echo  Iniciando entorno virtual y servidor Flask
echo ===========================================================
echo.

REM ---- Activar entorno virtual: probar local y luego un nivel arriba ----
if exist ".venv\Scripts\activate" (
    call ".venv\Scripts\activate"
) else if exist "..\.venv\Scripts\activate" (
    call "..\.venv\Scripts\activate"
) else (
    echo [ERROR] No se encontro el entorno virtual (.venv).
    echo Crea el venv y/o instala dependencias:
    echo    python -m venv .venv
    echo    .venv\Scripts\activate
    echo    pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM ---- Abrir el navegador ----
start "" "http://127.0.0.1:5000"

REM ---- Iniciar la aplicacion ----
python app.py

echo.
echo ===========================================================
echo  Servidor detenido. Presiona una tecla para cerrar.
echo ===========================================================
pause >nul
