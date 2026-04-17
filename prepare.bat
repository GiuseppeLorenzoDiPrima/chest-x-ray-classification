@echo off
REM =============================================================================
REM prepare.bat — Setup automatico dell'ambiente (Windows)
REM =============================================================================

echo.
echo ============================================================
echo   Chest X-Ray Classification - Setup ambiente
echo ============================================================

REM Crea virtual environment
echo.
echo [1/4] Creazione virtual environment (.venv)...
python -m venv .venv
call .venv\Scripts\activate

REM Aggiorna pip
echo.
echo [2/4] Aggiornamento pip...
python -m pip install --upgrade pip

REM Installa dipendenze
echo.
echo [3/4] Installazione requirements...
pip install -r requirements.txt

REM Reinstalla PyTorch con supporto CUDA 12.8
echo.
echo [4/4] Reinstallazione PyTorch con supporto CUDA 12.8...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall

echo.
echo ============================================================
echo   Setup completato.
echo.
echo   Per attivare l'ambiente:
echo     .venv\Scripts\activate
echo.
echo   Per eseguire la pipeline:
echo     python main.py
echo ============================================================
echo.
