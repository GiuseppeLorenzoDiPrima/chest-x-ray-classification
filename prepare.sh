#!/usr/bin/env bash
# =============================================================================
# prepare.sh — Setup automatico dell'ambiente (Linux / macOS)
# =============================================================================
set -e

echo ""
echo "============================================================"
echo "  Chest X-Ray Classification — Setup ambiente"
echo "============================================================"

# Crea virtual environment
echo ""
echo "[1/4] Creazione virtual environment (.venv)..."
python3 -m venv .venv
source .venv/bin/activate

# Aggiorna pip
echo ""
echo "[2/4] Aggiornamento pip..."
pip install --upgrade pip

# Installa dipendenze
echo ""
echo "[3/4] Installazione requirements..."
pip install -r requirements.txt

# Reinstalla PyTorch con supporto CUDA 12.8
echo ""
echo "[4/4] Reinstallazione PyTorch con supporto CUDA 12.8..."
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128 \
    --force-reinstall

echo ""
echo "============================================================"
echo "  Setup completato."
echo ""
echo "  Per attivare l'ambiente:"
echo "    source .venv/bin/activate"
echo ""
echo "  Per eseguire la pipeline:"
echo "    python main.py"
echo "============================================================"
echo ""
