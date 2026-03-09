#!/usr/bin/env bash
set -e
VENV_DIR=".venv"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
echo ""
echo "Installation terminee. Pour lancer le script :"
echo "  source .venv/bin/activate"
echo "  python meeting.py"
