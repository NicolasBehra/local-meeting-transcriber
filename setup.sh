#!/usr/bin/env bash
# Usage: ./setup.sh [--advanced]
#   --advanced : installe aussi PyTorch (CUDA 12.8) + pyannote.audio pour --diarization advanced
set -e

# Dependances systeme
APT_PKGS=()
command -v ffmpeg >/dev/null || APT_PKGS+=(ffmpeg)
command -v pactl >/dev/null || APT_PKGS+=(pulseaudio-utils)
command -v curl >/dev/null || APT_PKGS+=(curl)
python3 -c "import ensurepip" 2>/dev/null || APT_PKGS+=(python3-venv)
if [ ${#APT_PKGS[@]} -gt 0 ]; then
    echo "Installation des paquets systeme: ${APT_PKGS[*]}"
    sudo apt-get update
    sudo apt-get install -y "${APT_PKGS[@]}"
fi

# Ollama + modele
OLLAMA_MODEL="${OLLAMA_MODEL:-ministral-3:3b}"
if ! command -v ollama >/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi
# Attendre que le serveur Ollama reponde (il peut mettre quelques secondes a demarrer)
for _ in $(seq 30); do
    ollama list >/dev/null 2>&1 && break
    sleep 1
done
if ! ollama list >/dev/null 2>&1; then
    echo "Serveur Ollama injoignable. Lancez 'sudo systemctl start ollama' (ou 'ollama serve') puis relancez ./setup.sh"
    exit 1
fi
ollama pull "$OLLAMA_MODEL"

VENV_DIR=".venv"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt
# faster-whisper depend de onnxruntime (CPU), qui ecrase les fichiers de onnxruntime-gpu
if pip show onnxruntime >/dev/null 2>&1; then
    pip uninstall -y onnxruntime
    pip install --force-reinstall --no-deps "onnxruntime-gpu>=1.24,<1.27"
fi
if [ "$1" = "--advanced" ]; then
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
    pip install "pyannote.audio>=4.0"
fi
echo ""
echo "Installation terminee. Pour lancer le script :"
echo "  source .venv/bin/activate"
echo "  python meeting.py"
