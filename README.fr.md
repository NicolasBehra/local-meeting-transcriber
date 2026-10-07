# Meeting Recorder — Option A (100% Local)

Enregistrement de reunions avec transcription en temps reel et analyse IA, entierement en local.

## Que fait ce programme ?

Ce script capture simultanement le son de votre microphone et l'audio systeme (ce que vous entendez dans vos ecouteurs/haut-parleurs), les mixe en un seul flux, transcrit la parole en texte en temps reel sur GPU (NVIDIA Parakeet TDT 0.6B v3 par defaut, Whisper en alternative), et genere periodiquement des analyses de la reunion via un LLM local (Ministral 3 via Ollama).

A la fin de la reunion (Ctrl+C), il produit un compte-rendu complet au format Markdown contenant :
- Un resume final genere par l'IA
- La transcription integrale horodatee
- Toutes les analyses intermediaires

**Aucune donnee ne quitte votre machine.** Tout tourne en local : transcription (Parakeet via onnx-asr, ou faster-whisper) et analyse (Ollama/Ministral 3).

## Pre-requis systeme

| Composant | Requis |
|---|---|
| OS | Linux avec PipeWire ou PulseAudio |
| GPU | NVIDIA avec CUDA (RTX 3060 6GB minimum recommande) |
| Python | 3.12+ (avec `python3-venv`) |
| FFmpeg | Installe par `setup.sh` |
| Ollama | Installe par `setup.sh` (le serveur doit tourner) |
| pactl | Installe par `setup.sh` (paquet `pulseaudio-utils`) |

## Installation

```bash
cd option_a_local
chmod +x setup.sh
./setup.sh              # installation standard (modes off / simple)
./setup.sh --advanced   # + PyTorch CUDA 12.8 et pyannote.audio (mode advanced)
```

`setup.sh` s'occupe de tout (Debian/Ubuntu, `sudo` est demande si necessaire) :

1. **Paquets systeme** : installe ceux qui manquent parmi `ffmpeg`, `pulseaudio-utils` (`pactl`), `curl`, `python3-venv`
2. **Ollama** : l'installe via le script officiel s'il est absent, puis telecharge le modele (`ministral-3:3b` par defaut ; autre modele avec `OLLAMA_MODEL=qwen3.5:4b ./setup.sh`)
3. **Python** : cree `.venv` et installe `requirements.txt` : `onnx-asr` + `onnxruntime-gpu` (Parakeet), `faster-whisper`, et les librairies CUDA 12 en wheels pip (`nvidia-cuda-runtime-cu12`, `nvidia-cublas-cu12`, `nvidia-curand-cu12`, `nvidia-cufft-cu12`, `nvidia-cudnn-cu12`). Il supprime ensuite le `onnxruntime` CPU tire par faster-whisper (il ecraserait `onnxruntime-gpu`) et reinstalle `onnxruntime-gpu<1.27` (1.27+ exige CUDA 13)
4. **`--advanced` uniquement** : installe `torch` + `torchaudio` (index CUDA 12.8, ~3 Go) et `pyannote.audio>=4.0`

Le script est idempotent : les composants deja installes sont ignores.

## Utilisation

### Lancement basique

```bash
source .venv/bin/activate
python meeting.py
```

Le script :
1. Detecte automatiquement vos sources audio (micro et sortie par defaut du systeme)
2. Charge le modele de transcription sur le GPU — Parakeet par defaut, ou Whisper avec `--asr whisper` (avec fallback CPU si CUDA n'est pas disponible)
3. Demarre l'enregistrement et la transcription
4. Affiche un tableau de bord en temps reel dans le terminal

### Moteur de transcription

L'option `--asr` choisit le moteur de reconnaissance vocale :

```bash
python meeting.py                 # Parakeet TDT 0.6B v3 (defaut)
python meeting.py --asr whisper   # faster-whisper large-v3 (ancien moteur)
```

| Moteur | WER francais (FLEURS) | Taille | Notes |
|---|---|---|---|
| `parakeet` (defaut) | 4,97 % | 0,6B params | Beaucoup plus rapide, detection automatique de la langue (25 langues europeennes) |
| `whisper` | 7,15 % (large-v3) | 1,55B params | Chaine de secours large-v3 → medium → CPU |

### Langue de transcription

L'option `--language` permet de choisir la langue de transcription et d'analyse (defaut : francais) :

```bash
python meeting.py                    # Francais (defaut)
python meeting.py --language en      # Anglais
python meeting.py --language de      # Allemand
python meeting.py --language es      # Espagnol
```

Les prompts d'analyse IA s'adaptent automatiquement a la langue choisie. Avec `--asr whisper`, Whisper est aussi force dans cette langue ; Parakeet detecte la langue parlee automatiquement (fr, en, de, es et 21 autres langues europeennes).

### Traduction en temps reel

L'option `--translate` active la traduction en temps reel de la transcription vers une autre langue :

```bash
# Reunion en allemand, traduite en francais
python meeting.py --language de --translate fr

# Reunion en espagnol, traduite en anglais
python meeting.py --language es --translate en
```

La traduction utilise le meme LLM (Ollama, `ministral-3:3b` par defaut) que l'analyse. Toutes les 5 secondes, les nouvelles lignes de transcription sont envoyees en batch pour traduction. Le texte traduit apparait dans un panneau dedie a cote de la transcription originale. L'export Markdown inclut egalement une section "Traduction".

Note : `--translate` doit etre different de `--language` (traduire vers la meme langue est une erreur).

### Identification des locuteurs (diarisation)

L'option `--diarization` permet d'identifier qui parle dans la transcription :

```bash
# Pas d'identification (par defaut)
python meeting.py --diarization off

# Mode simple : "Moi" (micro) vs "Interlocuteur" (audio systeme)
python meeting.py --diarization simple

# Mode avance : identification IA via pyannote
python meeting.py --diarization advanced
```

**Mode simple** — Capture le micro et l'audio systeme separement (deux processus FFmpeg). Chaque ligne est prefixee `Moi:` ou `Interlocuteur:`. Aucune dependance supplementaire. Ideal pour les visioconferences ou il faut distinguer votre voix de celle des participants distants.

**Mode avance (pyannote)** — Utilise le modele IA `pyannote/speaker-diarization-community-1` (pyannote.audio 4.x) pour identifier chaque locuteur individuellement (SPEAKER_00, SPEAKER_01, etc.), meme quand plusieurs personnes parlent sur le meme canal audio. Configuration supplementaire requise :

1. **Creer un compte Hugging Face** sur https://huggingface.co/join, puis generer un token **Fine-grained** sur https://huggingface.co/settings/tokens avec au minimum la permission "Read access to contents of all public gated repos you can access"
2. **Accepter la licence du modele** (obligatoire, sinon le telechargement echoue) :
   - https://huggingface.co/pyannote/speaker-diarization-community-1 → cliquer "Agree and access repository"
3. **Installer les dependances** :
   ```bash
   ./setup.sh --advanced
   ```
   Cela installe `torch` et `torchaudio` (CUDA 12.8, ~3 Go) et `pyannote.audio>=4.0`.
4. **Definir le token** via un fichier `.env` (recommande) ou en variable d'environnement :
   ```bash
   # Option 1 : fichier .env (recommande — charge automatiquement)
   echo 'HF_TOKEN=hf_ABCDxxxxxxxx' > .env

   # Option 2 : variable d'environnement
   export HF_TOKEN="hf_ABCDxxxxxxxx"
   ```
   Le fichier `.env` est charge automatiquement au demarrage via `python-dotenv`. Pensez a ajouter `.env` dans votre `.gitignore` pour ne pas exposer votre token.

Le modele (~300 Mo) est telecharge au premier lancement, puis mis en cache localement.

### Arret de la reunion

- **1er Ctrl+C** : arret propre — stoppe l'enregistrement, termine la transcription des chunks restants, genere un resume final avec le LLM, exporte le fichier Markdown
- **2e Ctrl+C** : arret force — sauvegarde partielle immediate et quitte

### Fichier de sortie

Le rapport est sauvegarde dans :

```
option_a_local/outputs/meeting_YYYY-MM-DD_HHhMM.md
```

## Configuration

Les constantes sont definies en haut du fichier `meeting.py` (`OLLAMA_MODEL` et `OLLAMA_NUM_CTX` peuvent aussi etre definies en variable d'environnement ou dans le fichier `.env`) :

| Constante | Valeur par defaut | Description |
|---|---|---|
| `SAMPLE_RATE` | `16000` | Frequence d'echantillonnage audio (Hz) |
| `CHUNK_SECONDS` | `30` | Duree de chaque morceau audio envoye au moteur de transcription (secondes) |
| `OVERLAP_SECONDS` | `2` | Chevauchement entre les morceaux pour eviter de couper des mots |
| `ANALYSIS_INTERVAL` | `150` | Intervalle entre chaque analyse IA (secondes, ~2.5 min) |
| `PARAKEET_MODEL` | `"nemo-parakeet-tdt-0.6b-v3"` | Modele de transcription par defaut (onnx-asr) |
| `WHISPER_MODEL` | `"large-v3"` | Modele Whisper principal (~3 GB VRAM en int8_float16) |
| `WHISPER_FALLBACK` | `"medium"` | Modele de secours si large-v3 ne rentre pas en VRAM |
| `WHISPER_COMPUTE` | `"int8_float16"` | Type de quantification pour Whisper |
| `OLLAMA_MODEL` | `"ministral-3:3b"` | Modele LLM utilise via Ollama (env/`.env` ; ex : `qwen3.5:4b`, `gemma4`, `ministral-3:8b`) |
| `OLLAMA_NUM_CTX` | `16384` | Fenetre de contexte du LLM en tokens (env/`.env`). Sans elle, Ollama se limite a 4096 tokens sur un GPU 6 Go et tronque les longues reunions sans prevenir |
| `DIARIZATION_MODEL` | `"pyannote/speaker-diarization-community-1"` | Modele pyannote pour `--diarization advanced` |
| `TRANSLATION_INTERVAL` | `5` | Intervalle entre chaque batch de traduction (secondes) |
| `LANGUAGE_DEFAULT` | `"fr"` | Langue de transcription par defaut (modifiable via `--language`) |

Pour modifier un parametre, editez directement la constante dans le fichier. La langue et le moteur peuvent aussi etre changes au lancement avec `--language` et `--asr`.

`OLLAMA_MODEL` et `OLLAMA_NUM_CTX` se reglent sans toucher au code, dans le fichier `.env` (ou en variable d'environnement) :

```bash
# .env
HF_TOKEN=hf_xxx              # mode advanced uniquement
OLLAMA_MODEL=ministral-3:3b  # ex : mistral pour revenir a l'ancien modele (ollama pull mistral)
OLLAMA_NUM_CTX=16384         # 8192 si la VRAM est juste
```

## Interface terminal

L'affichage utilise Rich et se compose de quatre zones.

**Sans `--translate`** (par defaut) :

```
+----------------------------------+------------------+
|                                  |                  |
|         Transcription            |   Analyse IA     |
|    (20 dernieres lignes          |   (derniere      |
|     horodatees)                  |    analyse)      |
|                                  +------------------+
|                                  |                  |
|                                  |   Suggestions    |
|                                  |   (conseils      |
|                                  |    actionnables) |
+----------------------------------+------------------+
|  Duree: 0:05:23 | Chunks: 10 | Prochaine: 42s     |
+----------------------------------------------------+
```

**Avec `--translate`** :

```
+------------------+------------------+
|  Transcription   |   Traduction     |
|  (langue orig.)  |   (langue cible) |
+------------------+------------------+
|   Analyse IA     |   Suggestions    |
+------------------+------------------+
|  Duree: 0:05:23 | Chunks: 10      |
+------------------------------------+
```

- **Panneau Transcription** (vert) : les 20 dernieres lignes transcrites avec horodatage `[HH:MM:SS]`
- **Panneau Traduction** (jaune, uniquement avec `--translate`) : les 20 dernieres lignes traduites
- **Panneau Analyse IA** (bleu) : le contenu de la derniere analyse intermediaire ou finale
- **Panneau Suggestions** (magenta) : suggestions actionnables en temps reel — questions a poser, points a clarifier, alertes de consensus, rappels d'actions
- **Barre de statut** : duree ecoulee, nombre de chunks traites, compte a rebours avant la prochaine analyse

---

## Depannage

### Apres un deplacement du dossier du projet

Le virtual environment Python (`.venv`) contient des chemins absolus et **casse si le dossier est deplace**. Il faut le recreer :

```bash
python3 -m venv .venv --clear
./setup.sh              # ou ./setup.sh --advanced
```

### Erreur `libcublas.so.12 is not found`

Les paquets pip `nvidia-cuda-runtime-cu12`, `nvidia-cublas-cu12`, `nvidia-curand-cu12`, `nvidia-cufft-cu12` et `nvidia-cudnn-cu12` installent les libs CUDA 12 dans `.venv/lib/.../nvidia/*/lib/`, mais ni `ctranslate2` (Whisper) ni `onnxruntime-gpu` (Parakeet) ne les trouvent automatiquement. Le script les precharge au demarrage via `_preload_nvidia_libs()` (avant l'import de `faster_whisper` et `onnx_asr`).

Si l'erreur apparait, verifiez que ces paquets sont bien installes dans le venv :

```bash
.venv/bin/pip install -r requirements.txt
```

### Erreur `CUDAExecutionProvider` / `libcublasLt.so.13`

Si Parakeet affiche `Failed to create CUDAExecutionProvider ... Require cuDNN 9.* and CUDA 13.*` (ou `libcublasLt.so.13: cannot open shared object file`) puis tourne en CPU, la version de `onnxruntime-gpu` installee est trop recente : a partir de 1.27 elle exige CUDA 13, alors que le projet fournit CUDA 12 (partage avec `ctranslate2`). D'ou l'epinglage `onnxruntime-gpu>=1.24,<1.27` dans `requirements.txt`.

Autre cause possible : `faster-whisper` depend de `onnxruntime` (version CPU), qui ecrase les fichiers de `onnxruntime-gpu` s'il est installe apres. Dans les deux cas, la correction est :

```bash
./setup.sh
```

(il desinstalle `onnxruntime` CPU et reinstalle `onnxruntime-gpu<1.27`). Le message de chargement indique le peripherique reellement utilise : `Parakeet ... charge (cuda)` ou `(cpu)`.

### Partage VRAM entre transcription et Ollama (RTX 3060 6 Go)

Avec la configuration par defaut, Parakeet (~1-2 Go) et `ministral-3:3b` (4,2 Go avec un contexte de 16384, mesure via `ollama ps`) sont a la limite d'une carte 6 Go. Si tout ne tient pas, Ollama place une partie du modele en CPU (ca marche, mais l'analyse est plus lente). Pour tout garder sur le GPU, reduisez la fenetre de contexte (ex : `OLLAMA_NUM_CTX=8192` dans `.env`). Verifiez avec `ollama ps` (colonne PROCESSOR).

En revanche, Whisper (`large-v3`, `--asr whisper`) ou un LLM plus gros (ex : `ministral-3:8b`) ne tiennent pas avec l'autre modele. Si Ollama a deja charge un modele, Whisper ne pourra pas se charger sur le GPU et basculera en CPU (transcription ~3x plus lente).

**Solution dans ce cas** : forcer Ollama en CPU pour laisser toute la VRAM a la transcription.

```bash
sudo systemctl edit ollama
```

Ajouter :

```ini
[Service]
Environment="CUDA_VISIBLE_DEVICES="
```

Puis :

```bash
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

L'analyse IA sera un peu plus lente sur CPU, mais la transcription restera rapide sur GPU. C'est le meilleur compromis car la transcription est en temps reel (latence critique) alors que l'analyse ne tourne que toutes les 2.5 minutes.

Pour revenir en arriere (Ollama sur GPU) :

```bash
sudo systemctl revert ollama
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

---

## Comment les elements fonctionnent entre eux

### Architecture generale

Le script repose sur **3 threads de travail** coordonnes par un **thread principal** :

```
Micro ─┐                  ┌──────────────┐     ┌──────────────┐
       ├─► FFmpeg (amix) ─┤ AudioCapture ├────►│ Transcriber  │
Monitor┘    subprocess     │   Thread 1   │Queue│   Thread 2   │
                           └──────────────┘     └──────┬───────┘
                                                       │ transcript_log
                                                       ▼
                                                ┌──────────────┐
                                                │   Analyzer   │
                                                │   Thread 3   │
                                                └──────┬───────┘
                                                       │ analysis_log
                                                       ▼
                                                ┌──────────────┐
                                                │  Main Thread │
                                                │ Rich Display │
                                                │ + Signal     │
                                                │ + Export MD  │
                                                └──────────────┘
```

### Flux de donnees

1. **AudioCapture** lance un processus FFmpeg qui capture le micro et le monitor, les mixe, et envoie le flux PCM brut via un pipe. Le thread lit ce pipe seconde par seconde, accumule dans un buffer, et decoupe en chunks de 30 secondes avec 2 secondes de chevauchement. Chaque chunk est place dans une `Queue` thread-safe.

2. **Transcriber** consomme les chunks depuis la Queue. Pour chaque chunk, il appelle le moteur choisi (Parakeet ou faster-whisper) via `_segments()`, qui renvoie des segments de texte avec leurs temps de debut/fin. Chaque segment est horodate et ajoute a `transcript_log` (liste partagee, protegee par un `threading.Lock`).

3. **Analyzer** se reveille toutes les 2.5 minutes. Il lit les nouvelles lignes de `transcript_log` (depuis son dernier index), les envoie a Ollama (via `llm_chat()`) avec un prompt demandant une analyse structuree, et stocke le resultat dans `analysis_log`.

4. **Main thread** fait tourner un affichage Rich Live a 2 FPS qui lit `transcript_log` et `analysis_log` pour mettre a jour les panneaux. Il gere aussi le signal Ctrl+C pour orchestrer l'arret propre.

### Synchronisation

- **Queue** (taille max 50) : entre AudioCapture et Transcriber. Si Transcriber est trop lent, la queue se remplit et AudioCapture bloque (backpressure).
- **Lock** : protege `transcript_log` en ecriture (Transcriber) et en lecture (Analyzer, Main, export).
- **stop_event** : `threading.Event` partage par tous les threads. Quand il est active (Ctrl+C), chaque thread termine sa boucle proprement.

### Sequence d'arret

```
Ctrl+C ─► signal_handler
           ├── stop_event.set()
           ├── audio.stop() (termine FFmpeg)
           ▼
         Main loop sort
           ├── transcriber.join(30s) — drain la queue
           ├── analyzer.run_final() — resume sur toute la transcription
           └── export_markdown() — ecrit le fichier .md
```

---

## Comment fonctionne chaque outil en detail

### FFmpeg — Capture audio

FFmpeg est lance comme sous-processus avec la commande suivante :

```
ffmpeg -hide_banner -loglevel error \
  -f pulse -i <source_micro> \
  -f pulse -i <source_monitor> \
  -filter_complex "amix=inputs=2:duration=longest" \
  -ac 1 -ar 16000 \
  -f s16le -acodec pcm_s16le \
  pipe:1
```

- **`-f pulse`** : utilise le backend PulseAudio (compatible PipeWire via `pipewire-pulse`)
- **Deux entrees `-i`** : le micro physique et le "monitor" du sink (= ce qui sort des haut-parleurs). Cela permet de capturer a la fois votre voix et celles des participants distants
- **`amix=inputs=2:duration=longest`** : mixe les deux flux en un seul. `duration=longest` garde le flux actif tant qu'au moins une source produit du son
- **`-ac 1 -ar 16000`** : conversion en mono 16 kHz, le format attendu par les modeles de transcription
- **`-f s16le -acodec pcm_s16le`** : sortie en PCM brut, entiers signes 16 bits little-endian (pas d'en-tete WAV, pas de compression)
- **`pipe:1`** : ecrit sur stdout, que Python lit via `subprocess.PIPE`

**Detection des sources** : la fonction `detect_sources()` utilise `pactl get-default-source` et `pactl get-default-sink` pour trouver l'entree et la sortie audio par defaut du systeme. Le monitor est derive en ajoutant `.monitor` au nom du sink par defaut. Cela suit automatiquement les parametres audio — si vous passez des haut-parleurs a un casque Bluetooth, le script utilisera le monitor du casque sans configuration manuelle.

### Parakeet (onnx-asr) — Transcription (par defaut)

NVIDIA Parakeet TDT 0.6B v3 est un modele de reconnaissance vocale multilingue (25 langues europeennes) qui fait mieux que Whisper large-v3 en francais (4,97 % contre 7,15 % de WER sur FLEURS), tout en etant beaucoup plus rapide et ~4x plus petit. Il tourne via `onnx-asr` sur `onnxruntime-gpu` (CUDA 12), sans PyTorch ni NeMo.

**Chargement du modele** (`ensure_parakeet_model()`, thread principal) :
- Charge le modele et la VAD Silero sur CUDA, bascule sur CPU si besoin
- onnxruntime bascule silencieusement en CPU quand les libs CUDA ne se chargent pas : le script verifie le provider actif et affiche le peripherique reel
- L'arena memoire GPU ne grossit qu'a la demande (`kSameAsRequested`) pour laisser de la VRAM a Ollama
- Les libs CUDA 12 (wheels pip) sont prechargees par `_preload_nvidia_libs()` avant tout import, sans `LD_LIBRARY_PATH`

**Transcription** :
- La VAD Silero (`min_silence_duration_ms=500`) decoupe chaque chunk de 30 secondes en segments de parole et ignore le silence
- Chaque segment est renvoye avec son texte et ses temps de debut/fin (utilises pour l'alignement avec la diarisation)
- La langue est detectee automatiquement

### faster-whisper — Transcription (`--asr whisper`)

faster-whisper est une reimplementation de Whisper d'OpenAI utilisant CTranslate2 pour l'inference. C'est 4x plus rapide que l'implementation originale pour une qualite equivalente.

**Chargement du modele** (effectue dans le thread principal avant le demarrage des workers) :
- Tente d'abord `large-v3` sur CUDA en `int8_float16` (~3 GB VRAM). Ce mode quantifie les poids en int8 mais garde les calculs en float16, offrant le meilleur rapport qualite/vitesse
- Si echec, tente `medium` sur CUDA (~2 GB VRAM)
- Si CUDA n'est pas disponible, bascule sur `medium` en CPU (`int8`)
- Si tout echoue, quitte avec un message d'erreur clair

**Transcription** :
- Chaque chunk de 30 secondes (tableau numpy float32, normalise entre -1 et 1) est passe a `model.transcribe()`
- Parametres : `language` defini via l'option `--language` (defaut `"fr"`, pas de detection automatique), `beam_size=5` (recherche en faisceau pour une meilleure qualite)
- **Filtre VAD** (Silero Voice Activity Detection) active pour ignorer le silence et eviter les hallucinations (ex : Whisper qui genere "Merci" ou "Sous-titrage FR" sur de l'audio silencieux)
- La methode retourne un iterateur de segments, chacun avec un attribut `.text`
- Les segments vides sont filtres

**Chevauchement** : chaque chunk partage ses 2 dernieres secondes avec le debut du chunk suivant. Cela evite de couper un mot a la frontiere entre deux chunks, car Whisper peut ainsi "voir" le contexte.

### Ollama / Ministral 3 — Analyse IA

Ollama est un runtime pour executer des LLM en local. Le modele par defaut est `ministral-3:3b` (Mistral, decembre 2025), performant dans les langues europeennes, ~3 Go : il tient sur le GPU a cote de Parakeet. N'importe quel modele Ollama peut etre utilise via `OLLAMA_MODEL`.

**Fonctionnement** :
- Le client Python `ollama` communique avec le serveur Ollama qui doit tourner en arriere-plan
- Tous les appels passent par `llm_chat()`, qui fixe `num_ctx` (`OLLAMA_NUM_CTX`, defaut 16384), `keep_alive="30m"` (le modele reste charge entre deux analyses) et la temperature (0.2 pour l'analyse et la traduction, 0.5 pour les suggestions)
- Si un prompt depasse la fenetre de contexte, un avertissement s'affiche dans le panneau d'erreurs au lieu de tronquer la transcription sans prevenir
- A chaque analyse, le script envoie les nouvelles lignes de transcription (depuis la derniere analyse) avec un prompt structurant la reponse
- Le prompt demande : resume des points, decisions prises, actions a mener (avec responsable), questions ouvertes
- L'analyse finale reprend la transcription complete pour un resume global

**Analyses intermediaires** :
- Declenchees toutes les 150 secondes (~2.5 min)
- Traitent uniquement les nouvelles lignes depuis la derniere analyse (`last_analysis_index`)
- Permettent de suivre l'evolution de la reunion en temps reel

**Analyse finale** :
- Declenchee a l'arret (Ctrl+C), apres que la transcription soit terminee
- Reenvoie la totalite de la transcription pour un resume complet et coherent

### Rich — Affichage terminal

Rich est une bibliotheque Python pour creer des interfaces terminal riches.

**Composants utilises** :
- **`Live`** : met a jour l'affichage en place (sans scroll) a 2 rafraichissements par seconde
- **`Layout`** : organise l'ecran en grille. Ici : deux colonnes en haut (transcription 2/3, panneau droit 1/3 divise en analyse + suggestions), une barre en bas
- **`Panel`** : encadre chaque section avec un titre et une bordure coloree (vert pour la transcription, bleu pour l'analyse, magenta pour les suggestions)
- **`Text`** : texte style pour la barre de statut (blanc gras sur fond bleu fonce)

**Donnees affichees** :
- Les 20 dernieres lignes de transcription (pour ne pas surcharger l'ecran)
- La derniere analyse (intermediaire ou finale)
- Les dernieres suggestions (conseils actionnables pour le participant)
- Duree ecoulee, nombre de chunks traites, secondes restantes avant la prochaine analyse

### NumPy — Traitement du signal

NumPy est utilise pour manipuler efficacement les donnees audio brutes :

- **Lecture du pipe FFmpeg** : les octets bruts sont convertis en tableau `int16` via `np.frombuffer()`
- **Buffer d'accumulation** : les echantillons arrivent par blocs de 1 seconde et sont concatenes avec `np.concatenate()`
- **Decoupage en chunks** : quand le buffer atteint 480 000 echantillons (30s a 16 kHz), les premiers 480 000 sont extraits, et le buffer est raccourci en gardant les 32 000 derniers echantillons (2s d'overlap)
- **Normalisation** : conversion `int16 → float32` par division par 32768.0, produisant des valeurs entre -1.0 et 1.0, le format attendu par les modeles de transcription

### Export Markdown

A la fin de la reunion, le script genere un fichier Markdown structure :

```markdown
# Compte-rendu de reunion — 2026-03-06 14:30

## Resume final
[Contenu genere par le LLM sur la transcription complete]

## Transcription complete
[HH:MM:SS] Premiere phrase transcrite...
[HH:MM:SS] Deuxieme phrase transcrite...
...

## Analyses intermediaires
### Analyse a 14:32:30
[Contenu de l'analyse intermediaire]

### Analyse a 14:35:00
[Contenu de l'analyse intermediaire]
```

Le fichier est cree dans `outputs/` avec un nom base sur la date et l'heure de debut : `meeting_2026-03-06_14h30.md`.
