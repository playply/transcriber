from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

from google.colab import drive, userdata

LAUNCHER_BUILD = "bootstrap-v1"
REPO_URL = "https://github.com/playply/transcriber.git"
REPO_DIR = Path("/content/transcriber")
DRIVE_MOUNT = Path("/content/drive")
UI_VENV = Path("/content/transcriber-ui-venv-6.27.0")

print(f"Interview Transcriber launcher: {LAUNCHER_BUILD}", flush=True)

# 1) Require an NVIDIA GPU before doing heavy setup.
nvidia_smi = shutil.which("nvidia-smi")
if not nvidia_smi:
    raise RuntimeError(
        "NVIDIA GPU is not attached to this Colab runtime. "
        "Select Runtime → Change runtime type → T4 GPU, reconnect the runtime, then run START again."
    )
gpu_check = subprocess.run(
    [nvidia_smi],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
if gpu_check.returncode != 0:
    raise RuntimeError(
        "NVIDIA GPU check failed. Select Runtime → Change runtime type → T4 GPU, "
        "reconnect the runtime, then run START again."
    )
print("✓ GPU available", flush=True)

# 2) Mount Google Drive.
drive.mount(str(DRIVE_MOUNT))
print("✓ Google Drive mounted", flush=True)

# 3) Read the Hugging Face read-only token from Colab Secrets.
try:
    hf_token = userdata.get("HF_TOKEN")
except Exception as exc:
    raise RuntimeError(
        "HF_TOKEN is not available. Add a read-only HF_TOKEN in Colab Secrets "
        "and enable notebook access."
    ) from exc
if not hf_token:
    raise RuntimeError("HF_TOKEN is empty in Colab Secrets.")
print("✓ HF_TOKEN loaded from Colab Secrets", flush=True)

# 4) Get the latest application code.
if (REPO_DIR / ".git").exists():
    subprocess.run(
        ["git", "-C", str(REPO_DIR), "fetch", "--quiet", "origin", "main"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(REPO_DIR), "reset", "--hard", "origin/main"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
else:
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    subprocess.run(
        ["git", "clone", "--quiet", "--branch", "main", REPO_URL, str(REPO_DIR)],
        check=True,
    )

repo_head = subprocess.run(
    ["git", "-C", str(REPO_DIR), "rev-parse", "--short", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
print(f"✓ Repository ready ({repo_head})", flush=True)

# 5) Install only the tested transcription stack in the main Colab Python.
print("Installing transcription dependencies...", flush=True)
core_install = subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        str(REPO_DIR / "requirements.txt"),
        "python-docx",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
if core_install.returncode != 0:
    print(core_install.stdout)
    raise RuntimeError(
        "Transcription dependency installation failed. The complete pip output is shown above."
    )
print("✓ Transcription dependencies installed", flush=True)

# 6) Keep Gradio isolated from WhisperX/pyannote dependencies.
if not (UI_VENV / "bin" / "python").exists():
    print("Creating isolated UI environment...", flush=True)
    subprocess.run([sys.executable, "-m", "venv", str(UI_VENV)], check=True)

ui_python = UI_VENV / "bin" / "python"
print("Installing UI dependencies...", flush=True)
ui_install = subprocess.run(
    [str(ui_python), "-m", "pip", "install", "gradio==6.27.0"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
if ui_install.returncode != 0:
    print(ui_install.stdout)
    raise RuntimeError(
        "Gradio installation failed. The complete pip output is shown above."
    )

try:
    ui_check = subprocess.run(
        [str(ui_python), "-c", "import gradio; print(gradio.__version__)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=90,
    )
except subprocess.TimeoutExpired as exc:
    raise RuntimeError(
        "Gradio import timed out in the isolated UI environment. "
        "Restart the Colab runtime and run START again."
    ) from exc

if ui_check.returncode != 0:
    print(ui_check.stdout)
    raise RuntimeError("Gradio import check failed in the isolated UI environment.")
print(f"✓ UI environment ready (Gradio {ui_check.stdout.strip()})", flush=True)

# 7) Launch the temporary authenticated Gradio UI.
env = os.environ.copy()
env["HF_TOKEN"] = hf_token
env["TRANSCRIBER_DRIVE_ROOT"] = "/content/drive/MyDrive"
env["TRANSCRIBER_UI_PASSWORD"] = secrets.token_urlsafe(10)
env["TRANSCRIBER_APP_PYTHON"] = sys.executable
env["PYTHONUNBUFFERED"] = "1"

print("\nStarting Interview Transcriber...", flush=True)
print("Keep this Colab cell running while you use the web UI.", flush=True)
subprocess.run(
    [str(ui_python), "-u", str(REPO_DIR / "ui.py")],
    env=env,
    check=True,
)
