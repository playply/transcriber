from __future__ import annotations

import hashlib
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from google.colab import drive, output, userdata
from google.colab.output import eval_js

LAUNCHER_BUILD = "bootstrap-v12"
REPO_URL = "https://github.com/playply/transcriber.git"
REPO_DIR = Path("/content/transcriber")
DRIVE_MOUNT = Path("/content/drive")
UI_VENV = Path("/content/transcriber-ui-venv-6.27.0")
CORE_VERSIONS = {
    "whisperx": "3.8.5",
    "transformers": "4.57.6",
    "python-docx": "1.2.0",
}
GRADIO_VERSION = "6.27.0"


def _installed_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _versions_match(expected: dict[str, str]) -> bool:
    return all(_installed_version(name) == wanted for name, wanted in expected.items())


def _run_streamed(command: list[str], *, env: dict[str, str] | None = None) -> None:
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


print(f"Interview Transcriber launcher: {LAUNCHER_BUILD}", flush=True)

# 1) Detect GPU. Lack of GPU must not block maintenance work on existing transcripts.
nvidia_smi = shutil.which("nvidia-smi")
gpu_available = False
if nvidia_smi:
    gpu_check = subprocess.run(
        [nvidia_smi],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    gpu_available = gpu_check.returncode == 0

if gpu_available:
    print("✓ GPU available", flush=True)
else:
    print(
        "⚠ GPU unavailable — starting CPU maintenance mode. "
        "Existing transcript/registry operations remain available; new transcription requires a GPU.",
        flush=True,
    )

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

# 5) Install the tested transcription stack only when it is actually missing/mismatched.
if _versions_match(CORE_VERSIONS):
    print("✓ Transcription dependencies already satisfied", flush=True)
else:
    current = {name: _installed_version(name) for name in CORE_VERSIONS}
    print(f"Installing transcription dependencies (current: {current})...", flush=True)
    print("pip progress:", flush=True)
    try:
        _run_streamed(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "-r",
                str(REPO_DIR / "requirements.txt"),
                "python-docx==1.2.0",
            ]
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Transcription dependency installation failed. See the pip output above."
        ) from exc

    if not _versions_match(CORE_VERSIONS):
        actual = {name: _installed_version(name) for name in CORE_VERSIONS}
        raise RuntimeError(
            f"Transcription dependency versions are still incorrect after installation: {actual}"
        )
    print("✓ Transcription dependencies installed", flush=True)

# 6) Keep Gradio isolated from WhisperX/pyannote dependencies.
ui_python = UI_VENV / "bin" / "python"
ui_env_ok = False
if ui_python.exists():
    pip_check = subprocess.run(
        [str(ui_python), "-m", "pip", "--version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    ui_env_ok = pip_check.returncode == 0

if not ui_env_ok:
    if UI_VENV.exists():
        shutil.rmtree(UI_VENV)

    print("Creating isolated UI environment...", flush=True)
    try:
        _run_streamed(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "virtualenv"]
        )
        _run_streamed([sys.executable, "-m", "virtualenv", str(UI_VENV)])
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Could not create the isolated UI environment. See the output above."
        ) from exc

    ui_python = UI_VENV / "bin" / "python"
    pip_check = subprocess.run(
        [str(ui_python), "-m", "pip", "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if pip_check.returncode != 0:
        print(pip_check.stdout)
        raise RuntimeError("The isolated UI environment was created without a working pip.")

ui_gradio_version = subprocess.run(
    [
        str(ui_python),
        "-c",
        "from importlib.metadata import version; print(version('gradio'))",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    text=True,
).stdout.strip()

if ui_gradio_version == GRADIO_VERSION:
    print(f"✓ UI dependencies already satisfied (Gradio {GRADIO_VERSION})", flush=True)
else:
    print(
        f"Installing UI dependencies (current Gradio: {ui_gradio_version or 'not installed'})...",
        flush=True,
    )
    print("pip progress:", flush=True)
    try:
        _run_streamed(
            [
                str(ui_python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                f"gradio=={GRADIO_VERSION}",
            ]
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Gradio installation failed. See the pip output above.") from exc

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
if ui_check.stdout.strip() != GRADIO_VERSION:
    raise RuntimeError(
        f"Unexpected Gradio version in UI environment: {ui_check.stdout.strip()}"
    )
print(f"✓ UI environment ready (Gradio {ui_check.stdout.strip()})", flush=True)

# 7) Launch the temporary authenticated Gradio UI and explicitly stream child output.
# Stop any stale UI server left behind by an interrupted launcher run.
subprocess.run(
    ["pkill", "-f", str(REPO_DIR / "ui.py")],
    check=False,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

ui_port = None
for candidate_port in range(7860, 7871):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", candidate_port))
        except OSError:
            continue
        ui_port = candidate_port
        break
if ui_port is None:
    raise RuntimeError("No free local port available for the Gradio UI (7860-7870).")

# Colab's authenticated port proxy is currently unreliable for this Gradio app,
# so use a temporary Cloudflare Quick Tunnel. It exists only for this runtime.
CLOUDFLARED_VERSION = "2026.9.1"
CLOUDFLARED_SHA256 = "03f1f25d1cc93b9ad6c60569d44060bc4f17ed97075760ed8cfca4b12dcd68cc"
cloudflared_path = Path(f"/content/cloudflared-{CLOUDFLARED_VERSION}")
if not cloudflared_path.exists():
    print("Preparing temporary UI tunnel...", flush=True)
    download_url = (
        "https://github.com/cloudflare/cloudflared/releases/download/"
        f"{CLOUDFLARED_VERSION}/cloudflared-linux-amd64"
    )
    urllib.request.urlretrieve(download_url, cloudflared_path)
    digest = hashlib.sha256(cloudflared_path.read_bytes()).hexdigest()
    if digest != CLOUDFLARED_SHA256:
        cloudflared_path.unlink(missing_ok=True)
        raise RuntimeError("cloudflared checksum verification failed.")
    cloudflared_path.chmod(0o755)

subprocess.run(
    ["pkill", "-f", "cloudflared.*tunnel.*--url"],
    check=False,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

env = os.environ.copy()
env["HF_TOKEN"] = hf_token
env["TRANSCRIBER_DRIVE_ROOT"] = "/content/drive/MyDrive"
env["TRANSCRIBER_UI_PASSWORD"] = secrets.token_urlsafe(10)
env["TRANSCRIBER_APP_PYTHON"] = sys.executable
env["TRANSCRIBER_GPU_AVAILABLE"] = "1" if gpu_available else "0"
env["TRANSCRIBER_REPO_HEAD"] = repo_head
env["TRANSCRIBER_COLAB_EMBEDDED"] = "0"
env["TRANSCRIBER_EXTERNAL_TUNNEL"] = "1"
env["TRANSCRIBER_GRADIO_SHARE"] = "1"
env["TRANSCRIBER_UI_PORT"] = str(ui_port)
env["PYTHONUNBUFFERED"] = "1"

mode = "GPU transcription mode" if gpu_available else "CPU maintenance mode"
print(f"\nStarting Interview Transcriber ({mode})...", flush=True)
print("Keep this Colab cell running while you use the web UI.", flush=True)
print("Streaming UI startup log below:", flush=True)

ui_process = subprocess.Popen(
    [str(ui_python), "-u", str(REPO_DIR / "ui.py")],
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

assert ui_process.stdout is not None
tunnel_process = None
tunnel_log_handle = None
for line in ui_process.stdout:
    print(line, end="", flush=True)
    if tunnel_process is None and "Running on local URL:" in line:
        try:
            colab_proxy_url = eval_js(f"google.colab.kernel.proxyPort({ui_port})")
            print(f"\nOPTION C — COLAB PROXY (experimental): {colab_proxy_url}", flush=True)
        except Exception as exc:
            print(f"\nColab proxy URL unavailable: {exc}", flush=True)
        print("Starting Cloudflare fallback tunnel in parallel...", flush=True)
        tunnel_log_path = Path("/content/interview-transcriber-cloudflared.log")
        tunnel_log_handle = tunnel_log_path.open("w", encoding="utf-8")
        tunnel_process = subprocess.Popen(
            [
                str(cloudflared_path),
                "tunnel",
                "--url",
                f"http://127.0.0.1:{ui_port}",
                "--no-autoupdate",
            ],
            stdout=tunnel_log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

        tunnel_url = None
        deadline = time.time() + 30
        url_pattern = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
        while time.time() < deadline:
            if tunnel_process.poll() is not None:
                break
            if tunnel_log_path.exists():
                tunnel_text = tunnel_log_path.read_text(
                    encoding="utf-8", errors="replace"
                )
                match = url_pattern.search(tunnel_text)
                if match:
                    tunnel_url = match.group(0)
                    break
            time.sleep(0.5)

        if tunnel_url is None:
            if tunnel_process.poll() is None:
                tunnel_process.terminate()
            print(
                "\n⚠ Cloudflare fallback tunnel did not start. "
                "The Gradio UI is still running; use the Colab proxy URL above "
                "or the Gradio public URL if it appears.",
                flush=True,
            )
            print(
                "Cloudflare diagnostics: /content/interview-transcriber-cloudflared.log",
                flush=True,
            )
        else:
            print(f"\nOPTION B — CLOUDFLARE: {tunnel_url}", flush=True)

        print(
            "OPTION A — GRADIO SHARE will appear above as 'Running on public URL' "
            "if the Gradio tunnel succeeds.",
            flush=True,
        )
        print(
            "All external options use the username/password printed above. "
            "The temporary URLs disappear with the Colab runtime.",
            flush=True,
        )

ui_return_code = ui_process.wait()

if tunnel_process is not None and tunnel_process.poll() is None:
    tunnel_process.terminate()
if tunnel_log_handle is not None:
    tunnel_log_handle.close()

if ui_return_code != 0:
    raise RuntimeError(f"Gradio UI process exited with status {ui_return_code}.")
