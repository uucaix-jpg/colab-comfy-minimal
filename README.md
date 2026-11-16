# colab-comfy-minimal

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uucaix-jpg/colab-comfy-minimal/blob/main/colab_comfyui_deploy.ipynb)

First-principles minimum-path integration pairing [`uucaix-jpg/ComfyUI`](https://github.com/uucaix-jpg/ComfyUI)'s headless inference runtime (`comfy-aimdo` + `comfy-kitchen` + CUDA Graphs) with a zero-dependency local Studio semantic client.

## Files

- `colab_comfyui_deploy.ipynb`: Single-cell idempotent Colab GPU compute node (`0.0s` warm re-run, `/dev/shm` + Drive cold-mirror weight tiering, `cloudflared` tunnel, `COLAB_READY_JSON` + automatic `127.0.0.1:18080/api/admin/config` handshake).
- `comfy_studio_client.py`: Pure Python stdlib client (zero third-party dependencies) that compiles `T2VA` / `I2VA` / `FL2VA` / `Ref2VA` / multi-shot (`1–50` shots chained in GPU RAM) / optional `TextGenerateV2` into a single atomic ComfyUI `/prompt` DAG.

## Quick Start

### 1. Start the Colab Compute Node (Optional Drive Cold Mirror)

Open [`colab_comfyui_deploy.ipynb`](https://colab.research.google.com/github/uucaix-jpg/colab-comfy-minimal/blob/main/colab_comfyui_deploy.ipynb) in Colab and run the single code cell. Copy the emitted `colab_url` (or let your local `127.0.0.1:18080` Studio receive it automatically).

### 2. Run Locally Against Colab (or Against a Local GPU ComfyUI)

```bash
git clone https://github.com/uucaix-jpg/colab-comfy-minimal.git
cd colab-comfy-minimal

# Single-shot 8-step Turbo FL2VA (works on 24GB L4 / 40GB A100 / 80GB A100 / H100)
python3 comfy_studio_client.py \
  --url "https://xxxx.trycloudflare.com" \
  --prompt "Camera pushes forward along a rainy neon alley, crisp footsteps" \
  --first-frame start.png --last-frame end.png --turbo --out shot.mp4

# Multi-shot continuation (4x 5s = 20s) in ONE atomic /prompt DAG + optional vLLM storyboard split
python3 comfy_studio_client.py \
  --url "https://xxxx-comfy.trycloudflare.com" \
  --llm-url "https://xxxx-vllm.trycloudflare.com" \
  --shots 4 --turbo \
  --prompt "A traveler opens an old bookshop door as wind chimes ring" \
  --out film_20s.mp4
```

To run **100% locally** on your own NVIDIA GPU, start ComfyUI on `127.0.0.1:8188` and pass `--url http://127.0.0.1:8188` with zero code changes.
