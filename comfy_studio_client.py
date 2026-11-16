#!/usr/bin/env python3
"""Zero-dependency semantic client compiling Studio requests into atomic ComfyUI DAGs."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable

H3_CLIP_CKPT = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
H3_VIDEO_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
H3_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
H3_FL2VA_UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
H3_REF2VA_UNET = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
H3_FL2VA_TURBO_LORA = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
H3_REF2VA_TURBO_LORA = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
H3_FPS = 24.0


def align_h3_frames(length: int) -> int:
    n = max(5, int(length))
    while n % 17 != 5:
        n += 1
    return n


def split_storyboard_via_llm(
    prompt: str,
    num_shots: int,
    llm_base_url: str,
    llm_model: str = "",
    llm_api_key: str = "EMPTY",
    timeout: float = 90.0,
) -> list[str]:
    """Expand a user concept into N chronological audio-visual shot descriptions via OpenAI-compatible vLLM/LLM."""
    if num_shots <= 1:
        return [prompt]
    base = llm_base_url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {llm_api_key}"}

    if not llm_model:
        req = urllib.request.Request(f"{base}/models", headers=headers)
        with urllib.request.urlopen(req, timeout=15.0) as resp:
            models = json.loads(resp.read().decode("utf-8")).get("data") or []
            llm_model = models[0]["id"] if models else "default"

    sys_msg = (
        f"Split the user's concept into {num_shots} continuous 5-second visual+audio shots where each shot "
        "starts seamlessly from the previous shot's ending pose. Output ONLY a JSON array of strings, "
        f"length {num_shots}, each describing concrete camera framing, motion, lighting, ambient sound, and quoted dialogue."
    )
    body = json.dumps({
        "model": llm_model,
        "messages": [{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}],
        "temperature": 0.6,
    }).encode("utf-8")
    req = urllib.request.Request(f"{base}/chat/completions", data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        content = json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"]
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    match = re.search(r"\[.*\]", content, flags=re.DOTALL)
    if match:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, list) and len(parsed) >= num_shots:
            return [str(x).strip() for x in parsed[:num_shots]]
    return [f"{prompt} (Shot {i + 1}/{num_shots})" for i in range(num_shots)]


def build_textgen_graph(
    prompt: str,
    clip_name: str,
    clip_type: str = "joyimage",
    image_names: list[str] | None = None,
    max_length: int = 1024,
    temperature: float = 0.7,
    seed: int = 42,
    thinking: bool = False,
) -> tuple[dict[str, Any], str]:
    """Compile a native ComfyUI TextGenerateV2 + PreviewAny DAG (for full generative VLMs with lm_head)."""
    dag: dict[str, Any] = {
        "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_name, "type": clip_type}},
    }
    tg_inputs: dict[str, Any] = {
        "clip": ["1", 0],
        "prompt": prompt,
        "max_length": int(max_length),
        "sampling_mode": "on",
        "sampling_mode.temperature": float(temperature),
        "sampling_mode.top_k": 64,
        "sampling_mode.top_p": 0.95,
        "sampling_mode.min_p": 0.05,
        "sampling_mode.repetition_penalty": 1.05,
        "sampling_mode.presence_penalty": 0.0,
        "sampling_mode.seed": int(seed),
        "thinking": bool(thinking),
        "use_default_template": True,
        "mtp": "auto",
    }
    nid = 10
    for idx, img_name in enumerate(image_names or [], start=1):
        img_id = str(nid)
        nid += 1
        dag[img_id] = {"class_type": "LoadImage", "inputs": {"image": img_name}}
        tg_inputs[f"images.image{idx}"] = [img_id, 0]

    dag["2"] = {"class_type": "TextGenerateV2", "inputs": tg_inputs}
    dag["3"] = {"class_type": "PreviewAny", "inputs": {"source": ["2", 0]}}
    return dag, "3"


def build_h3_video_graph(
    prompts: str | list[str],
    *,
    mode: str = "auto",
    width: int = 1344,
    height: int = 768,
    length: int = 124,
    steps: int | None = None,
    cfg: float = 1.0,
    seed: int = 42,
    sampler_name: str = "res_multistep",
    scheduler: str = "simple",
    shift_video: float = 12.0,
    shift_audio: float = 3.0,
    turbo: bool = False,
    first_frame: str | None = None,
    last_frame: str | None = None,
    ref_images: list[str] | None = None,
    ref_audios: list[str] | None = None,
    ref_videos: list[str] | None = None,
    ref_image_size: str = "match",
    guides: list[dict[str, Any]] | None = None,
    filename_prefix: str = "MiniMax_H3/studio",
    crf: float = 15.0,
) -> tuple[dict[str, Any], str]:
    """Compile single-shot or multi-shot (5s-60s) MiniMax-H3 AV generation into ONE atomic ComfyUI DAG."""
    shot_prompts = [prompts] if isinstance(prompts, str) else list(prompts)
    if not shot_prompts:
        raise ValueError("prompts cannot be empty")
    if len(shot_prompts) > 50:
        raise ValueError("BatchImagesNode supports up to 50 chained shots")

    has_refs = bool(ref_images or ref_audios or ref_videos)
    use_ref2va = (mode == "ref2va") or (mode == "auto" and has_refs)
    unet_name = H3_REF2VA_UNET if use_ref2va else H3_FL2VA_UNET
    lora_name = H3_REF2VA_TURBO_LORA if use_ref2va else H3_FL2VA_TURBO_LORA
    if steps is None:
        steps = (4 if use_ref2va else 8) if turbo else 25
    frame_length = align_h3_frames(length)

    dag: dict[str, Any] = {
        "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": H3_CLIP_CKPT, "type": "minimax"}},
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": H3_VIDEO_VAE}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": H3_AUDIO_VAE}},
    }

    model_ref = ["2", 0]
    if turbo:
        dag["5"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": model_ref, "lora_name": lora_name, "strength_model": 1.0},
        }
        model_ref = ["5", 0]

    dag["6"] = {
        "class_type": "MiniMaxH3SigmaShift",
        "inputs": {"model": model_ref, "shift_video": float(shift_video), "shift_audio": float(shift_audio)},
    }
    patched_model = ["6", 0]

    next_id = 20

    def alloc() -> str:
        nonlocal next_id
        s = str(next_id)
        next_id += 1
        return s

    first_img_ref = None
    if first_frame:
        nid = alloc()
        dag[nid] = {"class_type": "LoadImage", "inputs": {"image": first_frame}}
        first_img_ref = [nid, 0]

    last_img_ref = None
    if last_frame:
        nid = alloc()
        dag[nid] = {"class_type": "LoadImage", "inputs": {"image": last_frame}}
        last_img_ref = [nid, 0]

    ref_img_refs: list[list[Any]] = []
    for name in ref_images or []:
        nid = alloc()
        dag[nid] = {"class_type": "LoadImage", "inputs": {"image": name}}
        ref_img_refs.append([nid, 0])

    ref_aud_refs: list[list[Any]] = []
    for name in ref_audios or []:
        nid = alloc()
        dag[nid] = {"class_type": "LoadAudio", "inputs": {"audio": name}}
        ref_aud_refs.append([nid, 0])

    ref_vid_refs: list[tuple[list[Any], list[Any]]] = []
    for name in ref_videos or []:
        vid_id = alloc()
        cmp_id = alloc()
        dag[vid_id] = {"class_type": "LoadVideo", "inputs": {"file": name}}
        dag[cmp_id] = {"class_type": "GetVideoComponents", "inputs": {"video": [vid_id, 0]}}
        ref_vid_refs.append(([cmp_id, 0], [cmp_id, 1]))

    shot_video_streams: list[list[Any]] = []
    shot_audio_streams: list[list[Any]] = []
    prev_raw_frames: list[Any] | None = None

    for idx, shot_prompt in enumerate(shot_prompts):
        is_first_shot = idx == 0
        is_last_shot = idx == len(shot_prompts) - 1
        cond_id = alloc()

        if use_ref2va and is_first_shot:
            c_inputs: dict[str, Any] = {
                "clip": ["1", 0],
                "vae": ["3", 0],
                "audio_vae": ["4", 0],
                "prompt": shot_prompt,
                "width": int(width),
                "height": int(height),
                "length": frame_length,
                "ref_image_size": ref_image_size,
            }
            for i, r in enumerate(ref_img_refs):
                c_inputs[f"ref_images.ref_image_{i}"] = r
            for i, (v_ref, a_ref) in enumerate(ref_vid_refs):
                c_inputs[f"ref_videos.ref_video_{i}"] = v_ref
                c_inputs[f"ref_video_audios.ref_video_audio_{i}"] = a_ref
            for i, r in enumerate(ref_aud_refs):
                c_inputs[f"ref_audios.ref_audio_{i}"] = r
            dag[cond_id] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": c_inputs}
        else:
            c_inputs = {
                "clip": ["1", 0],
                "vae": ["3", 0],
                "prompt": shot_prompt,
                "width": int(width),
                "height": int(height),
                "length": frame_length,
            }
            if is_first_shot and first_img_ref is not None:
                c_inputs["first_frame"] = first_img_ref
            elif not is_first_shot and prev_raw_frames is not None:
                tail_id = alloc()
                dag[tail_id] = {
                    "class_type": "ImageFromBatch",
                    "inputs": {"image": prev_raw_frames, "batch_index": -1, "length": 1},
                }
                c_inputs["first_frame"] = [tail_id, 0]
            if is_last_shot and last_img_ref is not None:
                c_inputs["last_frame"] = last_img_ref
            dag[cond_id] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": c_inputs}

        pos_ref = [cond_id, 0]
        lat_ref = [cond_id, 1]

        if is_first_shot and guides:
            for g in guides:
                gid = alloc()
                g_inputs: dict[str, Any] = {
                    "positive": pos_ref,
                    "latent": lat_ref,
                    "frame_idx": int(g.get("frame_idx", 0)),
                }
                if g.get("image"):
                    img_id = alloc()
                    dag[img_id] = {"class_type": "LoadImage", "inputs": {"image": g["image"]}}
                    g_inputs["vae"] = ["3", 0]
                    g_inputs["image"] = [img_id, 0]
                if g.get("audio"):
                    aud_id = alloc()
                    dag[aud_id] = {"class_type": "LoadAudio", "inputs": {"audio": g["audio"]}}
                    g_inputs["audio_vae"] = ["4", 0]
                    g_inputs["audio"] = [aud_id, 0]
                dag[gid] = {"class_type": "MiniMaxH3AddGuide", "inputs": g_inputs}
                pos_ref = [gid, 0]

        neg_id = alloc()
        ks_id = alloc()
        sep_id = alloc()
        vdec_id = alloc()
        adec_id = alloc()

        dag[neg_id] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": pos_ref}}
        dag[ks_id] = {
            "class_type": "KSampler",
            "inputs": {
                "model": patched_model,
                "seed": int(seed) + idx,
                "steps": int(steps),
                "cfg": float(cfg),
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "positive": pos_ref,
                "negative": [neg_id, 0],
                "latent_image": lat_ref,
                "denoise": 1.0,
            },
        }
        dag[sep_id] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": [ks_id, 0]}}
        dag[vdec_id] = {"class_type": "VAEDecode", "inputs": {"samples": [sep_id, 0], "vae": ["3", 0]}}
        dag[adec_id] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": [sep_id, 1], "vae": ["4", 0]}}

        prev_raw_frames = [vdec_id, 0]
        if is_first_shot:
            shot_video_streams.append([vdec_id, 0])
            shot_audio_streams.append([adec_id, 0])
        else:
            # Drop boundary frame 0 and matching 1/24s audio to avoid duplicate freeze-frame or AV drift
            vtrim_id = alloc()
            atrim_id = alloc()
            dag[vtrim_id] = {
                "class_type": "ImageFromBatch",
                "inputs": {"image": [vdec_id, 0], "batch_index": 1, "length": 4096},
            }
            dag[atrim_id] = {
                "class_type": "TrimAudioDuration",
                "inputs": {"audio": [adec_id, 0], "start_index": round(1.0 / H3_FPS, 6), "duration": 3600.0},
            }
            shot_video_streams.append([vtrim_id, 0])
            shot_audio_streams.append([atrim_id, 0])

    if len(shot_video_streams) == 1:
        final_images = shot_video_streams[0]
        final_audio = shot_audio_streams[0]
    else:
        batch_id = alloc()
        dag[batch_id] = {
            "class_type": "BatchImagesNode",
            "inputs": {f"images.image{i}": ref for i, ref in enumerate(shot_video_streams)},
        }
        final_images = [batch_id, 0]
        cur_audio = shot_audio_streams[0]
        for nxt_audio in shot_audio_streams[1:]:
            cat_id = alloc()
            dag[cat_id] = {
                "class_type": "AudioConcat",
                "inputs": {"audio1": cur_audio, "audio2": nxt_audio, "direction": "after"},
            }
            cur_audio = [cat_id, 0]
        final_audio = cur_audio

    cv_id = alloc()
    sv_id = alloc()
    dag[cv_id] = {"class_type": "CreateVideo", "inputs": {"images": final_images, "audio": final_audio, "fps": H3_FPS}}
    dag[sv_id] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": [cv_id, 0],
            "filename_prefix": filename_prefix,
            "format": "mp4",
            "format.codec": "h264",
            "format.codec.encoding": "re-encode",
            "format.codec.encoding.crf": float(crf),
        },
    }
    return dag, sv_id


class ComfyColabClient:
    """Thin stdlib HTTP client for a remote Colab ComfyUI compute node."""

    def __init__(self, base_url: str | None = None, state_file: str | Path = "/content/colab_gateway_state.json"):
        resolved = base_url or os.environ.get("COMFY_BASE_URL") or ""
        if not resolved and Path(state_file).exists():
            resolved = json.loads(Path(state_file).read_text(encoding="utf-8")).get("colab_url", "")
        if not resolved:
            raise ValueError("Provide base_url, set COMFY_BASE_URL, or point state_file to colab_gateway_state.json")
        self.base_url = resolved.rstrip("/")
        self.client_id = uuid.uuid4().hex
        self._uploaded_hashes: dict[str, str] = {}

    def _json_req(self, method: str, path: str, payload: dict[str, Any] | None = None, timeout: float = 60.0) -> Any:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if data is not None else {}
        req = urllib.request.Request(f"{self.base_url}{path}", data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} {path}: {detail}") from exc

    def system_stats(self) -> dict[str, Any]:
        return self._json_req("GET", "/system_stats", timeout=10.0)

    def upload_media(self, local_path: str | Path) -> str:
        """Upload image/audio/video once by SHA-256 content-addressed filename into ComfyUI's input directory."""
        p = Path(local_path)
        raw = p.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()[:16]
        if digest in self._uploaded_hashes:
            return self._uploaded_hashes[digest]

        safe_ext = re.sub(r"[^a-zA-Z0-9.]", "", p.suffix.lower()) or ".bin"
        remote_name = f"sha256_{digest}{safe_ext}"
        mime = mimetypes.guess_type(remote_name)[0] or "application/octet-stream"
        boundary = f"----ComfyBoundary{uuid.uuid4().hex}"
        parts = [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n".encode(),
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ninput\r\n".encode(),
            (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{remote_name}\"\r\n"
                f"Content-Type: {mime}\r\n\r\n"
            ).encode()
            + raw
            + f"\r\n--{boundary}--\r\n".encode(),
        ]
        body = b"".join(parts)
        req = urllib.request.Request(
            f"{self.base_url}/upload/image",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180.0) as resp:
            saved = json.loads(resp.read().decode("utf-8"))["name"]
        self._uploaded_hashes[digest] = saved
        return saved

    def submit_and_wait(
        self,
        dag: dict[str, Any],
        *,
        poll_interval: float = 2.0,
        timeout: float = 3600.0,
        on_status: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        queued = self._json_req("POST", "/prompt", {"prompt": dag, "client_id": self.client_id})
        prompt_id = queued["prompt_id"]
        if on_status:
            on_status(f"queued:{prompt_id}")

        deadline = time.time() + timeout
        while time.time() < deadline:
            hist = self._json_req("GET", f"/history/{prompt_id}", timeout=30.0)
            if prompt_id in hist:
                entry = hist[prompt_id]
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI execution failed: {json.dumps(status.get('messages', []), ensure_ascii=False)}")
                if entry.get("outputs"):
                    return entry["outputs"]
            time.sleep(poll_interval)
        raise TimeoutError(f"Prompt {prompt_id} exceeded timeout ({timeout}s)")

    def download_output_file(self, descriptor: dict[str, str], output_path: str | Path) -> Path:
        query = urllib.parse.urlencode({
            "filename": descriptor["filename"],
            "subfolder": descriptor.get("subfolder", ""),
            "type": descriptor.get("type", "output"),
        })
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{self.base_url}/view?{query}", timeout=300.0) as resp:
            dest.write_bytes(resp.read())
        return dest

    def generate_video(
        self,
        prompt: str,
        output_path: str | Path = "output.mp4",
        *,
        num_shots: int = 1,
        llm_base_url: str | None = None,
        llm_model: str = "",
        llm_api_key: str = "EMPTY",
        mode: str = "auto",
        width: int = 1344,
        height: int = 768,
        length: int = 124,
        steps: int | None = None,
        seed: int = 42,
        turbo: bool = False,
        first_frame_path: str | Path | None = None,
        last_frame_path: str | Path | None = None,
        ref_image_paths: list[str | Path] | None = None,
        ref_audio_paths: list[str | Path] | None = None,
        ref_video_paths: list[str | Path] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> Path:
        if num_shots > 1 and llm_base_url:
            prompts: str | list[str] = split_storyboard_via_llm(
                prompt, num_shots, llm_base_url, llm_model=llm_model, llm_api_key=llm_api_key
            )
        elif num_shots > 1:
            prompts = [prompt] * num_shots
        else:
            prompts = prompt

        first_remote = self.upload_media(first_frame_path) if first_frame_path else None
        last_remote = self.upload_media(last_frame_path) if last_frame_path else None
        ref_imgs = [self.upload_media(p) for p in (ref_image_paths or [])]
        ref_auds = [self.upload_media(p) for p in (ref_audio_paths or [])]
        ref_vids = [self.upload_media(p) for p in (ref_video_paths or [])]

        dag, save_id = build_h3_video_graph(
            prompts,
            mode=mode,
            width=width,
            height=height,
            length=length,
            steps=steps,
            seed=seed,
            turbo=turbo,
            first_frame=first_remote,
            last_frame=last_remote,
            ref_images=ref_imgs,
            ref_audios=ref_auds,
            ref_videos=ref_vids,
        )
        outputs = self.submit_and_wait(dag, on_status=on_status)
        saved_items = outputs[save_id].get("images") or outputs[save_id].get("videos") or []
        if not saved_items:
            raise RuntimeError(f"SaveVideo node {save_id} returned no file descriptor: {outputs[save_id]}")
        return self.download_output_file(saved_items[0], output_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Submit an atomic MiniMax-H3 workflow to a Colab ComfyUI compute node.")
    ap.add_argument("--url", default=os.environ.get("COMFY_BASE_URL"), help="Colab trycloudflare URL")
    ap.add_argument("--prompt", required=True, help="Scene prompt (use <Picture 1> / <Audio 1> / <Video 1> for ref2va)")
    ap.add_argument("--out", default="studio_output.mp4", help="Local output MP4 path")
    ap.add_argument("--mode", choices=["auto", "fl2va", "ref2va"], default="auto")
    ap.add_argument("--shots", type=int, default=1, help="Number of chained 5s shots in a single atomic DAG (1-12)")
    ap.add_argument("--llm-url", default=os.environ.get("LLM_BASE_URL"), help="Optional vLLM tunnel URL for shot splitting")
    ap.add_argument("--width", type=int, default=1344)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--length", type=int, default=124)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--turbo", action="store_true", help="Enable 8-step (fl2va) / 4-step (ref2va) Turbo LoRA")
    ap.add_argument("--first-frame", default=None)
    ap.add_argument("--last-frame", default=None)
    ap.add_argument("--ref-image", action="append", default=[])
    ap.add_argument("--ref-audio", action="append", default=[])
    ap.add_argument("--ref-video", action="append", default=[])
    ap.add_argument("--dump-dag", action="store_true", help="Print compiled ComfyUI /prompt JSON and exit")
    args = ap.parse_args()

    if args.dump_dag:
        dag, _ = build_h3_video_graph(
            [args.prompt] * max(1, args.shots),
            mode=args.mode,
            width=args.width,
            height=args.height,
            length=args.length,
            steps=args.steps,
            seed=args.seed,
            turbo=args.turbo,
            first_frame=args.first_frame,
            last_frame=args.last_frame,
            ref_images=args.ref_image,
            ref_audios=args.ref_audio,
            ref_videos=args.ref_video,
        )
        print(json.dumps(dag, ensure_ascii=False, indent=2))
        return

    client = ComfyColabClient(base_url=args.url)
    saved = client.generate_video(
        args.prompt,
        args.out,
        num_shots=args.shots,
        llm_base_url=args.llm_url,
        mode=args.mode,
        width=args.width,
        height=args.height,
        length=args.length,
        steps=args.steps,
        seed=args.seed,
        turbo=args.turbo,
        first_frame_path=args.first_frame,
        last_frame_path=args.last_frame,
        ref_image_paths=args.ref_image,
        ref_audio_paths=args.ref_audio,
        ref_video_paths=args.ref_video,
        on_status=lambda s: print(f"[comfy] {s}"),
    )
    print(f"[done] saved -> {saved}")


if __name__ == "__main__":
    main()
