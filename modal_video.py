"""
modal_video.py
==============
PersonaVid — Modal.com GPU video generation.

Runs Wan-2.1 on a rented A10G GPU (serverless, scales to zero).
Supports text-to-video and image-to-video.

Deploy:  modal deploy modal_video.py
Call:    from modal_video import generate_video_modal

Cost: ~$0.05-0.10 per 10-second video (A10G @ ~$1/hr, ~30s per video)
"""

import modal
import io
import os
import base64
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

HF_TOKEN = os.getenv("HF_TOKEN", "")

# ── Modal app definition ────────────────────────────────────────────────────────

app = modal.App("personavid-wan")

# Container image — install diffusers + Wan dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "torchvision",
        extra_options="--index-url https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "diffusers>=0.32.0",
        "transformers>=4.44.0",
        "accelerate>=0.33.0",
        "sentencepiece",
        "imageio[ffmpeg]",
        "Pillow",
        "numpy",
        "huggingface_hub",
    )
)

# Cache the model weights in Modal's persistent volume
model_volume = modal.Volume.from_name("personavid-models", create_if_missing=True)


# ── Text-to-Video ───────────────────────────────────────────────────────────────

hf_secret = modal.Secret.from_dict({"HF_TOKEN": HF_TOKEN, "HUGGING_FACE_HUB_TOKEN": HF_TOKEN})


@app.function(
    image=image,
    gpu="A10G",
    timeout=300,
    volumes={"/models": model_volume},
    secrets=[hf_secret],
)
def text_to_video(
    prompt: str,
    num_frames: int = 81,
    height: int = 832,
    width: int = 480,
    num_inference_steps: int = 20,
    guidance_scale: float = 5.0,
    seed: int = 42,
) -> bytes:
    """
    Generate a video from text using Wan-2.1 T2V 1.3B.
    Returns raw MP4 bytes.

    height=832, width=480 = 9:16 vertical for TikTok/Instagram.
    num_frames=81 ≈ 5 seconds at ~16fps.
    """
    import torch
    from diffusers import WanPipeline
    from diffusers.utils import export_to_video
    import tempfile

    model_id = "Wan-AI/Wan2.1-T2V-1.3B"
    cache_dir = "/models/wan-t2v-1.3b"

    print(f"Loading Wan T2V 1.3B from {cache_dir}...")
    pipe = WanPipeline.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")
    pipe.enable_model_cpu_offload()

    print(f"Generating: {prompt[:80]}...")
    generator = torch.Generator("cuda").manual_seed(seed)
    output = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        export_to_video(output.frames[0], tmp.name, fps=16)
        tmp_path = tmp.name

    with open(tmp_path, "rb") as f:
        video_bytes = f.read()

    os.unlink(tmp_path)
    print(f"Done — {len(video_bytes):,} bytes")
    return video_bytes


# ── Image-to-Video ──────────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="A10G",
    timeout=300,
    volumes={"/models": model_volume},
    secrets=[hf_secret],
)
def image_to_video(
    prompt: str,
    image_b64: str,
    num_frames: int = 81,
    height: int = 832,
    width: int = 480,
    num_inference_steps: int = 20,
    guidance_scale: float = 5.0,
    seed: int = 42,
) -> bytes:
    """
    Animate a source image using Wan-2.1 I2V 14B.
    image_b64: base64-encoded JPEG/PNG.
    Returns raw MP4 bytes.
    """
    import torch
    from diffusers import WanImageToVideoPipeline
    from diffusers.utils import export_to_video, load_image
    from PIL import Image
    import tempfile

    model_id = "Wan-AI/Wan2.1-I2V-14B-480P"
    cache_dir = "/models/wan-i2v-14b"

    print(f"Loading Wan I2V 14B from {cache_dir}...")
    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    pipe.enable_model_cpu_offload()

    # Decode source image
    img_bytes = base64.b64decode(image_b64)
    source_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    source_image = source_image.resize((width, height))

    print(f"Animating image with prompt: {prompt[:80]}...")
    generator = torch.Generator("cuda").manual_seed(seed)
    output = pipe(
        image=source_image,
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        export_to_video(output.frames[0], tmp.name, fps=16)
        tmp_path = tmp.name

    with open(tmp_path, "rb") as f:
        video_bytes = f.read()

    os.unlink(tmp_path)
    print(f"Done — {len(video_bytes):,} bytes")
    return video_bytes


# ── Local caller (used by generator.py) ────────────────────────────────────────

def generate_video_modal(
    prompt: str,
    image_path: str | None = None,
    output_path: str | None = None,
    num_frames: int = 81,
    num_inference_steps: int = 20,
) -> str:
    """
    Call the Modal GPU function from local code.
    Returns the local path of the saved MP4.
    """
    import tempfile

    if image_path:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
        print(f"  [Modal] Submitting I2V job (source: {Path(image_path).name})...")
        video_bytes = image_to_video.remote(
            prompt=prompt,
            image_b64=image_b64,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
        )
    else:
        print(f"  [Modal] Submitting T2V job...")
        video_bytes = text_to_video.remote(
            prompt=prompt,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
        )

    if not output_path:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        output_path = tmp.name
        tmp.close()

    with open(output_path, "wb") as f:
        f.write(video_bytes)

    print(f"  [Modal] Video saved: {output_path} ({len(video_bytes):,} bytes)")
    return output_path


# ── CLI test ────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    prompt = (
        "An orange tabby cat sitting on a cozy living room rug, "
        "warm natural lighting, handheld candid footage, calm mood"
    )
    print("Running test generation...")
    out = generate_video_modal(prompt, output_path="/tmp/modal_test.mp4")
    print(f"Saved to {out}")
