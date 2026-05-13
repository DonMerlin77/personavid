"""
modal_extract_batch.py — Batch RVM extraction on Modal A10G GPU.

Reads footage_catalog.json, finds all cat clips without existing extraction,
uploads each to Modal, runs RVM on GPU, saves fg + alpha locally.

Usage:
    python3 modal_extract_batch.py --dry-run   # show clips + cost estimate
    python3 modal_extract_batch.py             # extract all unextracted cat clips
    python3 modal_extract_batch.py --rvm-ready # only clips marked rvm_ready=True
    python3 modal_extract_batch.py --limit 5   # process first N clips only
"""

import json
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
import modal

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

CATALOG_FILE  = ROOT / "footage_catalog.json"
FOOTAGE_DIR   = ROOT / "references" / "footage"
EXTRACTED_DIR = ROOT / "outputs" / "extracted"
EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

A10G_PER_SEC = 0.000583

# ── Modal app + GPU function (must be global scope) ───────────────────────────

app = modal.App("personavid-extractor")

def _download_rvm():
    import torch
    torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3", trust_repo=True)
    print("RVM model pre-baked into image.")

EXTRACT_IMAGE = (
    modal.Image.from_registry("pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime")
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "UTC"})
    .apt_install("ffmpeg", "git")
    .pip_install("tqdm", "numpy", "python-dotenv", "torchvision", "av")
    .run_function(_download_rvm)
)


@app.function(
    image=EXTRACT_IMAGE,
    gpu="A10G",
    timeout=600,
)
def rvm_extract(clip_bytes: bytes, clip_name: str) -> tuple[bytes, bytes]:
    import tempfile, subprocess
    import torch
    import numpy as np

    tmp = tempfile.mkdtemp()
    clip_path  = f"{tmp}/{clip_name}"
    fg_path    = f"{tmp}/fg.mp4"
    alpha_path = f"{tmp}/alpha.mp4"

    with open(clip_path, "wb") as f:
        f.write(clip_bytes)

    # Get video info via ffprobe
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "csv=p=0", clip_path],
        capture_output=True, text=True
    )
    parts = probe.stdout.strip().split(",")
    W, H = int(parts[0]), int(parts[1])
    fps_str = parts[2].strip()
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(int(num) / int(den))
    else:
        fps = float(fps_str)
    fps_arg = str(round(fps, 4))

    # Load RVM model (pre-baked into image — no download needed)
    model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3", trust_repo=True)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)

    # Read frames via torchvision (uses av as backend — reading works fine with av 17)
    import torchvision
    reader = torchvision.io.VideoReader(clip_path, "video")

    # Detect available encoder: prefer nvenc (GPU, fast), fall back to libx264 or mpeg4
    enc_probe = subprocess.run(
        ["ffmpeg", "-encoders"], capture_output=True, text=True
    )
    enc_out = enc_probe.stdout + enc_probe.stderr
    if "h264_nvenc" in enc_out:
        encoder, pix = "h264_nvenc", "yuv420p"
    elif "libx264" in enc_out:
        encoder, pix = "libx264", "yuv420p"
    else:
        encoder, pix = "mpeg4", "yuv420p"

    def _ffmpeg_writer(out_path):
        return subprocess.Popen(
            ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{H}", "-r", fps_arg, "-i", "-",
             "-c:v", encoder, "-pix_fmt", pix, out_path],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )

    fg_proc    = _ffmpeg_writer(fg_path)
    alpha_proc = _ffmpeg_writer(alpha_path)

    # Verify writers started
    import time as _time; _time.sleep(0.3)
    if fg_proc.poll() is not None:
        raise RuntimeError(f"fg ffmpeg died ({encoder}): {fg_proc.stderr.read()[-200:].decode()}")
    if alpha_proc.poll() is not None:
        raise RuntimeError(f"alpha ffmpeg died ({encoder}): {alpha_proc.stderr.read()[-200:].decode()}")

    rec = [None] * 4
    for packet in reader:
        t = packet["data"].float().div(255).unsqueeze(0).to(device)
        with torch.no_grad():
            fgr, pha, *rec = model(t, *rec, 0.25)
        fgr_np  = (fgr[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        pha_np  = (pha[0, 0].cpu().numpy() * 255).astype("uint8")
        fg_proc.stdin.write(fgr_np.tobytes())
        alpha_proc.stdin.write(np.stack([pha_np, pha_np, pha_np], 2).tobytes())

    fg_proc.stdin.close()
    fg_err = fg_proc.stderr.read(); fg_proc.wait()
    alpha_proc.stdin.close()
    alpha_err = alpha_proc.stderr.read(); alpha_proc.wait()

    if fg_proc.returncode != 0:
        raise RuntimeError(f"fg ffmpeg failed (encoder={encoder}): {fg_err[-300:].decode()}")
    if alpha_proc.returncode != 0:
        raise RuntimeError(f"alpha ffmpeg failed: {alpha_err[-300:].decode()}")

    return open(fg_path, "rb").read(), open(alpha_path, "rb").read()


# ── Local helpers ─────────────────────────────────────────────────────────────

def get_unextracted_clips(rvm_ready_only: bool = False, names: list[str] = None) -> list[Path]:
    catalog = json.loads(CATALOG_FILE.read_text())
    clips = []
    for entry in catalog.values():
        if entry.get("skip"):
            continue
        if entry.get("primary_subject") not in ("carter", "cash", "both"):
            continue
        if rvm_ready_only and not entry.get("rvm_ready"):
            continue
        windows = [w for w in entry.get("usable_windows", []) if "human" not in w.get("subjects", [])]
        if not windows:
            continue
        fname = entry["filename"]
        if names and fname not in names:
            continue
        stem  = Path(fname).stem.lower().replace(" ", "_")
        if (EXTRACTED_DIR / f"{stem}_fg.mp4").exists():
            continue
        clip_path = FOOTAGE_DIR / fname
        if not clip_path.exists():
            continue
        clips.append(clip_path)
    return clips


def estimate_cost(clips: list[Path]) -> float:
    total_mb = sum(c.stat().st_size for c in clips) / (1024 * 1024)
    est_gpu_sec = total_mb * 1.2
    return est_gpu_sec * A10G_PER_SEC


def run_dry(clips: list[Path]):
    print(f"\nDRY RUN — {len(clips)} clips to extract\n")
    total_mb = 0
    for i, clip in enumerate(clips):
        mb = clip.stat().st_size / (1024 * 1024)
        total_mb += mb
        print(f"  [{i+1:3d}] {clip.name:50s} {mb:6.1f}MB")
    cost = estimate_cost(clips)
    print(f"\n  Total: {total_mb:.0f}MB")
    print(f"  Estimated cost: ${cost:.2f}")
    print(f"  Estimated time: ~{len(clips) * 30 // 60}min (parallel on Modal)")


def run_batch(clips: list[Path]):
    print(f"\nExtracting {len(clips)} clips on Modal A10G GPU")
    print(f"Estimated cost: ${estimate_cost(clips):.2f}\n")

    success, failed = 0, []

    with app.run():
        for i, clip_path in enumerate(clips):
            stem      = clip_path.stem.lower().replace(" ", "_")
            fg_out    = EXTRACTED_DIR / f"{stem}_fg.mp4"
            alpha_out = EXTRACTED_DIR / f"{stem}_alpha.mp4"
            mb        = clip_path.stat().st_size / (1024 * 1024)

            print(f"[{i+1}/{len(clips)}] {clip_path.name} ({mb:.1f}MB)...", flush=True)
            try:
                t0 = time.time()
                fg_bytes, alpha_bytes = rvm_extract.remote(clip_path.read_bytes(), clip_path.name)
                elapsed = time.time() - t0
                fg_out.write_bytes(fg_bytes)
                alpha_out.write_bytes(alpha_bytes)
                print(f"  Done {elapsed:.0f}s (~${elapsed * A10G_PER_SEC:.3f}) → {fg_out.name}")
                success += 1
            except Exception as e:
                print(f"  FAILED: {e}")
                failed.append(clip_path.name)

    print(f"\n{'='*55}")
    print(f"  Extracted: {success}/{len(clips)}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print(f"  Saved to: {EXTRACTED_DIR}")


def main():
    args           = sys.argv[1:]
    dry_run        = "--dry-run" in args
    rvm_ready_only = "--rvm-ready" in args
    limit          = int(args[args.index("--limit") + 1]) if "--limit" in args else None

    clips = get_unextracted_clips(rvm_ready_only=rvm_ready_only)
    if limit:
        clips = clips[:limit]

    if not clips:
        print("All cat clips already extracted.")
        return

    if dry_run:
        run_dry(clips)
    else:
        run_batch(clips)


if __name__ == "__main__":
    main()
