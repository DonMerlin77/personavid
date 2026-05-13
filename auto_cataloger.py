"""
auto_cataloger.py — Watch footage/ for new clips and auto-process them.

Runs in two modes:
  --watch   Continuously watch references/footage/ for new files (run in background)
  --scan    One-shot: find all un-cataloged clips and process them now

Processing pipeline per new clip:
  1. Detect codec — if HEVC (iPhone), transcode via prep_iphone_footage.py
  2. Verify output dimensions — flag if not portrait (1080×1920) when source was portrait
  3. Check backdrop color — warn if likely white/cream (RVM will fail)
  4. Run footage_catalog.py on the processed file to generate catalog entry
  5. If backdrop warning, add skip=True + skip_reason to catalog entry
  6. Print summary

Usage:
    python3 auto_cataloger.py --scan              # process all new clips now
    python3 auto_cataloger.py --scan --force      # re-catalog everything
    python3 auto_cataloger.py --watch             # run as daemon (Ctrl+C to stop)
    python3 auto_cataloger.py --check <file>      # check one file only
"""

import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime

ROOT        = Path(__file__).parent
FOOTAGE_DIR = ROOT / "references" / "footage"
CATALOG_PATH = ROOT / "footage_catalog.json"
PROCESSED_LOG = ROOT / "logs" / "auto_catalog_log.json"

(ROOT / "logs").mkdir(exist_ok=True)

VIDEO_EXTS = {".mp4", ".mov", ".MP4", ".MOV", ".m4v"}

WATCH_INTERVAL = 30  # seconds between scans in --watch mode


# ── Codec and dimension detection ─────────────────────────────────────────────

def get_codec(path: Path) -> str:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name", "-of", "default=nw=1", str(path)],
        capture_output=True, text=True
    )
    return result.stdout.strip().lower()


def get_dimensions(path: Path) -> tuple[int, int]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True
    )
    parts = result.stdout.strip().split(",")
    if len(parts) == 2:
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            pass
    return 0, 0


def detect_backdrop_color(path: Path) -> str:
    """
    Sample the center-top region of the first frame to estimate backdrop color.
    Returns 'white', 'blue', 'green', 'dark', or 'unknown'.
    Samples a 100x100 region from the top-center of the frame.
    """
    import tempfile
    import struct

    tmp = Path(tempfile.mktemp(suffix=".ppm"))
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", "1", "-i", str(path),
             "-vframes", "1", "-vf", "crop=100:100:iw/2-50:ih*0.1",
             str(tmp)],
            capture_output=True, check=False
        )
        if not tmp.exists() or tmp.stat().st_size < 100:
            return "unknown"

        data = tmp.read_bytes()
        # PPM header: P6\n<w> <h>\n<maxval>\n then raw RGB
        lines = []
        i = 0
        while len(lines) < 3 and i < len(data):
            end = data.index(b"\n", i)
            lines.append(data[i:end].decode("ascii", errors="ignore").strip())
            i = end + 1
        pixel_data = data[i:]

        # Average RGB over the sampled region
        r_sum = g_sum = b_sum = 0
        n = len(pixel_data) // 3
        for j in range(n):
            r_sum += pixel_data[j * 3]
            g_sum += pixel_data[j * 3 + 1]
            b_sum += pixel_data[j * 3 + 2]

        if n == 0:
            return "unknown"

        r = r_sum / n
        g = g_sum / n
        b = b_sum / n

        if r > 200 and g > 200 and b > 200:
            return "white"
        if b > r + 30 and b > g + 20 and b > 80:
            return "blue"
        if g > r + 20 and g > b + 20 and g > 80:
            return "green"
        if r < 60 and g < 60 and b < 60:
            return "dark"
        return "other"

    except Exception:
        return "unknown"
    finally:
        if tmp.exists():
            tmp.unlink()


# ── Processing pipeline ───────────────────────────────────────────────────────

def load_catalog() -> dict:
    if CATALOG_PATH.exists():
        try:
            return json.loads(CATALOG_PATH.read_text())
        except Exception:
            pass
    return {}


def save_catalog(catalog: dict):
    CATALOG_PATH.write_text(json.dumps(catalog, indent=2))


def load_log() -> list:
    if PROCESSED_LOG.exists():
        try:
            return json.loads(PROCESSED_LOG.read_text())
        except Exception:
            pass
    return []


def save_log(entries: list):
    PROCESSED_LOG.write_text(json.dumps(entries[-200:], indent=2))


def get_unprocessed_clips(force: bool = False) -> list[Path]:
    """Return footage/ clips that aren't yet in the catalog."""
    catalog = load_catalog()
    clips = []
    for path in sorted(FOOTAGE_DIR.iterdir()):
        if path.suffix not in VIDEO_EXTS:
            continue
        if path.name.endswith("Zone.Identifier"):
            continue
        if not force and path.name in catalog:
            continue
        clips.append(path)
    return clips


def transcode_if_needed(path: Path) -> Path | None:
    """
    If the file is HEVC (iPhone), transcode to H.264 _web.mp4 in the same directory.
    Returns the output path (which may be the original if no transcode needed).
    Returns None on failure.
    """
    codec = get_codec(path)
    if "hevc" not in codec and "hvc1" not in codec:
        return path  # already H.264

    web_path = path.parent / (path.stem + "_web.mp4")
    if web_path.exists():
        print(f"  [transcode] {web_path.name} already exists, using it")
        return web_path

    print(f"  [transcode] HEVC detected — transcoding {path.name}...")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prep_iphone_footage.py"),
            str(path),
        ],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  [transcode] FAILED: {result.stderr[-200:]}")
        return None
    if web_path.exists():
        return web_path
    print(f"  [transcode] Expected output {web_path.name} not found")
    return None


def run_cataloger(path: Path) -> bool:
    """Run footage_catalog.py on a single clip. Returns True on success."""
    print(f"  [catalog] Running footage_catalog.py on {path.name}...")
    result = subprocess.run(
        [sys.executable, str(ROOT / "footage_catalog.py"), str(path)],
        capture_output=True, text=True, cwd=str(ROOT)
    )
    if result.returncode != 0:
        print(f"  [catalog] FAILED: {result.stderr[-200:]}")
        return False
    return True


def process_clip(path: Path) -> dict:
    """Full pipeline for one clip. Returns a log entry."""
    entry = {
        "filename":   path.name,
        "processed_at": datetime.now().isoformat(),
        "steps":      [],
        "warnings":   [],
        "outcome":    "ok",
    }

    # Step 1: transcode if needed
    processed = transcode_if_needed(path)
    if processed is None:
        entry["outcome"] = "transcode_failed"
        entry["steps"].append("transcode: FAILED")
        return entry

    if processed != path:
        entry["steps"].append(f"transcode: {path.name} → {processed.name}")
        path = processed
    else:
        entry["steps"].append("transcode: not needed (already H.264)")

    # Step 2: verify dimensions
    w, h = get_dimensions(path)
    entry["dimensions"] = f"{w}x{h}"
    entry["steps"].append(f"dimensions: {w}x{h}")

    # Portrait check (only for clips that were originally iPhone portrait)
    if w > h and max(w, h) > 1000:
        entry["warnings"].append(f"Landscape clip ({w}x{h}) — verify this is intentional")

    # Step 3: backdrop color check
    backdrop = detect_backdrop_color(path)
    entry["backdrop_color"] = backdrop
    entry["steps"].append(f"backdrop: {backdrop}")

    if backdrop in ("white", "unknown") and (h > w):
        entry["warnings"].append(
            f"Backdrop appears {backdrop} — RVM may fail (white/cream fur disappears against white). "
            "Use a blue or green sheet behind Carter/Cash."
        )

    # Step 4: run footage_catalog.py
    ok = run_cataloger(path)
    if not ok:
        entry["outcome"] = "catalog_failed"
        entry["steps"].append("catalog: FAILED")
        return entry

    entry["steps"].append("catalog: OK")

    # Step 5: if white backdrop, mark skip in catalog
    if backdrop == "white" and (h > w):
        catalog = load_catalog()
        if path.name in catalog:
            catalog[path.name]["skip"] = True
            catalog[path.name]["skip_reason"] = "white_backdrop"
            save_catalog(catalog)
            entry["steps"].append("catalog: marked skip=True (white_backdrop)")
            entry["warnings"].append("Added skip=True to catalog entry — re-film with colored backdrop")

    if entry["warnings"]:
        entry["outcome"] = "ok_with_warnings"

    return entry


def scan(force: bool = False):
    clips = get_unprocessed_clips(force=force)
    if not clips:
        print("[AutoCataloger] No new clips to process.")
        return

    print(f"[AutoCataloger] Found {len(clips)} clip(s) to process")
    log = load_log()

    for i, clip in enumerate(clips):
        print(f"\n[{i+1}/{len(clips)}] {clip.name}")
        entry = process_clip(clip)
        log.append(entry)

        for w in entry["warnings"]:
            print(f"  [WARN] {w}")
        print(f"  → {entry['outcome']}")

    save_log(log)
    print(f"\n[AutoCataloger] Done. Log → {PROCESSED_LOG}")


def watch():
    print(f"[AutoCataloger] Watching {FOOTAGE_DIR} every {WATCH_INTERVAL}s (Ctrl+C to stop)")
    seen = set(p.name for p in FOOTAGE_DIR.iterdir() if p.suffix in VIDEO_EXTS)

    while True:
        time.sleep(WATCH_INTERVAL)
        try:
            current = set(p.name for p in FOOTAGE_DIR.iterdir() if p.suffix in VIDEO_EXTS)
            new = current - seen
            if new:
                print(f"\n[AutoCataloger] {len(new)} new file(s): {', '.join(sorted(new))}")
                for fname in sorted(new):
                    path = FOOTAGE_DIR / fname
                    entry = process_clip(path)
                    log = load_log()
                    log.append(entry)
                    save_log(log)
                    for w in entry["warnings"]:
                        print(f"  [WARN] {w}")
                    print(f"  → {entry['outcome']}")
            seen = current
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[AutoCataloger] Error: {e}")


if __name__ == "__main__":
    args = sys.argv[1:]
    force = "--force" in args

    if "--watch" in args:
        watch()
    elif "--check" in args:
        idx = args.index("--check")
        if idx + 1 < len(args):
            p = Path(args[idx + 1])
            if not p.exists():
                p = FOOTAGE_DIR / args[idx + 1]
            entry = process_clip(p)
            print(json.dumps(entry, indent=2))
        else:
            print("Usage: --check <filename>")
    elif "--scan" in args or not args:
        scan(force=force)
    else:
        print(__doc__)
