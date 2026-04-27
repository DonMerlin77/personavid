"""
app.py
======
PersonaVid — Web Interface (Gradio).

A clean browser UI for the full PersonaVid pipeline:
  - Upload videos → build style profile
  - View your style fingerprint
  - Generate personalized video prompts (or submit to Kling/Runway)
  - Download or copy the generated prompts

Run:  python3 app.py
      Then open http://localhost:7860
"""

import os
import sys
import json
import shutil
import tempfile
from pathlib import Path
from datetime import datetime

import gradio as gr
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from analyzer import analyze_video, get_style_summary, load_profile, PROFILES
from generator import generate_video


# ── Helpers ─────────────────────────────────────────────────────────────────────

def list_profiles() -> list:
    """Return all existing user profile IDs."""
    return [p.stem for p in PROFILES.glob("*.json")]


def get_profile_stats(user_id: str) -> str:
    """Return formatted profile stats."""
    if not user_id:
        return "No user selected."
    profile = load_profile(user_id)
    if profile["videos_analyzed"] == 0:
        return f"Profile '{user_id}' — no videos analyzed yet."
    fp = profile.get("style_fingerprint", {})
    lines = [
        f"**{user_id}** — {profile['videos_analyzed']} video(s) analyzed",
        "",
        f"🎭 **Mood:** {fp.get('mood', '—')}",
        f"📷 **Motion:** {fp.get('motion_style', '—')}",
        f"💡 **Lighting:** {fp.get('lighting_style', '—')}",
        f"🎨 **Color:** {fp.get('color_temperature', '—')}, {fp.get('saturation', '—')} saturation",
        f"🖼️ **Shot style:** {fp.get('dominant_shot', '—')}",
        f"✨ **Aesthetic:** {', '.join(fp.get('aesthetic_keywords', [])[:6])}",
        f"👤 **Subjects:** {', '.join(fp.get('subjects', [])[:5])}",
        f"📢 **Best for:** {', '.join(fp.get('ad_suitability', []))}",
        "",
        f"*{fp.get('brand_feel', '')}*",
    ]
    return "\n".join(lines)


# ── Tab 1: Upload & Analyze ──────────────────────────────────────────────────────

def handle_upload(video_file, user_id: str, progress=gr.Progress()):
    """Process an uploaded video and update the style profile."""
    if video_file is None:
        return "❌ Please upload a video first.", gr.update()

    user_id = user_id.strip() or "default"
    # Sanitize user_id
    user_id = "".join(c for c in user_id if c.isalnum() or c in "-_")

    try:
        progress(0.1, desc="Extracting frames...")
        profile = analyze_video(video_file, user_id, verbose=False)
        progress(0.9, desc="Updating style profile...")

        n = profile["videos_analyzed"]
        fp = profile.get("style_fingerprint", {})
        summary = (
            f"✅ Video analyzed! Style profile updated.\n\n"
            f"**{n} video(s) in profile**\n"
            f"Mood: {fp.get('mood', '—')} | "
            f"Motion: {fp.get('motion_style', '—')} | "
            f"Lighting: {fp.get('lighting_style', '—')}\n"
            f"Aesthetic: {', '.join(fp.get('aesthetic_keywords', [])[:5])}"
        )
        progress(1.0, desc="Done!")
        return summary, gr.update(value=user_id)

    except Exception as e:
        return f"❌ Error: {str(e)}", gr.update()


# ── Photo Reference Upload ───────────────────────────────────────────────────────

def handle_photo_upload(photo_file, user_id: str, label: str):
    """Analyze a reference photo and store it in the profile."""
    if photo_file is None:
        return "❌ Please upload a photo first."
    if not label.strip():
        return "❌ Please enter a label (e.g. 'carter', 'moon_lamp', 'tote_bag')."

    user_id = (user_id.strip() or "default")
    user_id = "".join(c for c in user_id if c.isalnum() or c in "-_")
    label   = label.strip().lower().replace(" ", "_")

    try:
        from analyzer import analyze_photo
        profile = analyze_photo(photo_file, user_id, label=label, verbose=False)
        refs = profile.get("references", {})
        desc = refs.get(label, {}).get("description", "")
        return (
            f"✅ **Reference '{label}' added!**\n\n"
            f"Claude's description:\n\n*{desc}*\n\n"
            f"This will now be injected into every prompt for **{user_id}**."
        )
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ── Tab 2: Style Profile Viewer ──────────────────────────────────────────────────

def view_profile(user_id: str):
    """Display the style profile for a user."""
    if not user_id:
        return "Select a user profile to view."
    from analyzer import get_references
    out = get_profile_stats(user_id)
    refs = get_references(user_id)
    if refs:
        out += "\n\n**📸 Photo References:**\n"
        for label, r in refs.items():
            out += f"\n- **{label}:** {r['description']}"
    return out


def view_profile_raw(user_id: str):
    """Return raw JSON profile."""
    if not user_id:
        return "{}"
    profile = load_profile(user_id)
    return json.dumps(profile.get("style_fingerprint", {}), indent=2)


# ── Tab 3: Generate ──────────────────────────────────────────────────────────────

def handle_generate(user_id: str, brief: str, backend: str,
                     duration: int, ad_type: str, progress=gr.Progress()):
    """Run the generation pipeline and return the prompt + result."""
    user_id = (user_id or "default").strip()
    brief   = (brief or "").strip()

    if not brief:
        return "❌ Please enter a creative brief.", "", ""

    try:
        progress(0.2, desc="Loading style profile...")
        progress(0.4, desc="Building personalized prompt...")

        result = generate_video(
            user_id  = user_id,
            brief    = brief,
            backend  = backend.lower(),
            duration = duration,
            ad_type  = ad_type,
            verbose  = False,
        )

        progress(1.0, desc="Done!")

        prompt  = result.get("final_prompt", "")
        raw     = result.get("raw_prompt", "")
        job_info = ""

        if result.get("output_url"):
            job_info = f"✅ Video generated!\n{result['output_url']}"
        elif result.get("job"):
            job_info = f"✅ Job submitted to {backend}:\n{json.dumps(result['job'], indent=2)}"
        else:
            job_info = (
                "📋 **Prompt-only mode** — copy this prompt into:\n"
                "• [Kling AI](https://klingai.com) → Text to Video\n"
                "• [Runway](https://runwayml.com) → Gen-3 → Text to Video\n"
                "• [Pika](https://pika.art) → Generate\n"
                "• Any other AI video tool"
            )

        return prompt, raw, job_info

    except Exception as e:
        return f"❌ Error: {str(e)}", "", ""


# ── Tab 4: History ───────────────────────────────────────────────────────────────

def view_history(user_id: str):
    """Show generation history for a user — full prompts."""
    outputs_dir = ROOT / "outputs"
    if not outputs_dir.exists():
        return "No history yet."

    files = sorted(outputs_dir.glob(f"{user_id}_*.json"), reverse=True)[:10]
    if not files:
        return f"No history for user '{user_id}'."

    sections = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        ts     = data.get("generated_at", "")[:16].replace("T", "  ")
        brief  = data.get("brief", "")
        bk     = data.get("backend", "prompt")
        prompt = data.get("final_prompt", "")

        sections.append(
            f"---\n"
            f"**{ts}** | {bk}\n\n"
            f"*Brief: {brief}*\n\n"
            f"```\n{prompt}\n```"
        )

    return "\n\n".join(sections)


# ── Build UI ─────────────────────────────────────────────────────────────────────

def build_ui():
    profiles = list_profiles() or ["default"]

    with gr.Blocks(title="PersonaVid") as demo:

        gr.Markdown("""
# 🎬 PersonaVid
**AI-powered personalized video generation — trained on your footage**
""")

        with gr.Tab("📤 Upload & Analyze"):
            gr.Markdown("""
Upload **videos** to build your visual style fingerprint.
Upload **photos** as named references (Carter, product shots, settings) — these get injected into every prompt so the AI knows exactly what to recreate.
""")
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Tab("🎬 Video"):
                        video_input = gr.Video(label="Upload Video", sources=["upload"])
                        user_id_input = gr.Textbox(
                            label="Brand / User ID",
                            placeholder="e.g. continuum_goods",
                            value="default",
                        )
                        analyze_btn = gr.Button("🔍 Analyze Video", variant="primary")

                    with gr.Tab("📸 Photo Reference"):
                        photo_input = gr.Image(
                            label="Upload Photo (Carter, product, setting)",
                            type="filepath",
                            sources=["upload"],
                        )
                        photo_user_id = gr.Textbox(
                            label="Brand / User ID",
                            value=profiles[0] if profiles else "default",
                        )
                        photo_label = gr.Textbox(
                            label="Reference Label",
                            placeholder="e.g. carter, moon_lamp, tote_bag, kitchen",
                        )
                        photo_btn = gr.Button("📸 Add Photo Reference", variant="primary")

                with gr.Column(scale=2):
                    analyze_output = gr.Markdown(
                        value="Upload a video or photo reference to get started."
                    )
                    user_id_sync = gr.Textbox(visible=False)

            analyze_btn.click(
                fn=handle_upload,
                inputs=[video_input, user_id_input],
                outputs=[analyze_output, user_id_sync],
            )
            photo_btn.click(
                fn=handle_photo_upload,
                inputs=[photo_input, photo_user_id, photo_label],
                outputs=[analyze_output],
            )

        with gr.Tab("🎨 Style Profile"):
            gr.Markdown("""
View the visual style fingerprint built from your uploaded videos.
This is what drives the personalization of your generated content.
""")
            with gr.Row():
                with gr.Column(scale=1):
                    profile_select = gr.Dropdown(
                        choices=list_profiles() or ["default"],
                        label="Select Profile",
                        value=profiles[0] if profiles else "default",
                    )
                    refresh_btn = gr.Button("🔄 Refresh Profiles")
                with gr.Column(scale=3):
                    default_profile = view_profile(profiles[0]) if profiles else "No profiles yet."
                    default_json    = view_profile_raw(profiles[0]) if profiles else "{}"
                    profile_view = gr.Markdown(value=default_profile)
                    profile_json = gr.Code(
                        label="Raw Style Fingerprint (JSON)",
                        language="json",
                        value=default_json,
                        visible=True,
                    )

            profile_select.change(
                fn=lambda u: (view_profile(u), view_profile_raw(u)),
                inputs=[profile_select],
                outputs=[profile_view, profile_json],
            )
            refresh_btn.click(
                fn=lambda: gr.update(choices=list_profiles()),
                outputs=[profile_select],
            )

        with gr.Tab("🚀 Generate"):
            gr.Markdown("""
Generate a video prompt personalized to your style.
Use **Prompt Only** mode to copy it into Kling AI, Runway, or Pika.
Connect your API keys to generate automatically.
""")
            with gr.Row():
                with gr.Column(scale=1):
                    gen_user = gr.Dropdown(
                        choices=list_profiles() or ["default"],
                        label="Style Profile",
                        value=profiles[0] if profiles else "default",
                    )
                    gen_brief = gr.Textbox(
                        label="Creative Brief",
                        placeholder="e.g. 30-second ad showcasing our kitchen knife set, home kitchen setting",
                        lines=3,
                    )
                    with gr.Row():
                        gen_backend = gr.Dropdown(
                            choices=["Prompt", "Kling", "Runway"],
                            label="Backend",
                            value="Prompt",
                        )
                        gen_duration = gr.Slider(
                            minimum=5, maximum=60, step=5, value=10,
                            label="Duration (seconds)",
                        )
                    gen_ad_type = gr.Dropdown(
                        choices=["product", "lifestyle", "tutorial", "testimonial"],
                        label="Ad Type",
                        value="product",
                    )
                    gen_btn = gr.Button("✨ Generate Prompt", variant="primary")

                with gr.Column(scale=2):
                    gen_prompt_out = gr.Textbox(
                        label="Final Prompt (copy this into your AI video tool)",
                        lines=10,
                    )
                    gen_job_out = gr.Markdown(label="Generation Status")
                    with gr.Accordion("Raw Prompt (pre-Claude refinement)", open=False):
                        gen_raw_out = gr.Textbox(lines=6)

            gen_btn.click(
                fn=handle_generate,
                inputs=[gen_user, gen_brief, gen_backend, gen_duration, gen_ad_type],
                outputs=[gen_prompt_out, gen_raw_out, gen_job_out],
            )

        with gr.Tab("📋 History"):
            hist_user = gr.Dropdown(
                choices=list_profiles() or ["default"],
                label="User",
                value=profiles[0] if profiles else "default",
            )
            default_hist = view_history(profiles[0]) if profiles else "No history yet."
            hist_out = gr.Markdown(value=default_hist)
            hist_user.change(
                fn=view_history,
                inputs=[hist_user],
                outputs=[hist_out],
            )

        # ── Tab 5: Pipeline ──────────────────────────────────────────────────────
        with gr.Tab("🚀 Pipeline"):
            gr.Markdown("""
**Full auto pipeline** — brief → Runway video → auto-edit → approval queue.
Requires Runway API key. Takes 2–4 minutes end to end.
""")
            with gr.Row():
                with gr.Column(scale=1):
                    pipe_user = gr.Dropdown(
                        choices=list_profiles() or ["default"],
                        label="Style Profile",
                        value=profiles[0] if profiles else "default",
                    )
                    pipe_brief = gr.Textbox(
                        label="Creative Brief",
                        placeholder="Cash the cat sitting near the glowing moon lamp, cozy home at night",
                        lines=3,
                    )
                    pipe_hook = gr.Textbox(
                        label="Hook Text (top overlay)",
                        placeholder="he claimed it as his",
                    )
                    pipe_cta = gr.Textbox(
                        label="CTA Text (bottom overlay)",
                        placeholder="Moon Lamp — link in bio",
                    )
                    pipe_platform = gr.Dropdown(
                        choices=["tiktok", "instagram", "pinterest"],
                        label="Platform",
                        value="tiktok",
                    )
                    pipe_image = gr.Image(
                        label="Source Image (optional — animates from this frame)",
                        type="filepath",
                        sources=["upload"],
                    )
                    pipe_btn = gr.Button("Run Pipeline", variant="primary")

                with gr.Column(scale=2):
                    pipe_status = gr.Markdown("Press Run Pipeline to start.")

            def handle_pipeline(user_id, brief, hook, cta, platform, img, progress=gr.Progress()):
                if not brief.strip():
                    return "Please enter a creative brief."
                try:
                    progress(0.1, desc="Building prompt...")
                    from pipeline import run_pipeline
                    progress(0.3, desc="Submitting to Runway...")
                    state = run_pipeline(
                        user_id=user_id or "continuum_goods",
                        brief=brief,
                        platform=platform,
                        hook_text=hook or "",
                        cta_text=cta or "",
                        source_image=img,
                    )
                    progress(1.0, desc="Done!")
                    if state.get("error"):
                        return f"Pipeline error: {state['error']}"
                    return (
                        f"Video ready for review!\n\n"
                        f"**Job ID:** `{state['job_id']}`\n"
                        f"**Caption:** {state.get('caption', '')}\n\n"
                        f"Go to the **Approval** tab to post, reject, or schedule."
                    )
                except Exception as e:
                    return f"Error: {e}"

            pipe_btn.click(
                fn=handle_pipeline,
                inputs=[pipe_user, pipe_brief, pipe_hook, pipe_cta, pipe_platform, pipe_image],
                outputs=[pipe_status],
            )

        # ── Tab 6: Approval ──────────────────────────────────────────────────────
        with gr.Tab("Approval"):
            gr.Markdown("""
**Approve, reject, or post videos before they go live.**
Each video has a 15-minute window. Post Now sends immediately.
Let Post approves and auto-publishes when the window closes.
""")
            with gr.Row():
                with gr.Column(scale=1):
                    approval_job_id = gr.Textbox(label="Job ID", placeholder="tiktok_20260417_...")
                    with gr.Row():
                        post_now_btn  = gr.Button("Post Now", variant="primary")
                        let_post_btn  = gr.Button("Let Post", variant="secondary")
                        reject_btn    = gr.Button("Reject", variant="stop")
                    refresh_btn = gr.Button("Refresh Queue")
                with gr.Column(scale=2):
                    approval_queue_display = gr.Markdown(value="Press Refresh Queue to load.")
                    approval_result = gr.Markdown()

            def handle_approval_action(action: str, job_id: str):
                from approval import post_now, let_post, reject, format_queue_display
                if not job_id.strip():
                    return "Please enter a Job ID.", format_queue_display()
                if action == "post_now":
                    result = post_now(job_id.strip())
                elif action == "let_post":
                    result = let_post(job_id.strip())
                elif action == "reject":
                    result = reject(job_id.strip())
                else:
                    result = "Unknown action."
                return result, format_queue_display()

            def refresh_queue():
                from approval import format_queue_display
                return format_queue_display()

            post_now_btn.click(
                fn=lambda jid: handle_approval_action("post_now", jid),
                inputs=[approval_job_id],
                outputs=[approval_result, approval_queue_display],
            )
            let_post_btn.click(
                fn=lambda jid: handle_approval_action("let_post", jid),
                inputs=[approval_job_id],
                outputs=[approval_result, approval_queue_display],
            )
            reject_btn.click(
                fn=lambda jid: handle_approval_action("reject", jid),
                inputs=[approval_job_id],
                outputs=[approval_result, approval_queue_display],
            )
            refresh_btn.click(
                fn=refresh_queue,
                outputs=[approval_queue_display],
            )

        gr.Markdown("""
---
**PersonaVid** — Built with Claude Vision + Runway Gen-4.5 | [Continuum Goods](https://continuumgoods.com)
""")

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
