# PersonaVid

An AI video personalization pipeline that turns a library of raw footage and a product brief into finished, platform-ready short videos, end to end.

PersonaVid was built to answer a practical question: can one person run a full short-form video operation, from footage intake to a published clip, without a production team? The system handles cataloging, shot planning, generation, compositing, scoring, and publishing as a connected pipeline rather than a pile of manual steps.

## What it does

- **Footage catalog** (`footage_catalog.py`, `auto_cataloger.py`) — ingests raw clips, analyzes them, and builds a searchable catalog so the right shot can be pulled by query instead of by scrubbing.
- **Shot direction** (`director.py`, `brain.py`) — an agent brain plans shots from a product brief and a target platform, deciding what footage and generated content the final piece needs.
- **Generation** (`generator.py`, `modal_video.py`, `modal_extract_batch.py`) — runs diffusion video generation on Modal GPU containers, so the heavy compute stays off the local machine and scales on demand.
- **Compositing** (`compositor.py`, `video_editor.py`, `backgrounds.py`) — assembles generated and real footage, handles backgrounds, and cuts the final edit.
- **Scoring and approval** (`scorer.py`, `analyzer.py`, `approval.py`) — grades candidate outputs against quality criteria before anything ships.
- **Publishing** (`publisher.py`, `performance_reader.py`) — exports finished clips for scheduling (TikTok via an export folder, Instagram, Pinterest) and reads back performance.

## Design notes

The pipeline is profile-driven (`profiles/`): a single brand profile carries the voice, references, and product context, so the same machinery can serve different brands without code changes. The agent brain (`intelligence/personavid_brain.md`) is an editable text file, not hardcoded logic, so behavior can be tuned without touching Python.

## Stack

Python, Modal (serverless GPU), diffusion-based video generation, a modular agent pipeline. Model weights, generated media, and footage data are excluded from the repo by design; this is the pipeline code.

## Status

Built April to August 2026 and paused since. Text-to-video and image-to-video generation, scoring and export
worked end to end; TikTok's API is closed to single accounts, so publishing went through an export folder for a
scheduling tool instead. Built by Thomas Semrad.
