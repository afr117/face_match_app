### `face_match_app/README.md`
```markdown
# Face Match App (Python + OpenCV)

![App screenshot](docs/screenshot-face-match.png)

An experiment in face recognition against a small gallery of known identities. It loads encodings from `models/known_faces/` and flags matches from a webcam or video file with simple, readable code.

> **Ethics & compliance:** Always respect local laws and privacy. Obtain consent where required, disclose usage, and design for opt-in.

## Why this exists (the story)
I needed a minimal, hackable baseline to test thresholding, lighting variations, and simple alerting without pulling in heavyweight frameworks. This repo is my clean sandbox for controlled experiments.

## Features (implemented)
- Load known faces from `models/known_faces/<person>/*.jpg`
- Auto-build face encodings
- Live webcam or video file inference
- On-screen labels with distance thresholding
- Simple logging of positive matches

## Roadmap (next)
- Async frame queue for smoother FPS
- Configurable alerts (sound, webhook/email)
- GPU acceleration toggle
- Re-identification to avoid duplicate alerts
- Multi-frame consensus to reduce false positives

## Quickstart
```bash
git clone https://github.com/afr117/face_match_app.git
cd face_match_app
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt

# Put reference images in:
# models/known_faces/<person_name>/*.jpg

# Run (webcam)
python main.py

# Or run on a file
python main.py --video path/to/file.mp4
Project layout
css
Copy
Edit
models/
  known_faces/
main.py
requirements.txt
Hardest challenges
Robustness to lighting, pose, occlusion

Choosing a meaningful distance threshold

Packaging native deps consistently across OSes

Developer
I’m Alfred Figueroa, ML-leaning full-stack dev focused on useful automation.

LinkedIn: https://www.linkedin.com/in/alfred-figueroa-rosado-10b010208

Twitter/X: https://twitter.com/your_handle

Portfolio Project repo: https://github.com/afr117/portfolio
