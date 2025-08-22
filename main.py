#!/usr/bin/env python3
"""
Run with:
  python main.py                 # webcam 0 (visible window)
  python main.py --video "rtsp://user:pa%40ss@IP:554/stream"  # RTSP
Press 'q' in the video window to quit.

Put gallery photos in:
  known_faces/<PersonName>/*.jpg
"""

import os
import time
import argparse
import pathlib
from datetime import datetime

import cv2
from deepface import DeepFace

# ---------- Simple config ----------
DB_DIR = "known_faces"     # your gallery: known_faces/Person/*.jpg
EVENTS_DIR = "events"      # where snapshots of matches are saved
MODEL = "ArcFace"          # good accuracy/speed balance
DETECTOR = "opencv"        # 'opencv' is light & reliable on Windows; try 'retinaface' later
THRESH = 0.35              # lower=stricter (typical 0.30–0.45)
SAMPLE_EVERY_SEC = 1.0     # run matching roughly once per second
# -----------------------------------

os.makedirs(EVENTS_DIR, exist_ok=True)


def percent_from_distance(d: float) -> float:
    """
    Convert model distance to a friendly 'percent match' (similarity proxy).
    For ArcFace distances in DeepFace: smaller is better. We map:
      percent = clamp((1 - distance) * 100, 0..100)
    NOTE: This is NOT a calibrated probability; use it as a confidence meter.
    """
    sim = 1.0 - float(d)
    if sim < 0.0:
        sim = 0.0
    if sim > 1.0:
        sim = 1.0
    return sim * 100.0


def best_match(frame_bgr, thresh=THRESH):
    """
    Returns (label, distance, identity_path, percent) if a match passes threshold,
    else (None, None, None, None). If no face found, returns (None, None, None, None).
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    try:
        dfs = DeepFace.find(
            img_path=frame_rgb,
            db_path=DB_DIR,
            model_name=MODEL,
            detector_backend=DETECTOR,
            enforce_detection=False,
            silent=True,
        )
    except Exception as e:
        print(f"[DeepFace] error: {e}")
        return None, None, None, None

    if not dfs or len(dfs[0]) == 0:
        return None, None, None, None

    row = dfs[0].iloc[0]
    dist = float(row["distance"])
    if dist > thresh:
        return None, None, None, None

    identity_path = row["identity"]
    label = pathlib.Path(identity_path).parent.name
    pct = percent_from_distance(dist)
    return label, dist, identity_path, pct


def save_snapshot(frame_bgr, label: str, dist: float, pct: float, tag: str):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"{ts}_{tag}_{label}_{pct:.0f}pct_d{dist:.3f}.jpg"
    path = os.path.join(EVENTS_DIR, fname)
    cv2.imwrite(path, frame_bgr)
    return path


def open_capture(source):
    """
    Try to open the requested source.
    If it fails and source isn't the webcam (0), fall back to webcam 0.
    """
    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        return cap, str(source)

    # Fallback to webcam 0
    if source != 0:
        print(f"[WARN] Cannot open '{source}'. Falling back to webcam 0.")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            return cap, "0"

    raise RuntimeError(f"Cannot open video source: {source}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--video",
        help="Leave empty for webcam 0. Or provide RTSP/HTTP/file path.",
    )
    ap.add_argument(
        "--thresh",
        type=float,
        default=THRESH,
        help="Match threshold (lower=stricter). Default: 0.35",
    )
    args = ap.parse_args()

    source = args.video if args.video else 0
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    print("[INFO] Starting…")
    print(f"[INFO] Source: {source if source != 0 else 'webcam 0 (default)'}")
    print("[TIP] If your RTSP password has '@', encode it as '%40' in the URL.")
    print("[TIP] Press 'q' in the video window to quit.")

    try:
        cap, tag = open_capture(source)
    except RuntimeError as e:
        print(str(e))
        return

    last = 0.0
    win_name = f"Camera: {tag}"

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Frame read failed; retrying in 0.5s…")
            time.sleep(0.5)
            continue

        # Always show the camera (visible, not background)
        cv2.imshow(win_name, frame)

        # Sample periodically for matching
        now = time.time()
        if now - last >= SAMPLE_EVERY_SEC:
            last = now
            who, d, ident, pct = best_match(frame, thresh=args.thresh)
            if who:
                print(f"[MATCH] {who}  {pct:.1f}%  (distance={d:.3f})")
                save_snapshot(frame, who, d, pct, tag)
            else:
                # Optional: print a brief heartbeat so you see it's running
                print(f"[INFO] No match (thr={args.thresh:.2f})")

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
