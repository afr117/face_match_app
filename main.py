#!/usr/bin/env python3
import cv2, os, time, argparse, pathlib
from datetime import datetime
from deepface import DeepFace

DB_DIR = "known_faces"       # gallery folder
EVENTS_DIR = "events"        # snapshot output
MODEL = "ArcFace"            # fast + strong
DETECTOR = "retinaface"      # good default; set "opencv" if you get detector errors
THRESH = 0.35                # lower = stricter; ~0.3–0.5 typical for ArcFace
SAMPLE_EVERY_SEC = 1.0       # check roughly once per second

os.makedirs(EVENTS_DIR, exist_ok=True)

def best_match(frame_bgr):
    """Return (label, distance, identity_path) or (None, None, None)."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # DeepFace.find searches a folder and returns top matches with distances
    try:
        dfs = DeepFace.find(
            img_path=frame_rgb,
            db_path=DB_DIR,
            model_name=MODEL,
            detector_backend=DETECTOR,
            enforce_detection=False,
            silent=True
        )
    except Exception as e:
        print(f"[DeepFace] error: {e}")
        return None, None, None

    if not dfs or len(dfs[0]) == 0:
        return None, None, None

    row = dfs[0].iloc[0]  # best match
    dist = float(row["distance"])
    if dist > THRESH:
        return None, None, None

    identity_path = row["identity"]
    # label = folder name above the matched image
    label = pathlib.Path(identity_path).parent.name
    return label, dist, identity_path

def save_snapshot(frame_bgr, label, dist):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(EVENTS_DIR, f"{ts}_{label}_{dist:.3f}.jpg")
    cv2.imwrite(path, frame_bgr)
    return path

def open_source(src):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {src}")
    return cap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="RTSP URL, HTTP URL, file path, or leave empty for webcam 0")
    ap.add_argument("--thresh", type=float, default=THRESH, help="match threshold (lower=stricter)")
    args = ap.parse_args()

    global THRESH
    THRESH = args.thresh

    src = args.video if args.video else 0  # default webcam
    print(f"[INFO] Opening source: {src}")
    print("[TIP] If your password has '@', encode it as '%40' in the RTSP URL.")
    cap = open_source(src)

    last = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Frame read failed; retrying in 0.5s...")
            time.sleep(0.5)
            continue

        now = time.time()
        if now - last >= SAMPLE_EVERY_SEC:
            last = now
            who, d, ident = best_match(frame)
            if who:
                snap = save_snapshot(frame, who, d)
                print(f"[MATCH] {who} (distance={d:.3f})  -> saved {snap}")

        cv2.imshow("Feed (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
