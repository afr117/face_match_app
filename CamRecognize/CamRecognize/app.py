import streamlit as st
import cv2
import json
import os
import numpy as np
from PIL import Image
import datetime
import time
import threading
from camera_manager import CameraManager
from face_recognition_engine import FaceRecognitionEngine
from notification_service import NotificationService

# Import streamlit-webrtc components
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import queue

# ---- Optional YOLO (ultralytics) import --------------------------------------
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO

    class YOLODetector:
        """
        Thin wrapper over Ultralytics YOLO for person/object detection.
        Default class filter: person only (COCO id = 0).
        """
        def __init__(self, model_name="yolov8s.pt", conf=0.35, imgsz=960, classes=None, iou=0.5):
            self.model = YOLO(model_name)
            self.conf = float(conf)
            self.imgsz = int(imgsz)
            self.iou = float(iou)
            # classes: None for all, [0] for person-only
            self.classes = classes
            try:
                self.names = self.model.model.names  # newer
            except Exception:
                self.names = self.model.names

        def predict(self, frame_bgr):
            """
            Return list of detections:
            [{'label': str, 'conf': float, 'box': (x1,y1,x2,y2), 'cls_id': int}]
            """
            if frame_bgr is None or frame_bgr.size == 0:
                return []
            results = self.model.predict(
                source=frame_bgr,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                classes=self.classes,
                verbose=False
            )
            detections = []
            if not results:
                return detections
            r0 = results[0]
            if not hasattr(r0, "boxes") or r0.boxes is None:
                return detections
            for b in r0.boxes:
                xyxy = b.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, xyxy)
                conf = float(b.conf[0].item()) if hasattr(b, "conf") else 0.0
                cls_id = int(b.cls[0].item()) if hasattr(b, "cls") else -1
                label = (
                    self.names.get(cls_id, str(cls_id)) if isinstance(self.names, dict)
                    else (self.names[cls_id] if 0 <= cls_id < len(self.names) else str(cls_id))
                )
                detections.append({
                    "label": label,
                    "conf": conf,
                    "box": (x1, y1, x2, y2),
                    "cls_id": cls_id
                })
            return detections

    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False
# -----------------------------------------------------------------------------

# Initialize session state
if 'camera_manager' not in st.session_state:
    st.session_state.camera_manager = CameraManager()
if 'face_engine' not in st.session_state:
    st.session_state.face_engine = FaceRecognitionEngine()
if 'notification_service' not in st.session_state:
    st.session_state.notification_service = NotificationService()
if 'monitoring_active' not in st.session_state:
    st.session_state.monitoring_active = False
if 'last_notifications' not in st.session_state:
    st.session_state.last_notifications = {}
if 'camera_status_cache' not in st.session_state:
    st.session_state.camera_status_cache = {}
if 'cache_timestamps' not in st.session_state:
    st.session_state.cache_timestamps = {}

# Detection backend + YOLO params in state (with defaults)
if 'detector_backend' not in st.session_state:
    st.session_state.detector_backend = "Haar (basic)"  # or "YOLO (recommended)"
if 'yolo_conf' not in st.session_state:
    st.session_state.yolo_conf = 0.35
if 'yolo_imgsz' not in st.session_state:
    st.session_state.yolo_imgsz = 960
if 'yolo_person_only' not in st.session_state:
    st.session_state.yolo_person_only = True
if 'yolo_model_name' not in st.session_state:
    st.session_state.yolo_model_name = "yolov8s.pt"  # good balance; try 'yolov8n.pt' for speed, 'yolov8m.pt' for range

# YOLO detector cache (re-init when settings change)
def _maybe_init_yolo():
    if not YOLO_AVAILABLE:
        return None
    classes = [0] if st.session_state.yolo_person_only else None  # person only
    return YOLODetector(
        model_name=st.session_state.yolo_model_name,
        conf=st.session_state.yolo_conf,
        imgsz=st.session_state.yolo_imgsz,
        classes=classes,
        iou=0.5
    )

if 'yolo_detector' not in st.session_state:
    st.session_state.yolo_detector = _maybe_init_yolo()

def get_cached_camera_status(camera_url: str) -> bool:
    """Get camera status with 30-second caching to reduce load"""
    import time
    cache_key = f"status_{camera_url}"
    current_time = time.time()
    if (cache_key in st.session_state.camera_status_cache and
        cache_key in st.session_state.cache_timestamps and
        current_time - st.session_state.cache_timestamps[cache_key] < 30):
        return st.session_state.camera_status_cache[cache_key]
    status = st.session_state.camera_manager.check_camera_status(camera_url)
    st.session_state.camera_status_cache[cache_key] = status
    st.session_state.cache_timestamps[cache_key] = current_time
    return status

def should_send_notification(match_name: str, camera_name: str) -> bool:
    """Check if notification should be sent (debouncing)"""
    import time
    notification_key = f"{match_name}_{camera_name}"
    current_time = time.time()
    if (notification_key in st.session_state.last_notifications and
        current_time - st.session_state.last_notifications[notification_key] < 60):
        return False
    st.session_state.last_notifications[notification_key] = current_time
    return True

# WebRTC Video Transformer for face recognition + YOLO overlay
class FaceRecognitionTransformer:
    def __init__(self, face_engine, notification_service, camera_name):
        self.face_engine = face_engine
        self.notification_service = notification_service
        self.camera_name = camera_name
        self.last_detection_time = 0
        self.detection_interval = 3  # Run face detection every 3 seconds
        self.latest_matches = []
        self.frame_count = 0

        # YOLO timing: run a bit more often for smoother boxes
        self.yolo_last_time = 0
        self.yolo_interval = 0.5  # seconds
        self.last_yolo_dets = []

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        current_time = time.time()
        self.frame_count += 1

        # 1) YOLO overlay (if enabled & available) — helps long-range visibility
        if st.session_state.detector_backend.startswith("YOLO") and st.session_state.yolo_detector is not None:
            if (current_time - self.yolo_last_time) > self.yolo_interval:
                try:
                    self.last_yolo_dets = st.session_state.yolo_detector.predict(img)
                except Exception as e:
                    # If YOLO errors at runtime, don't crash the stream; just fallback silently
                    self.last_yolo_dets = []
                self.yolo_last_time = current_time

            # Draw YOLO boxes
            for d in self.last_yolo_dets:
                (x1, y1, x2, y2) = d["box"]
                label = f'{d["label"]} {d["conf"]:.2f}'
                # Orange-ish for general detection boxes (distinct from your face colors)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 180, 255), 2)
                cv2.putText(img, label, (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)

        # 2) Face recognition on interval (your original flow)
        if current_time - self.last_detection_time > self.detection_interval:
            try:
                matches = self.face_engine.detect_faces_in_frame(img, self.camera_name)
                self.latest_matches = matches
                self.last_detection_time = current_time

                # Send notifications for matches
                for match in matches:
                    if match.get('name') and match['name'] != 'No Match':
                        if should_send_notification(match['name'], self.camera_name):
                            self.notification_service.send_notification(
                                f"Live Stream Alert: {match['name']} detected with {match['percentage']:.1f}% confidence"
                            )
            except Exception as e:
                print(f"Face detection error: {e}")

        # 3) Draw your face-recognition overlays (unchanged, just cleaner)
        if self.latest_matches:
            for match in self.latest_matches:
                if 'face_coordinates' in match:
                    x, y, w, h = match['face_coordinates']
                    color = (0, 255, 0) if match['name'] != 'No Match' else (0, 0, 255)
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 4)

                    name_label = f"{match['name']}"
                    percentage_label = f"{match['percentage']:.1f}%"

                    font = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 1.2
                    thickness = 2
                    (name_w, name_h), _ = cv2.getTextSize(name_label, font, font_scale, thickness)
                    (perc_w, perc_h), _ = cv2.getTextSize(percentage_label, font, font_scale + 0.3, thickness + 1)

                    overlay = img.copy()
                    cv2.rectangle(overlay, (x, y - name_h - 35), (x + max(name_w, perc_w) + 10, y - 5), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

                    cv2.putText(img, name_label, (x + 5, y - perc_h - 15), font, font_scale, (255, 255, 255), thickness)
                    percentage_color = (0, 255, 0) if match['percentage'] >= 60 else (0, 165, 255)
                    cv2.putText(img, percentage_label, (x + 5, y - 5), font, font_scale + 0.3, percentage_color, thickness + 1)

        # 4) Frame counter + status
        cv2.putText(img, f"Frame: {self.frame_count}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if (current_time - self.last_detection_time) < 1:
            status_text = "ANALYZING..."
            cv2.putText(img, status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.set_page_config(
        page_title="Camera Monitoring System",
        page_icon="📹",
        layout="wide"
    )

    st.title("📹 Camera Monitoring System with Face Recognition")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Dashboard", "Camera Management", "Camera Feeds", "Face References", "Live Monitoring", "Settings"]
    )

    if page == "Dashboard":
        show_dashboard()
    elif page == "Camera Management":
        show_camera_management()
    elif page == "Camera Feeds":
        show_camera_feeds()
    elif page == "Face References":
        show_face_references()
    elif page == "Live Monitoring":
        show_live_monitoring()
    elif page == "Settings":
        show_settings()

def show_dashboard():
    st.header("📊 Dashboard")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("System Overview")
    with col2:
        if st.button("🔄 Refresh Status"):
            st.rerun()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active Cameras", len(st.session_state.camera_manager.get_cameras()))
    with col2:
        st.metric("Reference Faces", len(st.session_state.face_engine.get_reference_faces()))
    with col3:
        detection_count = len(st.session_state.face_engine.get_recent_detections())
        st.metric("Recent Detections", detection_count)

    st.subheader("Camera Status")
    cameras = st.session_state.camera_manager.get_cameras()
    if cameras:
        for camera in cameras:
            with st.expander(f"📹 {camera['name']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**URL:** {camera['url']}")
                    st.write(f"**Added:** {camera['added_date']}")
                with col2:
                    status = get_cached_camera_status(camera['url'])
                    if status:
                        st.success("✅ Online")
                    else:
                        st.error("❌ Offline")
    else:
        st.info("No cameras configured. Add cameras in the Camera Management section.")

    st.subheader("Recent Detections")
    detections = st.session_state.face_engine.get_recent_detections()
    if detections:
        for detection in detections[-10:]:
            with st.expander(f"Detection at {detection['timestamp']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Camera:** {detection['camera_name']}")
                    st.write(f"**Match:** {detection['matched_face']}")
                    st.write(f"**Confidence:** {detection['confidence']:.2f}")
                with col2:
                    if os.path.exists(detection['image_path']):
                        st.image(detection['image_path'], width=200)
    else:
        st.info("No recent detections.")

def show_camera_management():
    st.header("⚙️ Camera Management")

    st.subheader("Add New Camera")

    st.info("💻 **Quick Setup:** Add your laptop camera here, then use it in Live Monitoring (it will work through your browser)")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎥 Add Laptop Camera (Index 0)", type="primary"):
            success = st.session_state.camera_manager.add_camera("Laptop Camera", "0")
            if success:
                st.success("✅ Laptop camera added! Now go to 'Live Monitoring' to start streaming.")
                st.rerun()
            else:
                st.error("Camera name already exists. Try removing it first or use a different name.")
    with col2:
        if st.button("🎥 Try Camera Index 1"):
            success = st.session_state.camera_manager.add_camera("Camera Index 1", "1")
            if success:
                st.success("Camera added successfully!")
                st.rerun()
            else:
                st.error("Camera not found at index 1.")

    st.markdown("---")
    st.subheader("Manual Camera Setup")
    with st.form("add_camera_form"):
        camera_name = st.text_input("Camera Name")
        camera_type = st.radio(
            "Camera Type",
            ["Laptop/USB Webcam (use camera index: 0, 1, 2, etc.)", "Network Camera (HTTP/HTTPS URL)"]
        )
        if "Laptop/USB" in camera_type:
            camera_url = st.text_input("Camera Index (0 for built-in, 1 for USB, etc.)", value="0")
            st.caption("Common values: 0 = built-in laptop camera, 1 = first USB camera")
        else:
            camera_url = st.text_input("Camera URL (HTTP/HTTPS stream)")
            st.caption("Example: http://192.168.1.100:8080/video")

        submitted = st.form_submit_button("Add Camera")
        if submitted:
            if camera_name and camera_url:
                success = st.session_state.camera_manager.add_camera(camera_name, camera_url)
                if success:
                    st.success(f"✅ Camera '{camera_name}' added successfully!")
                    st.rerun()
                else:
                    if camera_url.isdigit():
                        st.error(f"Camera with this name or index already exists. Try a different name.")
                    else:
                        st.error("Failed to add camera. Either the name already exists or the network camera URL is not accessible.")
            else:
                st.error("Please fill in all fields.")

    st.subheader("Existing Cameras")
    cameras = st.session_state.camera_manager.get_cameras()
    if cameras:
        for i, camera in enumerate(cameras):
            with st.expander(f"📹 {camera['name']}"):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"**URL:** {camera['url']}")
                    st.write(f"**Added:** {camera['added_date']}")
                with col2:
                    if st.button("Test", key=f"test_{i}"):
                        status = st.session_state.camera_manager.check_camera_status(camera['url'])
                        if status:
                            st.success("✅ Working")
                        else:
                            st.error("❌ Failed")
                with col3:
                    if st.button("Remove", key=f"remove_{i}"):
                        st.session_state.camera_manager.remove_camera(i)
                        st.success("Camera removed!")
                        st.rerun()
    else:
        st.info("No cameras configured.")

def show_camera_feeds():
    st.header("📷 Camera Feed Snapshots")

    cameras = st.session_state.camera_manager.get_cameras()
    if not cameras:
        st.info("No cameras configured. Add cameras in Camera Management.")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"Camera Snapshots ({len(cameras)})")
        st.caption("This page shows snapshot previews. For live streaming, use the Live Monitoring page.")
    with col2:
        refresh_rate = st.selectbox("Update Rate", ["Manual only", "5 seconds", "10 seconds", "30 seconds"], index=0)
        if refresh_rate != "Manual only":
            refresh_seconds = int(refresh_rate.split()[0])
            if 'feeds_last_refresh' not in st.session_state:
                st.session_state.feeds_last_refresh = time.time()
            if time.time() - st.session_state.feeds_last_refresh > refresh_seconds:
                st.session_state.feeds_last_refresh = time.time()
                st.rerun()
        if st.button("🔄 Update All Snapshots"):
            st.rerun()

    for i, camera in enumerate(cameras):
        st.subheader(f"📹 {camera['name']}")
        status = get_cached_camera_status(camera['url'])
        col1, col2 = st.columns([3, 1])

        with col1:
            if status:
                frame = st.session_state.camera_manager.capture_frame(camera['url'])
                if frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    caption_text = f"Snapshot - {camera['name']}"
                    if refresh_rate != "Manual only":
                        caption_text += f" (Auto-updates every {refresh_rate.lower()})"
                    st.image(frame_rgb, caption=caption_text, use_column_width=True)

                    with st.expander("🔍 Face Detection Analysis", expanded=False):
                        if st.button(f"Analyze This Snapshot", key=f"analyze_{i}"):
                            with st.spinner("Analyzing snapshot for faces..."):
                                matches = st.session_state.face_engine.detect_faces_in_frame(frame, camera['name'])
                                if matches:
                                    valid_matches = [m for m in matches if m['name'] != 'No Match']
                                    if valid_matches:
                                        st.success(f"✅ Found {len(valid_matches)} known face(s)!")
                                        for match in valid_matches:
                                            st.write(f"🎯 **{match['name']}** - {match['percentage']:.1f}% confidence")
                                    else:
                                        st.info("👤 Face detected but no matches above threshold")
                                else:
                                    st.info("No faces detected in this snapshot")
                        st.caption("💡 For continuous face recognition monitoring, use the 'Live Monitoring' page")
                else:
                    st.error("❌ Failed to capture frame from camera")
            else:
                st.error(f"❌ Camera '{camera['name']}' is offline")
                st.image("https://via.placeholder.com/640x480?text=Camera+Offline", use_column_width=True)

        with col2:
            if status:
                st.success("🟢 Online")
            else:
                st.error("🔴 Offline")

            st.write(f"**Added:** {camera['added_date'][:10]}")
            st.write(f"**URL:** {camera['url'][:25]}..." if len(camera['url']) > 25 else f"**URL:** {camera['url']}")

            if st.button("🔧 Test Connection", key=f"test_{i}"):
                with st.spinner("Testing..."):
                    new_status = st.session_state.camera_manager.check_camera_status(camera['url'])
                    st.success("✅ Working" if new_status else "❌ Failed")

            if st.button("📷 Save Snapshot", key=f"save_{i}"):
                if status:
                    frame = st.session_state.camera_manager.capture_frame(camera['url'])
                    if frame is not None:
                        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"snapshot_{camera['name'].replace(' ', '_')}_{timestamp}.jpg"
                        filepath = os.path.join("data/detections", filename)
                        os.makedirs("data/detections", exist_ok=True)
                        cv2.imwrite(filepath, frame)
                        st.success(f"Saved: {filename}")
                    else:
                        st.error("Capture failed")
                else:
                    st.error("Camera offline")

            if st.button(f"🎥 Start Live Stream", key=f"monitor_{i}", type="primary"):
                st.info(f"Go to 'Live Monitoring' page to start live streaming (webcam mode currently available)")

        st.markdown("---")

    with st.expander("📖 Camera Feeds vs Live Monitoring - What's the Difference?", expanded=False):
        st.write("""
        **Camera Feeds Page**:
        - Snapshots for multiple cameras (manual/timed refresh)
        - On-demand face analysis (CPU-friendly)

        **Live Monitoring Page**:
        - Continuous live video (WebRTC webcam today)
        - Real-time face recognition and YOLO overlays
        - Alerts for matches
        """)

    if not st.session_state.camera_manager.get_cameras():
        st.info("💡 **Get Started**: Add cameras in 'Camera Management', then return here to view snapshots")
    elif not st.session_state.face_engine.get_reference_faces():
        st.info("💡 **Enable Face Recognition**: Upload reference faces in 'Face References' to identify people in camera feeds")

def show_face_references():
    st.header("👥 Face References")

    st.subheader("Upload Reference Face")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of the face you want to recognize"
    )

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=300)
        with col2:
            face_name = st.text_input("Name for this face")
            if st.button("Save Reference Face"):
                if face_name:
                    with st.spinner("Detecting face..."):
                        success = st.session_state.face_engine.add_reference_face(image, face_name)
                    if success:
                        st.success(f"✅ Reference face for '{face_name}' saved successfully!")
                        st.rerun()
                    else:
                        st.error("❌ No face detected in the image.")
                        st.info("""
                        **Tips for better face detection:**
                        - Make sure the face is clearly visible and facing forward
                        - Use well-lit photos (not too dark or too bright)
                        - Avoid sunglasses, hats, or hands covering the face
                        - Try a photo where the face takes up more of the frame
                        - Make sure the image is clear (not blurry)
                        """)
                else:
                    st.error("Please enter a name for the face.")

    st.subheader("Existing Reference Faces")
    reference_faces = st.session_state.face_engine.get_reference_faces()
    if reference_faces:
        cols = st.columns(min(3, len(reference_faces)))
        for i, (name, info) in enumerate(reference_faces.items()):
            with cols[i % 3]:
                if os.path.exists(info['image_path']):
                    st.image(info['image_path'], caption=name, width=150)
                else:
                    st.write(f"📷 {name}")
                if st.button(f"Remove {name}", key=f"remove_face_{i}"):
                    st.session_state.face_engine.remove_reference_face(name)
                    st.success(f"Reference face '{name}' removed!")
                    st.rerun()
    else:
        st.info("No reference faces uploaded.")

def show_live_monitoring():
    st.header("🎥 Live Video Streaming with Face Recognition")

    cameras = st.session_state.camera_manager.get_cameras()
    reference_faces = st.session_state.face_engine.get_reference_faces()

    if not cameras:
        st.warning("No cameras configured. Please add cameras first.")
        return

    if 'webrtc_transformer' not in st.session_state:
        st.session_state.webrtc_transformer = None

    st.subheader("Choose Streaming Source")
    streaming_mode = st.radio(
        "Select video source:",
        ["📹 Use Webcam (Built-in Camera)", "🔗 Use Network Camera (Coming Soon)"],
        help="For now, webcam streaming is available. Network camera streaming will be added soon."
    )

    if "Webcam" in streaming_mode:
        st.subheader("🔴 Live Webcam Stream with Face Recognition")

        if st.session_state.webrtc_transformer is None:
            st.session_state.webrtc_transformer = FaceRecognitionTransformer(
                st.session_state.face_engine,
                st.session_state.notification_service,
                "Webcam"
            )

        rtc_configuration = RTCConfiguration({
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:openrelay.metered.ca:80"]},
                {"urls": ["stun:stun.relay.metered.ca:80"]},
            ],
            "iceCandidatePoolSize": 10,
        })

        col1, col2 = st.columns([3, 1])
        with col1:
            webrtc_ctx = webrtc_streamer(
                key="live-monitoring",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=rtc_configuration,
                video_frame_callback=st.session_state.webrtc_transformer.transform,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            if webrtc_ctx.state.playing:
                st.success("🔴 LIVE - Webcam streaming active with face recognition")
                st.caption(
                    "YOLO overlays are enabled if selected in Settings. "
                    "Face recognition analyzes frames every 3 seconds."
                )
            else:
                st.info("▶️ Click the play button above to start live webcam streaming")

        with col2:
            st.subheader("🎯 LIVE DETECTION")
            if webrtc_ctx.state.playing:
                st.success("🔴 STREAMING")
                if st.session_state.webrtc_transformer and st.session_state.webrtc_transformer.latest_matches:
                    for i, match in enumerate(st.session_state.webrtc_transformer.latest_matches):
                        if match['name'] != 'No Match':
                            st.markdown(f"""
                            <div style="background-color: #1a4d1a; padding: 20px; border-radius: 10px; border: 3px solid #00ff00; margin: 10px 0;">
                                <h2 style="color: #00ff00; margin: 0; font-size: 24px;">✅ {match['name']}</h2>
                                <h1 style="color: #ffffff; margin: 10px 0; font-size: 48px; font-weight: bold;">{match['percentage']:.1f}%</h1>
                                <p style="color: #cccccc; margin: 0; font-size: 14px;">Confidence Match</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style="background-color: #4d1a1a; padding: 20px; border-radius: 10px; border: 3px solid #ff6666; margin: 10px 0;">
                                <h2 style="color: #ff6666; margin: 0; font-size: 24px;">❌ Unknown Face</h2>
                                <h1 style="color: #ffffff; margin: 10px 0; font-size: 48px; font-weight: bold;">{match['percentage']:.1f}%</h1>
                                <p style="color: #cccccc; margin: 0; font-size: 14px;">Below Threshold</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("👁️ Monitoring... waiting for faces")

                st.markdown("---")
                if st.session_state.webrtc_transformer:
                    st.metric("Frames", st.session_state.webrtc_transformer.frame_count)
                    st.metric("Detections", len(st.session_state.webrtc_transformer.latest_matches))
            else:
                st.info("⏸️ Stream Stopped")
                st.write("Click ▶️ to start")

            st.markdown("---")
            if reference_faces:
                st.caption("**Reference Faces:**")
                for name, info in reference_faces.items():
                    if os.path.exists(info['image_path']):
                        st.image(info['image_path'], caption=name, width=80)
                    else:
                        st.write(f"📷 {name}")
            else:
                st.warning("⚠️ No reference faces")

            st.markdown("---")
            current_threshold = st.session_state.face_engine.settings['confidence_threshold']
            st.caption(f"**Threshold:** {current_threshold*100:.1f}%")
            if st.button("🔄 Reset"):
                if st.session_state.webrtc_transformer:
                    st.session_state.webrtc_transformer.frame_count = 0
                    st.session_state.webrtc_transformer.latest_matches = []
                st.success("Reset!")

        if webrtc_ctx.state.playing and st.session_state.webrtc_transformer:
            if st.session_state.webrtc_transformer.latest_matches:
                st.subheader("🎯 Latest Face Detection Results")
                for i, match in enumerate(st.session_state.webrtc_transformer.latest_matches):
                    with st.expander(f"Face #{i+1} - {match['name']} ({match['percentage']:.1f}% confidence)", expanded=True):
                        detection_col1, detection_col2 = st.columns([1, 2])
                        with detection_col1:
                            if 'image_path' in match and os.path.exists(match['image_path']):
                                st.image(match['image_path'], caption="Detected Face", width=150)
                        with detection_col2:
                            st.write(f"**Match:** {match['name']}")
                            st.write(f"**Confidence:** {match['percentage']:.1f}%")
                            if match['name'] != 'No Match':
                                st.success("✅ Face recognized!")
                            else:
                                st.info("ℹ️ Face detected but no match above threshold")
                            if 'all_comparisons' in match and match['all_comparisons']:
                                st.write("**Comparison Results:**")
                                for comp in match['all_comparisons']:
                                    confidence_color = "🟢" if comp['percentage'] >= st.session_state.face_engine.settings['confidence_threshold']*100 else "🔴"
                                    st.write(f"{confidence_color} **{comp['reference_name']}**: {comp['percentage']:.1f}%")
            else:
                st.info("👁️ Live monitoring active - No faces detected in recent frames")

    else:
        st.info("🚧 Network camera streaming feature is coming soon! For now, please use the webcam option above.")
        if cameras:
            st.subheader("📹 Available Network Cameras (Preview)")
            for camera in cameras:
                with st.expander(f"📷 {camera['name']}"):
                    status = get_cached_camera_status(camera['url'])
                    if status:
                        frame = st.session_state.camera_manager.capture_frame(camera['url'])
                        if frame is not None:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            st.image(frame_rgb, caption=f"Static preview - {camera['name']}", use_column_width=True)
                    else:
                        st.error(f"Camera '{camera['name']}' is offline")

    with st.expander("💡 How to Use Live Video Streaming", expanded=False):
        st.write("""
        **Live Streaming Instructions:**
        1. Select "Use Webcam"
        2. Click ▶️ play and allow camera access
        3. YOLO (if enabled in Settings) draws orange person/object boxes
        4. Face recognition runs every 3s (green=match, red=unknown)

        **Requirements:**
        - Modern browser with WebRTC
        - Camera permission allowed
        """)

    if not reference_faces:
        st.info("💡 **Tip**: Upload reference faces in the 'Face References' section first to enable face recognition matching during live streaming.")

def show_settings():
    st.header("⚙️ Settings")

    # -------- Detection Backend ----------
    st.subheader("Detection Backend")
    if not YOLO_AVAILABLE:
        st.info("YOLO (Ultralytics) not detected. Install with: `pip install ultralytics`")
    backend = st.selectbox(
        "Select detection backend",
        ["Haar (basic)"] + (["YOLO (recommended)"] if YOLO_AVAILABLE else []),
        index=(1 if YOLO_AVAILABLE and st.session_state.detector_backend.startswith("YOLO") else 0)
    )

    # -------- YOLO Params (only shown if YOLO selectable) ----------
    if YOLO_AVAILABLE:
        st.markdown("**YOLO Parameters** (effective when backend is 'YOLO (recommended)')")
        col_a, col_b = st.columns(2)
        with col_a:
            conf = st.slider("YOLO confidence threshold", 0.1, 0.9, float(st.session_state.yolo_conf), 0.05)
            imgsz = st.select_slider("YOLO inference size", options=[640, 800, 960, 1280], value=int(st.session_state.yolo_imgsz))
        with col_b:
            person_only = st.checkbox("Detect person only", value=bool(st.session_state.yolo_person_only))
            model_name = st.selectbox("Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=["yolov8n.pt","yolov8s.pt","yolov8m.pt"].index(st.session_state.yolo_model_name))

    # -------- Email Notifications ----------
    st.subheader("Email Notifications")
    with st.form("email_settings"):
        email_enabled = st.checkbox("Enable email notifications")
        recipient_email = st.text_input("Recipient email address")
        smtp_server = st.text_input("SMTP server", value="smtp.gmail.com")
        smtp_port = st.number_input("SMTP port", value=587, min_value=1, max_value=65535)
        sender_email = st.text_input("Sender email address")
        sender_password = st.text_input("Sender email password", type="password")
        if st.form_submit_button("Save Email Settings"):
            settings = {
                'email_enabled': email_enabled,
                'recipient_email': recipient_email,
                'smtp_server': smtp_server,
                'smtp_port': smtp_port,
                'sender_email': sender_email,
                'sender_password': sender_password
            }
            st.session_state.notification_service.update_settings(settings)
            st.success("Email settings saved!")

    # -------- Face Detection/Recognition Settings ----------
    st.subheader("Face Recognition Settings")
    with st.form("detection_settings"):
        confidence_threshold = st.slider("Face recognition confidence threshold", 0.0, 1.0, float(st.session_state.face_engine.settings.get('confidence_threshold', 0.6)), 0.01)
        detection_frequency = st.number_input("Detection frequency (seconds)", value=int(st.session_state.face_engine.settings.get('detection_frequency', 5)), min_value=1, max_value=3600)
        save_det = st.form_submit_button("Save Face Settings")
        if save_det:
            st.session_state.face_engine.update_settings({
                'confidence_threshold': confidence_threshold,
                'detection_frequency': detection_frequency
            })
            st.success("Detection settings saved!")

    # -------- Save backend + YOLO selections ----------
    if st.button("💾 Apply Detector Settings"):
        st.session_state.detector_backend = backend
        if YOLO_AVAILABLE:
            st.session_state.yolo_conf = float(conf)
            st.session_state.yolo_imgsz = int(imgsz)
            st.session_state.yolo_person_only = bool(person_only)
            st.session_state.yolo_model_name = model_name
            # Re-init detector with new settings
            st.session_state.yolo_detector = _maybe_init_yolo()
        st.success("Detector settings applied!")

if __name__ == "__main__":
    main()
