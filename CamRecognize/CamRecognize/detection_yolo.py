# detection_yolo.py
from typing import List, Dict, Optional
import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError(
        "Ultralytics not installed. Run: pip install ultralytics"
    ) from e


class YOLODetector:
    """
    Thin wrapper around Ultralytics YOLO for fast person/object detection.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",   # good start; upgrade to 'yolov8s.pt' or 'yolov8m.pt' for range
        conf: float = 0.35,
        iou: float = 0.5,
        classes: Optional[List[int]] = None,  # e.g., [0] for person only; None for all COCO classes
        imgsz: int = 960                     # increase for long range; try 1280 on good GPUs/CPUs
    ):
        self.model = YOLO(model_name)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.classes = classes  # COCO: person=0

        # cache class names
        try:
            self.names = self.model.model.names  # newer
        except Exception:
            self.names = self.model.names

    def predict(self, frame_bgr: np.ndarray) -> List[Dict]:
        """
        Runs detection on a single BGR frame and returns a list of dicts:
        [{'label': str, 'conf': float, 'box': (x1, y1, x2, y2)}]
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        # Ultralytics expects numpy image in BGR just fine; set params:
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

        boxes = r0.boxes
        for b in boxes:
            xyxy = b.xyxy[0].tolist()  # x1,y1,x2,y2
            x1, y1, x2, y2 = map(int, xyxy)
            conf = float(b.conf[0].item()) if hasattr(b, "conf") else 0.0
            cls_id = int(b.cls[0].item()) if hasattr(b, "cls") else -1
            label = self.names.get(cls_id, str(cls_id)) if isinstance(self.names, dict) else (
                self.names[cls_id] if 0 <= cls_id < len(self.names) else str(cls_id)
            )
            detections.append({
                "label": label,
                "conf": conf,
                "box": (x1, y1, x2, y2),
                "cls_id": cls_id
            })
        return detections
