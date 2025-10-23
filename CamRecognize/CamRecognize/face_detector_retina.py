# face_detector_retina.py
import numpy as np
from typing import List, Tuple

try:
    import insightface
    from insightface.app import FaceAnalysis
except Exception as e:
    raise RuntimeError("Install insightface: pip install insightface onnxruntime") from e


class RetinaFaceDetector:
    def __init__(self, providers=None):
        # providers: ["CUDAExecutionProvider"] if GPU, else ["CPUExecutionProvider"]
        self.app = FaceAnalysis(name="buffalo_l")  # good default
        self.app.prepare(ctx_id=0 if providers and "CUDAExecutionProvider" in providers else -1)

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Returns list of face boxes (x1,y1,x2,y2) in BGR frame.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        faces = self.app.get(frame_bgr)  # expects BGR
        boxes = []
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int).tolist()
            boxes.append((x1, y1, x2, y2))
        return boxes
