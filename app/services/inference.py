from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

from app.models import Detection


class YoloObbService:
    def __init__(self, model_path: str, threshold: float = 0.4, image_size: int = 1024) -> None:
        self.model_path = model_path
        self.threshold = threshold
        self.image_size = image_size
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Ultralytics/YOLO could not be imported. Verify OpenCV + model dependencies.") from exc

        self.model = YOLO(model_path)

    def infer(self, frame_bgr: np.ndarray) -> tuple[list[Detection], float, np.ndarray]:
        start = time.perf_counter()
        results = self.model.predict(source=frame_bgr, conf=self.threshold, imgsz=self.image_size, verbose=False)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        detections: list[Detection] = []
        annotated = frame_bgr.copy()

        if not results:
            return detections, elapsed_ms, annotated

        result = results[0]
        names = self.model.names if isinstance(self.model.names, dict) else {}
        obb = getattr(result, "obb", None)
        if obb is None or getattr(obb, "xywhr", None) is None:
            return detections, elapsed_ms, annotated

        xywhr = obb.xywhr.detach().cpu().numpy()
        conf = obb.conf.detach().cpu().numpy()
        cls = obb.cls.detach().cpu().numpy()

        for i in range(len(conf)):
            class_id = int(cls[i])
            cx, cy, _w, _h, angle_rad = xywhr[i]
            angle_deg = float(angle_rad * 180.0 / math.pi)
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=str(names.get(class_id, f"class_{class_id}")),
                    confidence=float(conf[i]),
                    center_x=float(cx),
                    center_y=float(cy),
                    angle_deg=angle_deg,
                )
            )

        if hasattr(result, "plot"):
            annotated = result.plot()

        detections.sort(key=lambda det: det.confidence, reverse=True)
        return detections, elapsed_ms, annotated

    @staticmethod
    def save_image(image_bgr: np.ndarray, path: Path) -> None:
        """Save BGR ndarray as JPEG/PNG without requiring cv2 at import time."""
        path.parent.mkdir(parents=True, exist_ok=True)
        rgb = image_bgr[..., ::-1]
        Image.fromarray(rgb.astype("uint8")).save(path)
