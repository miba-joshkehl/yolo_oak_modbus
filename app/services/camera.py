from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OakCameraConfig:
    preview_width: int = 1280
    preview_height: int = 720


class OakCamera:
    """Wrapper around DepthAI for one-shot frame capture."""

    def __init__(self, config: OakCameraConfig) -> None:
        self.config = config

    def capture_frame(self) -> np.ndarray:
        try:
            import depthai as dai
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("DepthAI is not available. Install depthai and connect an OAK camera.") from exc

        pipeline = dai.Pipeline()
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setPreviewSize(self.config.preview_width, self.config.preview_height)
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        xout = pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("rgb")
        cam.preview.link(xout.input)

        with dai.Device(pipeline) as device:
            queue = device.getOutputQueue(name="rgb", maxSize=1, blocking=True)
            frame = queue.get().getCvFrame()
            if frame is None:
                raise RuntimeError("OAK camera returned no frame.")
            return frame


class OpenCVCameraFallback:
    """Fallback camera source if OAK cannot be initialized."""

    def __init__(self, index: int = 0) -> None:
        self.index = index

    def capture_frame(self) -> np.ndarray:
        try:
            import cv2
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("OpenCV camera fallback unavailable: cv2 could not be imported.") from exc

        cap = cv2.VideoCapture(self.index)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError("Fallback OpenCV camera capture failed.")
        return frame


def build_camera(preview_width: int, preview_height: int, use_oak: bool = True):
    # We don't probe hardware here to avoid long startup delays and side effects.
    # The selected backend is used lazily on trigger.
    if use_oak:
        logger.info("Configured to use OAK camera backend.")
        return OakCamera(OakCameraConfig(preview_width, preview_height))

    logger.info("Configured to use OpenCV fallback camera backend.")
    return OpenCVCameraFallback()
