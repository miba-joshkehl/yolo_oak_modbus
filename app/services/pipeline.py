from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from app.models import InferenceRun, SystemStatus
from app.services.inference import YoloObbService
from app.services.modbus_server import ModbusBridge

logger = logging.getLogger(__name__)


class VisionPipeline:
    def __init__(
        self,
        camera,
        inference_service: YoloObbService,
        modbus: ModbusBridge,
        capture_dir: Path,
        annotated_dir: Path,
        poll_interval_ms: int,
    ) -> None:
        self.camera = camera
        self.inference_service = inference_service
        self.modbus = modbus
        self.capture_dir = capture_dir
        self.annotated_dir = annotated_dir
        self.poll_interval_ms = poll_interval_ms

        self._lock = threading.RLock()
        self._last_run: InferenceRun | None = None
        self._running = False
        self._worker: threading.Thread | None = None
        self._is_busy = False
        self._has_error = False
        self._heartbeat = 0

    def start(self) -> None:
        self.modbus.start()
        self._running = True
        self._worker = threading.Thread(target=self._loop, name="vision-pipeline", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        while self._running:
            try:
                self._heartbeat = self.modbus.tick_heartbeat()
                if self.modbus.read_trigger_and_clear():
                    self.trigger("modbus")
            except Exception as exc:
                logger.exception("Background pipeline error: %s", exc)
            time.sleep(max(self.poll_interval_ms, 20) / 1000.0)

    def trigger(self, source: str = "api") -> InferenceRun:
        with self._lock:
            if self._is_busy:
                raise RuntimeError("Pipeline is already processing another trigger.")
            self._is_busy = True

        self.modbus.mark_busy(True)
        self.modbus.mark_ready(False)
        self.modbus.mark_error(False)

        stamp = datetime.now(timezone.utc)
        filename = stamp.strftime("%Y%m%d_%H%M%S_%f")

        try:
            frame = self.camera.capture_frame()
            capture_path = self.capture_dir / f"capture_{filename}.jpg"
            self.inference_service.save_image(frame, capture_path)

            detections, elapsed_ms, annotated = self.inference_service.infer(frame)
            annotated_path = self.annotated_dir / f"annotated_{filename}.jpg"
            self.inference_service.save_image(annotated, annotated_path)

            self.modbus.publish_results(detections)
            self.modbus.mark_ready(True)

            run = InferenceRun(
                trigger_source="modbus" if source == "modbus" else "api",
                captured_at_utc=stamp,
                image_path=str(capture_path),
                annotated_image_path=str(annotated_path),
                inference_ms=elapsed_ms,
                detections=detections,
                error=None,
            )

            with self._lock:
                self._last_run = run
                self._has_error = False
            return run
        except Exception as exc:
            self.modbus.mark_error(True)
            error_run = InferenceRun(
                trigger_source="modbus" if source == "modbus" else "api",
                captured_at_utc=stamp,
                image_path="",
                annotated_image_path="",
                inference_ms=0,
                detections=[],
                error=str(exc),
            )
            with self._lock:
                self._last_run = error_run
                self._has_error = True
            raise
        finally:
            with self._lock:
                self._is_busy = False
            self.modbus.mark_busy(False)

    def status(self) -> SystemStatus:
        with self._lock:
            return SystemStatus(
                is_busy=self._is_busy,
                ready=self.modbus.get_coil(self.modbus.layout.ready_coil),
                has_error=self._has_error,
                heartbeat=self._heartbeat,
                last_run=self._last_run,
            )
