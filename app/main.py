from __future__ import annotations

# =========================
# Imports
# =========================
import base64
import io
import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock, Thread
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field
from ultralytics import YOLO

# ---- NEW: DepthAI / OpenCV for OAK-4S capture (from your Cam_Test.py) ----
# Pattern mirrors your trigger + highest_res Script routing
import cv2
import depthai as dai

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("yolo_api")

# --- Optional Modbus imports (TCP only; graceful if not installed) ---
# NOTE:
#   * Import ONLY the TCP client here. Importing the Serial client requires the optional
#     'pyserial' extra; if it's missing, importing ModbusSerialClient would raise ImportError.
#   * Try both common TCP client import paths because PyModbus 3.x layouts can vary.
#   * We narrow the exception to ImportError so unrelated exceptions don't look like "not installed".
_PYMODBUS_IMPORT_ERROR = None
try:
    try:
        from pymodbus.client import ModbusTcpClient  # common 3.x import path
    except ImportError:
        from pymodbus.client.tcp import ModbusTcpClient  # fallback path seen in some installs
    HAVE_PYMODBUS = True
except ImportError as exc:
    HAVE_PYMODBUS = False
    _PYMODBUS_IMPORT_ERROR = exc
    logger.error("PyModbus import failed: %s", exc)

# Log the actual pymodbus version in use (helps diagnose API differences)
try:
    import pymodbus  # type: ignore
    logger.info("Using pymodbus %s", getattr(pymodbus, "__version__", "unknown"))
except Exception:
    pass

# ----------------------------
# Small utilities
# ----------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _resolve_existing_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    return path


# ----------------------------
# API Models
# ----------------------------
ModelSize = Literal["medium", "large"]
DeviceMode = Literal["auto", "cpu", "gpu"]


class OBBGeometry(BaseModel):
    # Four ordered points flattened: [x1,y1,x2,y2,x3,y3,x4,y4]
    corners_xyxyxyxy: List[float]
    # Center-based representation (xywhr); angle is radians, we also include degrees
    xc: float
    yc: float
    w: float
    h: float
    angle_rad: float
    angle_deg: float


class DetectionResult(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    # Back-compat axis-aligned rectangle (AABB) derived from the OBB polygon
    x1: float
    y1: float
    x2: float
    y2: float
    # Full oriented box geometry
    obb: OBBGeometry


class ModelInfoResponse(BaseModel):
    model_size: ModelSize
    model_alias: str
    weights_path: Optional[str]
    optimize_for_inference: bool
    device_mode: DeviceMode
    torch_device: str
    cuda_available: bool
    loaded_at_utc: str
    class_count: int
    class_names: Dict[int, str]


class LoadModelRequest(BaseModel):
    model_size: ModelSize
    weights_path: Optional[str] = None
    optimize_for_inference: bool = True
    device_mode: DeviceMode = "auto"


class InferRequest(BaseModel):
    image_path: str = Field(..., min_length=1)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    imgsz: int = Field(default=640, ge=64, le=8192, description="Inference image size (pixels)")
    save_annotated_image: bool = True
    annotated_image_path: Optional[str] = None
    return_image_base64: bool = True


class InferResponse(BaseModel):
    image_path: str
    model_size: ModelSize
    torch_device: str
    threshold: float
    inference_time_ms: float
    detections_count: int
    detections: List[DetectionResult]
    annotated_image_path: Optional[str]
    annotated_image_mime_type: Optional[str]
    annotated_image_base64: Optional[str]


# ----------------------------
# Core YOLO service
# ----------------------------
class YoloService:
    # Default OBB model aliases by size
    _MODEL_TYPES = {
        "medium": "yolo11m-obb.pt",
        "large": "yolo11l-obb.pt",
    }

    def __init__(
        self,
        default_model_size: ModelSize,
        default_weights_path: Optional[str],
        optimize_for_inference: bool,
        default_device_mode: DeviceMode,
    ) -> None:
        self._lock = RLock()
        self._model: Optional[YOLO] = None
        self._model_size: ModelSize = default_model_size
        self._model_alias: str = ""
        self._weights_path: Optional[str] = None
        self._optimize_for_inference = optimize_for_inference
        self._device_mode: DeviceMode = default_device_mode
        self._torch_device: str = "cpu"
        self._loaded_at_utc: str = _utc_now_iso()

        self.load_model(
            model_size=default_model_size,
            weights_path=default_weights_path,
            optimize_for_inference=optimize_for_inference,
            device_mode=default_device_mode,
        )

    @staticmethod
    def _normalize_weights_path(weights_path: Optional[str]) -> Optional[str]:
        if weights_path is None or not str(weights_path).strip():
            return None
        path = Path(weights_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Weights path does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"Weights path is not a file: {path}")
        return str(path)

    @staticmethod
    def _resolve_torch_device(device_mode: DeviceMode) -> str:
        if device_mode == "cpu":
            return "cpu"
        if device_mode == "gpu":
            if not torch.cuda.is_available():
                raise RuntimeError("GPU requested, but CUDA is not available on this machine.")
            return "cuda"
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _class_map(model: YOLO) -> Dict[int, str]:
        names = getattr(model, "names", {})
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, list):
            return {idx: str(value) for idx, value in enumerate(names)}
        return {}

    def load_model(
        self,
        model_size: ModelSize,
        weights_path: Optional[str],
        optimize_for_inference: bool,
        device_mode: DeviceMode,
    ) -> None:
        if model_size not in self._MODEL_TYPES:
            raise ValueError(f"Unsupported model size: {model_size}")

        model_alias = self._MODEL_TYPES[model_size]
        normalized_weights = self._normalize_weights_path(weights_path)
        selected_weights = normalized_weights or model_alias
        torch_device = self._resolve_torch_device(device_mode)

        try:
            model = YOLO(selected_weights)
            model.to(torch_device)
            if optimize_for_inference:
                model.fuse()
        except Exception as exc:
            raise RuntimeError(f"Failed to load YOLO model '{selected_weights}': {exc}") from exc

        with self._lock:
            self._model = model
            self._model_size = model_size
            self._model_alias = model_alias
            self._weights_path = normalized_weights
            self._optimize_for_inference = optimize_for_inference
            self._device_mode = device_mode
            self._torch_device = torch_device
            self._loaded_at_utc = _utc_now_iso()

    def info(self) -> ModelInfoResponse:
        with self._lock:
            if self._model is None:
                raise RuntimeError("No model loaded.")
            class_map = self._class_map(self._model)
            return ModelInfoResponse(
                model_size=self._model_size,
                model_alias=self._model_alias,
                weights_path=self._weights_path,
                optimize_for_inference=self._optimize_for_inference,
                device_mode=self._device_mode,
                torch_device=self._torch_device,
                cuda_available=torch.cuda.is_available(),
                loaded_at_utc=self._loaded_at_utc,
                class_count=len(class_map),
                class_names=class_map,
            )

    def predict(
        self,
        image: Image.Image,
        threshold: float,
        imgsz: int,
    ) -> tuple[List[DetectionResult], float, ModelSize, str]:
        with self._lock:
            if self._model is None:
                raise RuntimeError("No model loaded.")
            model = self._model
            class_map = self._class_map(model)
            model_size = self._model_size
            torch_device = self._torch_device

        # PIL (RGB) -> ndarray (BGR) for Ultralytics
        arr_bgr = np.array(image)[:, :, ::-1]

        start = time.perf_counter()
        results = model.predict(
            source=arr_bgr,
            conf=threshold,
            imgsz=imgsz,
            verbose=False,
            device=torch_device,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        # Debug candidate counts
        try:
            r = results[0]
            cnt_obb = 0 if getattr(r, "obb", None) is None or getattr(r.obb, "conf", None) is None else len(r.obb.conf)
            cnt_aabb = 0 if getattr(r, "boxes", None) is None or getattr(r.boxes, "conf", None) is None else len(r.boxes.conf)
            logger.info(f"DEBUG candidates: OBB={cnt_obb}, AABB={cnt_aabb}, conf={threshold}, imgsz={imgsz}, device={torch_device}")
        except Exception:
            pass

        detections: List[DetectionResult] = []
        if results:
            r = results[0]
            # Primary: OBB predictions
            obb = getattr(r, "obb", None)
            if obb is not None:
                corners = obb.xyxyxyxy.detach().cpu().numpy() if getattr(obb, "xyxyxyxy", None) is not None else np.empty((0, 8))
                xywhr = obb.xywhr.detach().cpu().numpy() if getattr(obb, "xywhr", None) is not None else np.empty((0, 5))
                conf = obb.conf.detach().cpu().numpy() if getattr(obb, "conf", None) is not None else np.empty((0,))
                cls = obb.cls.detach().cpu().numpy() if getattr(obb, "cls", None) is not None else np.empty((0,))
                N = len(corners)
                for i in range(N):
                    class_id = int(cls[i])
                    # Some Ultralytics versions return corners as (4,2); flatten to (8,)
                    c = corners[i]
                    c_flat = np.asarray(c, dtype=float).reshape(-1)  # -> (8,)
                    cx, cy, w, h, angle_rad = xywhr[i]
                    angle_deg = float(angle_rad * 180.0 / math.pi)
                    xs = c_flat[0::2]
                    ys = c_flat[1::2]
                    x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
                    detections.append(
                        DetectionResult(
                            class_id=class_id,
                            class_name=class_map.get(class_id, f"class_{class_id}"),
                            confidence=float(conf[i]),
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            obb=OBBGeometry(
                                corners_xyxyxyxy=[float(v) for v in c_flat.tolist()],
                                xc=float(cx),
                                yc=float(cy),
                                w=float(w),
                                h=float(h),
                                angle_rad=float(angle_rad),
                                angle_deg=angle_deg,
                            ),
                        )
                    )
            else:
                # Fallback: legacy AABB
                boxes = getattr(r, "boxes", None)
                if boxes is not None and getattr(boxes, "xyxy", None) is not None:
                    xyxy = boxes.xyxy.detach().cpu().numpy()
                    conf = boxes.conf.detach().cpu().numpy()
                    cls = boxes.cls.detach().cpu().numpy()
                    for i in range(len(xyxy)):
                        class_id = int(cls[i])
                        x1, y1, x2, y2 = map(float, xyxy[i])
                        detections.append(
                            DetectionResult(
                                class_id=class_id,
                                class_name=class_map.get(class_id, f"class_{class_id}"),
                                confidence=float(conf[i]),
                                x1=x1,
                                y1=y1,
                                x2=x2,
                                y2=y2,
                                obb=OBBGeometry(  # degenerate OBB = axis-aligned
                                    corners_xyxyxyxy=[x1, y1, x2, y1, x2, y2, x1, y2],
                                    xc=(x1 + x2) / 2.0,
                                    yc=(y1 + y2) / 2.0,
                                    w=(x2 - x1),
                                    h=(y2 - y1),
                                    angle_rad=0.0,
                                    angle_deg=0.0,
                                ),
                            )
                        )

        return detections, elapsed_ms, model_size, torch_device


# ----------------------------
# Visualization helpers
# ----------------------------
def _annotate_image(image: Image.Image, detections: List[DetectionResult]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for det in detections:
        pts = det.obb.corners_xyxyxyxy  # [x1,y1,x2,y2,x3,y3,x4,y4]
        polygon = [(pts[0], pts[1]), (pts[2], pts[3]), (pts[4], pts[5]), (pts[6], pts[7])]
        draw.polygon(polygon, outline=(0, 255, 0), width=2)
        # Label near the top-left of the polygon
        minx = min(p[0] for p in polygon)
        miny = min(p[1] for p in polygon)
        label = f"{det.class_name} {det.confidence:.2f} ({det.obb.angle_deg:.1f}°)"
        draw.text((minx + 3, max(miny - 14, 0)), label, fill=(0, 255, 0))
    return annotated


def _image_to_base64_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _resolve_annotated_output_path(
    input_path: Path,
    requested_output: Optional[str],
    default_output_dir: Optional[str],
) -> Path:
    if requested_output:
        output_path = Path(requested_output).expanduser()
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
    elif default_output_dir:
        output_dir = Path(default_output_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir
        output_dir = output_dir.resolve()
        output_path = output_dir / f"{input_path.stem}_detections.png"
    else:
        output_path = input_path.with_name(f"{input_path.stem}_detections.png")

    if not output_path.suffix:
        output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path.resolve()


# ----------------------------
# NEW: Lightweight UR Modbus client + push helper
# ----------------------------

# Helper to auto-detect the correct keyword for the unit/slave/device id across pymodbus versions
import inspect


def _kw_device_id(func):
    """
    Return the correct keyword name for unit/slave/device_id accepted by this pymodbus function.

    Order we try (based on release history):
      - device_id (3.10+)
      - slave     (<=3.9 and many examples)
      - unit      (older code / forks)

    Falls back to 'device_id' if the signature can't be inspected; most versions accept **kwargs.
    """
    try:
        sig = inspect.signature(func)
        params = sig.parameters
        for name in ("device_id", "slave", "unit"):
            if name in params:
                return name
    except Exception:
        pass
    return "device_id"


AddressBase = Literal["zero_based", "one_based"]


class URModbusClient:
    """
    Minimal Modbus/TCP client for a UR controller's server.
    Supports coils, u16 holding registers, and 32-bit float (two registers).
    """

    def __init__(
        self,
        host: str,
        port: int = 502,
        unit_id: int = 255,
        address_base: AddressBase = "zero_based",
        timeout: float = 2.0,
        retries: int = 3,
    ):
        if not HAVE_PYMODBUS:
            raise RuntimeError(f"PyModbus import failed or incompatible API: {_PYMODBUS_IMPORT_ERROR}")
        self.host = host
        self.port = port
        self.unit_id = unit_id
        self.address_base = address_base
        self.timeout = timeout
        self.retries = retries
        self.client = ModbusTcpClient(host=self.host, port=self.port, timeout=self.timeout)

    def __enter__(self):
        if not self.client.connect():
            raise ConnectionError(f"Could not connect to UR Modbus server at {self.host}:{self.port}")
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.client.close()
        except Exception:
            pass

    def _addr(self, addr: int) -> int:
        if self.address_base == "one_based":
            if addr <= 0:
                raise ValueError("1-based address must be >= 1")
            return addr - 1
        if addr < 0:
            raise ValueError("0-based address must be >= 0")
        return addr

    def write_coil(self, address: int, value: bool) -> bool:
        a = self._addr(address)
        kw = _kw_device_id(self.client.write_coil)
        for _ in range(self.retries):
            rr = self.client.write_coil(a, bool(value), **{kw: self.unit_id})
            if rr and not rr.isError():
                return True
        return False

    def write_u16(self, address: int, value: int) -> bool:
        v = int(value)
        if not (0 <= v <= 0xFFFF):
            raise ValueError("u16 value must be in [0, 65535]")
        a = self._addr(address)
        kw = _kw_device_id(self.client.write_register)
        for _ in range(self.retries):
            rr = self.client.write_register(a, v, **{kw: self.unit_id})
            if rr and not rr.isError():
                return True
        return False

    def write_f32(self, address: int, value: float, byteorder: str = "BIG", wordorder: str = "BIG") -> bool:
        """Write a 32-bit float using the current PyModbus conversion helper."""
        a = self._addr(address)
        try:
            dtype = getattr(self.client.DATATYPE, "FLOAT32")
            regs = self.client.convert_to_registers(
                float(value),
                data_type=dtype,
                word_order=(str(wordorder).upper() != "LITTLE"),
            )
        except Exception as exc:
            logger.error("PyModbus FLOAT32 conversion failed: %s", exc)
            return False

        kw = _kw_device_id(self.client.write_registers)
        for _ in range(self.retries):
            rr = self.client.write_registers(a, values=regs, **{kw: self.unit_id})
            if rr and not rr.isError():
                return True
        return False

def _maybe_push_modbus(
    detections: List[DetectionResult],
    inference_ms: float,
):
    """
    If MODBUS_CONFIG['enabled'] is True, push a summary to UR:
      - coil_alert: True if any detection
      - u16_count: number of detections (clamped [0, 65535])
      - u16_best_conf_x1000: best confidence scaled to 0..1000 (one register)
    Safe no-op if pymodbus is not installed or config is disabled.
    """
    cfg = MODBUS_CONFIG or {}
    if not cfg or not bool(cfg.get("enabled", False)):
        return
    if not HAVE_PYMODBUS:
        logger.warning("Modbus disabled: PyModbus import failed or incompatible API. Error: %s", _PYMODBUS_IMPORT_ERROR)
        return

    host = cfg.get("host", "127.0.0.1")
    port = int(cfg.get("port", 502))
    unit_id = int(cfg.get("unit_id", 255))
    address_base = cfg.get("address_base", "zero_based")
    byteorder_str = str(cfg.get("byteorder", "BIG")).upper()
    wordorder_str = str(cfg.get("wordorder", "BIG")).upper()
    map_ = cfg.get("map", {}) or {}

    coil_alert = map_.get("coil_alert")
    u16_count = map_.get("u16_count")
    u16_best_conf_x1000 = map_.get("u16_best_conf_x1000")

    count = len(detections)
    best_conf = max((d.confidence for d in detections), default=0.0)
    alert = count > 0

    # clamp/sanitize
    count_u16 = max(0, min(65535, int(count)))
    best_conf_x1000 = max(0, min(1000, int(round(float(best_conf) * 1000.0))))

    # byte/word order strings are kept for future FLOAT32 writes
    bo = byteorder_str  # noqa: F841
    wo = wordorder_str  # noqa: F841

    try:
        with URModbusClient(
            host=host, port=port, unit_id=unit_id, address_base=address_base
        ) as cli:
            if coil_alert is not None:
                ok = cli.write_coil(int(coil_alert), alert)
                if not ok:
                    logger.warning(f"Modbus write failed: coil {coil_alert} <- {alert}")

            if u16_count is not None:
                ok = cli.write_u16(int(u16_count), count_u16)
                if not ok:
                    logger.warning(f"Modbus write failed: HR {u16_count} <- {count_u16}")

            if u16_best_conf_x1000 is not None:
                ok = cli.write_u16(int(u16_best_conf_x1000), best_conf_x1000)
                if not ok:
                    logger.warning(f"Modbus write failed: HR {u16_best_conf_x1000} <- {best_conf_x1000}")
    except Exception as exc:
        logger.exception(f"Modbus push failed: {exc}")


# ----------------------------
# Config
# ----------------------------
CONFIG_PATH = Path.cwd() / "config" / "yolo.config.json"


def _load_startup_config(config_path: Path) -> Dict[str, Any]:
    # Defaults (merged with file content if present)
    default_config: Dict[str, Any] = {
        "default_model": "medium",
        "default_weights_path": "models/weights.pt",  # can be None or a custom absolute/relative path
        "default_device": "auto",
        "optimize_for_inference": True,
        "annotation_output_dir": "outputs/annotated",
        # NEW: capture defaults (for capture+infer helpers)
        "capture": {
            "save_dir": "captures",
            "imgsz": 640,
            "threshold": 0.5,
            "save_annotated_image": True
        },
        # NEW: Folder watcher (optional)
        "folder_watcher": {
            "enabled": False,
            "watch_dir": "watched_input",
            "imgsz": 640,
            "threshold": 0.5,
            "save_annotated_image": True
        },
        # NEW: Modbus defaults (UR as Modbus TCP server; API is client)
        "modbus": {
            "enabled": False,  # set True to activate
            "host": "192.168.1.11",
            "port": 502,
            "unit_id": 255,  # typically ignored on TCP; 255 is common default
            "address_base": "zero_based",
            "byteorder": "BIG",
            "wordorder": "BIG",
            "poll_ms": 100,
            "map": {
                "coil_alert": 128,              # boolean -> coil
                "u16_count": 300,               # u16 -> holding register
                "u16_best_conf_x1000": 302,     # scaled integer: round(best_conf*1000)
                "coil_capture": 129             # boolean capture trigger (rising edge)
            }
        },
    }

    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(default_config, indent=2), encoding="utf-8")

    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Invalid config file at '{config_path}': {exc}") from exc

    if not isinstance(raw, dict):
        raise RuntimeError(f"Invalid config file at '{config_path}': expected a JSON object.")

    # Merge file over defaults (shallow merge for top-level keys)
    config = {**default_config, **raw}

    model_size = str(config.get("default_model", "medium")).strip().lower()
    device_mode = str(config.get("default_device", "auto")).strip().lower()
    if model_size not in {"medium", "large"}:
        raise RuntimeError("config.default_model must be either 'medium' or 'large'.")
    if device_mode not in {"auto", "cpu", "gpu"}:
        raise RuntimeError("config.default_device must be one of: auto, cpu, gpu.")

    weights_raw = config.get("default_weights_path")
    default_weights_path = str(weights_raw).strip() if weights_raw is not None else ""
    annotation_raw = config.get("annotation_output_dir")
    annotation_output_dir = str(annotation_raw).strip() if annotation_raw is not None else ""

    # Normalize modbus block
    modbus_cfg = config.get("modbus") or {}
    if not isinstance(modbus_cfg, dict):
        modbus_cfg = {}
    modbus_cfg = {
        "enabled": bool(modbus_cfg.get("enabled", False)),
        "host": str(modbus_cfg.get("host", "192.168.0.10")),
        "port": int(modbus_cfg.get("port", 502)),
        "unit_id": int(modbus_cfg.get("unit_id", 255)),
        "address_base": str(modbus_cfg.get("address_base", "zero_based")),
        "byteorder": str(modbus_cfg.get("byteorder", "BIG")).upper(),
        "wordorder": str(modbus_cfg.get("wordorder", "BIG")).upper(),
        "poll_ms": int(modbus_cfg.get("poll_ms", 100)),
        "map": modbus_cfg.get("map", {}),
    }

    # Normalize capture block
    cap_cfg = config.get("capture") or {}
    cap_cfg = {
        "save_dir": str(cap_cfg.get("save_dir", "captures")),
        "imgsz": int(cap_cfg.get("imgsz", 640)),
        "threshold": float(cap_cfg.get("threshold", 0.5)),
        "save_annotated_image": bool(cap_cfg.get("save_annotated_image", True)),
    }

    # Normalize folder watcher block
    fw_cfg = config.get("folder_watcher") or {}
    fw_cfg = {
        "enabled": bool(fw_cfg.get("enabled", False)),
        "watch_dir": str(fw_cfg.get("watch_dir", "watched_input")),
        "imgsz": int(fw_cfg.get("imgsz", 640)),
        "threshold": float(fw_cfg.get("threshold", 0.5)),
        "save_annotated_image": bool(fw_cfg.get("save_annotated_image", True)),
    }

    return {
        "default_model": model_size,
        "default_weights_path": default_weights_path or None,
        "default_device": device_mode,
        "optimize_for_inference": bool(config.get("optimize_for_inference", True)),
        "annotation_output_dir": annotation_output_dir or None,
        "modbus": modbus_cfg,
        "capture": cap_cfg,
        "folder_watcher": fw_cfg,
    }


STARTUP_CONFIG = _load_startup_config(CONFIG_PATH)
DEFAULT_MODEL_SIZE = STARTUP_CONFIG["default_model"]
DEFAULT_WEIGHTS_PATH = STARTUP_CONFIG["default_weights_path"]
DEFAULT_OPTIMIZE = STARTUP_CONFIG["optimize_for_inference"]
DEFAULT_ANNOTATED_OUTPUT_DIR = STARTUP_CONFIG["annotation_output_dir"]
DEFAULT_DEVICE_MODE = STARTUP_CONFIG["default_device"]
MODBUS_CONFIG = STARTUP_CONFIG.get("modbus", {"enabled": False})
CAPTURE_CONFIG = STARTUP_CONFIG.get("capture", {})
FOLDER_WATCHER_CONFIG = STARTUP_CONFIG.get("folder_watcher", {})

# Prepare capture and watch dirs
CAPTURE_SAVE_DIR = Path(CAPTURE_CONFIG.get("save_dir", "captures")).resolve()
CAPTURE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
WATCH_DIR = Path(FOLDER_WATCHER_CONFIG.get("watch_dir", "watched_input")).resolve()
WATCH_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Service + FastAPI app
# ----------------------------
service = YoloService(
    default_model_size=DEFAULT_MODEL_SIZE,  # derived from your original main.py
    default_weights_path=DEFAULT_WEIGHTS_PATH,
    optimize_for_inference=DEFAULT_OPTIMIZE,
    default_device_mode=DEFAULT_DEVICE_MODE,
)

app = FastAPI(
    title="YOLO Offline Inference API",
    version="1.1.0",
    description="Offline YOLO OBB inference API with OAK-4S capture, folder watcher, and Modbus capture trigger.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Inference wrapper
# ----------------------------
def _run_inference(
    image: Image.Image,
    image_path: str,
    threshold: float,
    imgsz: int,
    save_annotated_image: bool,
    annotated_image_path: Optional[str],
    return_image_base64: bool,
) -> InferResponse:
    try:
        detections, elapsed_ms, model_size, torch_device = service.predict(
            image=image, threshold=threshold, imgsz=imgsz
        )
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    needs_annotated_image = save_annotated_image or return_image_base64
    annotated = _annotate_image(image=image, detections=detections) if needs_annotated_image else None

    annotated_path_str: Optional[str] = None
    if save_annotated_image and annotated is not None:
        try:
            output_path = _resolve_annotated_output_path(
                input_path=Path(image_path),
                requested_output=annotated_image_path,
                default_output_dir=DEFAULT_ANNOTATED_OUTPUT_DIR,
            )
            annotated.save(output_path)
            annotated_path_str = str(output_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to save annotated image: {exc}") from exc

    annotated_base64: Optional[str] = None
    mime_type: Optional[str] = None
    if return_image_base64 and annotated is not None:
        annotated_base64 = _image_to_base64_png(annotated)
        mime_type = "image/png"

    # Push Modbus summary (alert/count/best_conf_x1000) if enabled
    try:
        _maybe_push_modbus(detections=detections, inference_ms=elapsed_ms)
    except Exception:
        # Do not let fieldbus errors affect REST responses
        pass

    logger.info(f"DEBUG RESP threshold type={type(threshold)} val={threshold}")
    return InferResponse(
        image_path=image_path,
        model_size=model_size,
        torch_device=torch_device,
        threshold=float(threshold),
        inference_time_ms=round(float(elapsed_ms), 2),
        detections_count=int(len(detections)),
        detections=detections,
        annotated_image_path=annotated_path_str,
        annotated_image_mime_type=mime_type,
        annotated_image_base64=annotated_base64,
    )


# ----------------------------
# NEW: OAK-4S capture (inlined from your Cam_Test.py)
# ----------------------------
def capture_frame(resize_to: Optional[Tuple[int, int]] = (640, 640)) -> Image.Image:
    """
    Captures a single still using the OAK-4S pipeline pattern from your Cam_Test.py:
    - High-res stream routed via Script node
    - Host triggers capture via a short message
    - Optional resize
    Returns a PIL RGB image.
    """
    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        # Request max resolution frame output (as in your example)
        stream_highest_res = cam.requestFullResolutionOutput(useHighestResolution=True)

        script = pipeline.create(dai.node.Script)
        stream_highest_res.link(script.inputs["in"])
        script.setScript(
            """
while True:
    message = node.inputs["in"].get()
    trigger = node.inputs["trigger"].tryGet()
    if trigger is not None:
        node.io["highest_res"].send(message)
"""
        )

        imgManip = pipeline.create(dai.node.ImageManip)
        stream_highest_res.link(imgManip.inputImage)
        # Keep the same downscale staging from your Cam_Test
        imgManip.initialConfig.setOutputSize(1333, 1000)
        imgManip.setMaxOutputFrameSize(1333 * 1000 * 3)

        downscaled_q = imgManip.out.createOutputQueue()
        highest_q = script.outputs["highest_res"].createOutputQueue()
        q_trigger = script.inputs["trigger"].createInputQueue()

        # Connect/start device
        device = dai.Device(pipeline)  # noqa: F841 (kept for completeness)

        # Prime preview (optional)
        try:
            _ = downscaled_q.get()
        except Exception:
            pass

        # Send a single trigger to forward one highest-res frame
        q_trigger.send(dai.Buffer())

        # Block for the one-shot highest-res frame
        highres = highest_q.get()
        frame_full = highres.getCvFrame()  # BGR ndarray
        if resize_to:
            frame_full = cv2.resize(frame_full, resize_to, interpolation=cv2.INTER_AREA)

        # BGR -> RGB PIL
        rgb = cv2.cvtColor(frame_full, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)


# ----------------------------
# NEW: capture_and_infer helper + test endpoint
# ----------------------------
def capture_and_infer(
    threshold: float = CAPTURE_CONFIG.get("threshold", 0.5),
    imgsz: int = CAPTURE_CONFIG.get("imgsz", 640),
    save_annotated_image: bool = CAPTURE_CONFIG.get("save_annotated_image", True),
    annotated_image_path: Optional[str] = None,
    return_image_base64: bool = True,
) -> InferResponse:
    """
    Captures a single frame from OAK-4S and runs inference through the same pipeline as /infer.
    Also saves the raw capture into CAPTURE_SAVE_DIR for traceability.
    """
    # capture image
    image_pil: Image.Image = capture_frame(resize_to=(imgsz, imgsz))

    # persist raw capture
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    raw_path = CAPTURE_SAVE_DIR / f"capture_{ts}.png"
    image_pil.save(raw_path)

    # run inference using the same code path
    return _run_inference(
        image=image_pil,
        image_path=str(raw_path),
        threshold=threshold,
        imgsz=imgsz,
        save_annotated_image=save_annotated_image,
        annotated_image_path=annotated_image_path,
        return_image_base64=return_image_base64,
    )


@app.post("/capture_infer", response_model=InferResponse)
def api_capture_infer(
    threshold: float = CAPTURE_CONFIG.get("threshold", 0.5),
    imgsz: int = CAPTURE_CONFIG.get("imgsz", 640),
    save_annotated_image: bool = CAPTURE_CONFIG.get("save_annotated_image", True),
    annotated_image_path: Optional[str] = None,
    return_image_base64: bool = True,
) -> InferResponse:
    return capture_and_infer(
        threshold=threshold,
        imgsz=imgsz,
        save_annotated_image=save_annotated_image,
        annotated_image_path=annotated_image_path,
        return_image_base64=return_image_base64,
    )


# ----------------------------
# NEW: Folder watcher -> infer on new images
# ----------------------------
def _iter_new_images(watch_dir: Path, poll_sec=0.25):
    seen = set()
    while True:
        for p in watch_dir.glob("*"):
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"} and p not in seen:
                # small stabilization delay to ensure file is fully written
                try:
                    if time.time() - p.stat().st_mtime > 0.1:
                        seen.add(p)
                        yield p
                except FileNotFoundError:
                    # file might be removed quickly; skip
                    pass
        time.sleep(poll_sec)


def start_folder_watcher(
    threshold: float = FOLDER_WATCHER_CONFIG.get("threshold", 0.5),
    imgsz: int = FOLDER_WATCHER_CONFIG.get("imgsz", 640),
    save_annotated_image: bool = FOLDER_WATCHER_CONFIG.get("save_annotated_image", True),
):
    def worker():
        logger.info(f"Folder watcher active on: {WATCH_DIR}")
        for p in _iter_new_images(WATCH_DIR):
            try:
                with Image.open(p) as img:
                    img_rgb = img.convert("RGB")
                    _ = _run_inference(
                        image=img_rgb,
                        image_path=str(p),
                        threshold=threshold,
                        imgsz=imgsz,
                        save_annotated_image=save_annotated_image,
                        annotated_image_path=None,
                        return_image_base64=False,
                    )
            except Exception as exc:
                logger.exception(f"Folder watcher inference failed for {p}: {exc}")

    t = Thread(target=worker, name="folder_watcher", daemon=True)
    t.start()
    logger.info("Folder watcher thread started.")


# ----------------------------
# NEW: Modbus capture coil poller (Option A: UR is server; API is client)
# ----------------------------
def start_modbus_capture_poller():
    cfg = MODBUS_CONFIG or {}
    if not cfg.get("enabled", False):
        logger.info("Modbus disabled; capture poller not started.")
        return
    if not HAVE_PYMODBUS:
        logger.warning("Modbus disabled: PyModbus import failed or incompatible API. Error: %s", _PYMODBUS_IMPORT_ERROR)
        return

    host = cfg.get("host", "127.0.0.1")
    port = int(cfg.get("port", 502))
    unit_id = int(cfg.get("unit_id", 255))
    address_base = cfg.get("address_base", "zero_based")
    poll_ms = int(cfg.get("poll_ms", 100))
    capture_coil = (cfg.get("map", {}) or {}).get("coil_capture", None)

    if capture_coil is None:
        logger.info("No capture coil configured; poller not started.")
        return

    def worker():
        prev = False
        try:
            with URModbusClient(
                host=host, port=port, unit_id=unit_id, address_base=address_base
            ) as cli:
                logger.info(f"Modbus capture poller connected to {host}:{port}")
                kw = _kw_device_id(cli.client.read_coils)

                while True:
                    rr = cli.client.read_coils(
                        cli._addr(int(capture_coil)),
                        count=1,  # ✅ FIXED (keyword-only)
                        **{kw: cli.unit_id}
                    )

                    if rr and not rr.isError():
                        curr = bool(rr.bits[0])

                        if curr and not prev:
                            logger.info("Capture trigger detected via Modbus")
                            try:
                                capture_and_infer(
                                    threshold=CAPTURE_CONFIG.get("threshold", 0.5),
                                    imgsz=CAPTURE_CONFIG.get("imgsz", 640),
                                    save_annotated_image=CAPTURE_CONFIG.get("save_annotated_image", True),
                                    annotated_image_path=None,
                                    return_image_base64=False,
                                )
                            except Exception:
                                logger.exception("Capture+infer failed")

                        prev = curr
                    else:
                        logger.warning("Failed to read capture coil; retrying...")

                    time.sleep(max(0.01, poll_ms / 1000.0))

        except Exception:
            logger.exception("Modbus capture poller crashed; will not auto-restart.")

    t = Thread(target=worker, name="modbus_capture_poller", daemon=True)
    t.start()
    logger.info("Modbus capture poller thread started.")


# ----------------------------
# REST Endpoints (existing)
# ----------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
    }


@app.get("/model", response_model=ModelInfoResponse)
def get_model_info() -> ModelInfoResponse:
    try:
        return service.info()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read model information: {exc}") from exc


@app.get("/hardware")
def get_hardware_info() -> Dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
    }


@app.post("/model/load", response_model=ModelInfoResponse)
def load_model(request: LoadModelRequest) -> ModelInfoResponse:
    try:
        service.load_model(
            model_size=request.model_size,
            weights_path=request.weights_path,
            optimize_for_inference=request.optimize_for_inference,
            device_mode=request.device_mode,
        )
        return service.info()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {exc}") from exc


@app.post("/infer", response_model=InferResponse)
def infer(request: InferRequest) -> InferResponse:
    try:
        input_path = _resolve_existing_path(request.image_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        with Image.open(input_path) as image_file:
            image = image_file.convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to open image: {exc}") from exc

    return _run_inference(
        image=image,
        image_path=str(input_path),
        threshold=request.threshold,
        imgsz=request.imgsz,
        save_annotated_image=request.save_annotated_image,
        annotated_image_path=request.annotated_image_path,
        return_image_base64=request.return_image_base64,
    )


# ---------- Robust form parsing helpers ----------
def _to_bool(x: Any) -> bool:
    if isinstance(x, (list, tuple)) and x:
        x = x[0]
    return str(x).strip().lower() in {"1", "true", "yes", "on"}


def _first(x: Any) -> Any:
    return (x[0] if isinstance(x, (list, tuple)) and x else x)


async def _handle_upload_common(
    file: UploadFile,
    threshold: Any,
    imgsz: Any,
    save_annotated_image: Any,
    annotated_image_path: Optional[str],
    return_image_base64: Any,
) -> InferResponse:
    logger.info(f"UPLOAD raw: threshold type={type(threshold)} val={threshold}, imgsz type={type(imgsz)} val={imgsz}")

    # robust coercion
    threshold = _first(threshold)
    imgsz = _first(imgsz)
    save_annotated_image = _to_bool(save_annotated_image)
    return_image_base64 = _to_bool(return_image_base64)

    try:
        threshold = float(threshold)
    except Exception:
        raise HTTPException(status_code=400, detail="threshold must be a float in [0,1].")
    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(status_code=400, detail="threshold must be between 0.0 and 1.0")

    try:
        imgsz = int(imgsz)
    except Exception:
        raise HTTPException(status_code=400, detail="imgsz must be an integer.")
    if not 64 <= imgsz <= 8192:
        raise HTTPException(status_code=400, detail="imgsz must be between 64 and 8192")

    logger.info(f"UPLOAD coerced: threshold={threshold} ({type(threshold)}), imgsz={imgsz} ({type(imgsz)})")

    # read image
    try:
        payload = await file.read()
        with Image.open(io.BytesIO(payload)) as image_file:
            image = image_file.convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to open uploaded image: {exc}") from exc

    pseudo_path = file.filename or "uploaded_image.png"

    # run inference
    return _run_inference(
        image=image,
        image_path=pseudo_path,
        threshold=threshold,
        imgsz=imgsz,
        save_annotated_image=save_annotated_image,
        annotated_image_path=annotated_image_path,
        return_image_base64=return_image_base64,
    )


# New, robust endpoint (recommended)
@app.post("/infer/upload2", response_model=InferResponse)
async def infer_upload2(
    file: UploadFile = File(...),
    threshold: Any = Form(0.5),
    imgsz: Any = Form(640),
    save_annotated_image: Any = Form(False),
    annotated_image_path: Optional[str] = Form(None),
    return_image_base64: Any = Form(True),
) -> InferResponse:
    return await _handle_upload_common(
        file=file,
        threshold=threshold,
        imgsz=imgsz,
        save_annotated_image=save_annotated_image,
        annotated_image_path=annotated_image_path,
        return_image_base64=return_image_base64,
    )


# Backward-compatible route, uses the same robust handler
@app.post("/infer/upload", response_model=InferResponse)
async def infer_upload(
    file: UploadFile = File(...),
    threshold: Any = Form(0.5),
    imgsz: Any = Form(640),
    save_annotated_image: Any = Form(False),
    annotated_image_path: Optional[str] = Form(None),
    return_image_base64: Any = Form(True),
) -> InferResponse:
    return await _handle_upload_common(
        file=file,
        threshold=threshold,
        imgsz=imgsz,
        save_annotated_image=save_annotated_image,
        annotated_image_path=annotated_image_path,
        return_image_base64=return_image_base64,
    )


# ----------------------------
# Startup hooks
# ----------------------------
@app.on_event("startup")
def on_startup():
    # Optionally start folder watcher
    if FOLDER_WATCHER_CONFIG.get("enabled", False):
        start_folder_watcher(
            threshold=FOLDER_WATCHER_CONFIG.get("threshold", 0.5),
            imgsz=FOLDER_WATCHER_CONFIG.get("imgsz", 640),
            save_annotated_image=FOLDER_WATCHER_CONFIG.get("save_annotated_image", True),
        )
    # Start Modbus capture coil poller (if enabled)
    start_modbus_capture_poller()


# Optional: allow `python main.py` for local testing
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)