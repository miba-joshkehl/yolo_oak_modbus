from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModbusMap:
    trigger_coil: int = 0
    busy_coil: int = 1
    ready_coil: int = 2
    error_coil: int = 3
    detection_count_reg: int = 0
    max_detections_reg: int = 1
    base_result_reg: int = 10
    heartbeat_reg: int = 2


@dataclass
class ModbusConfig:
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 1502
    unit_id: int = 1
    coils_size: int = 64
    holding_size: int = 512
    poll_interval_ms: int = 100
    map: ModbusMap = field(default_factory=ModbusMap)


@dataclass
class AppConfig:
    model_path: str = "models/yolo11n-obb.pt"
    confidence_threshold: float = 0.4
    image_size: int = 1024
    max_detections: int = 5
    capture_dir: str = "outputs/captures"
    annotated_dir: str = "outputs/annotated"
    use_oak_camera: bool = True
    oak_preview_width: int = 1280
    oak_preview_height: int = 720
    modbus: ModbusConfig = field(default_factory=ModbusConfig)



def _merge_dict(default: dict, override: dict) -> dict:
    merged = dict(default)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str = "config/yolo.config.json") -> AppConfig:
    cfg_path = Path(path)
    raw = {}
    if cfg_path.exists():
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))

    defaults = AppConfig()
    payload = _merge_dict(defaults.__dict__, raw)

    modbus_payload = _merge_dict(defaults.modbus.__dict__, payload.get("modbus", {}))
    modbus_map_payload = _merge_dict(defaults.modbus.map.__dict__, modbus_payload.get("map", {}))

    modbus_cfg = ModbusConfig(**{**modbus_payload, "map": ModbusMap(**modbus_map_payload)})
    return AppConfig(**{**payload, "modbus": modbus_cfg})
