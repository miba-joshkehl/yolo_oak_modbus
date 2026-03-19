from __future__ import annotations

import logging
import struct
import threading
from dataclasses import dataclass

from pymodbus.datastore import ModbusSequentialDataBlock, ModbusServerContext
from pymodbus.server import StartTcpServer

try:  # pymodbus >= 3.10
    from pymodbus.datastore import ModbusDeviceContext as _ModbusDeviceContext
except Exception:  # pragma: no cover - compatibility fallback
    from pymodbus.datastore import ModbusSlaveContext as _ModbusDeviceContext  # type: ignore

try:  # pymodbus >= 3.10
    from pymodbus.pdu.device import ModbusDeviceIdentification
except Exception:  # pragma: no cover - compatibility fallback
    from pymodbus.server.base import ModbusDeviceIdentification  # type: ignore

from app.config import ModbusConfig
from app.models import Detection

logger = logging.getLogger(__name__)


@dataclass
class RegisterLayout:
    trigger_coil: int
    busy_coil: int
    ready_coil: int
    error_coil: int
    detection_count_reg: int
    max_detections_reg: int
    base_result_reg: int
    heartbeat_reg: int


class ModbusBridge:
    """Hosts a Modbus TCP server and exposes helpers for pipeline state/results."""

    REGS_PER_DETECTION = 8  # 4 float32 values, each 2 x u16 registers

    def __init__(self, config: ModbusConfig, max_detections: int) -> None:
        self.config = config
        self.layout = RegisterLayout(**config.map.__dict__)
        self.max_detections = max_detections
        self._heartbeat = 0

        device_context = _ModbusDeviceContext(
            co=ModbusSequentialDataBlock(0, [0] * config.coils_size),
            hr=ModbusSequentialDataBlock(0, [0] * config.holding_size),
        )
        # single=True means unit-id is ignored and every request uses this context.
        self.context = ModbusServerContext(devices=device_context, single=True)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.config.enabled:
            logger.info("Modbus server disabled by configuration.")
            return
        if self._thread and self._thread.is_alive():
            return

        def _run_server() -> None:
            identity = ModbusDeviceIdentification()
            identity.VendorName = "yolo-oak-modbus"
            identity.ProductName = "Vision Modbus Bridge"
            identity.ModelName = "YOLO OBB"
            StartTcpServer(context=self.context, identity=identity, address=(self.config.host, self.config.port))

        self._thread = threading.Thread(target=_run_server, name="modbus-server", daemon=True)
        self._thread.start()
        logger.info("Started Modbus TCP server at %s:%s", self.config.host, self.config.port)

        self.set_coil(self.layout.busy_coil, False)
        self.set_coil(self.layout.ready_coil, False)
        self.set_coil(self.layout.error_coil, False)
        self.write_u16(self.layout.max_detections_reg, self.max_detections)

    def _device(self):
        return self.context[0]

    def get_coil(self, address: int) -> bool:
        return bool(self._device().getValues(1, address, count=1)[0])

    def set_coil(self, address: int, value: bool) -> None:
        self._device().setValues(1, address, [1 if value else 0])

    def write_u16(self, address: int, value: int) -> None:
        self._device().setValues(3, address, [int(value) & 0xFFFF])

    def read_trigger_and_clear(self) -> bool:
        trigger = self.get_coil(self.layout.trigger_coil)
        if trigger:
            self.set_coil(self.layout.trigger_coil, False)
        return trigger

    def publish_results(self, detections: list[Detection]) -> None:
        payload = detections[: self.max_detections]
        self.write_u16(self.layout.detection_count_reg, len(payload))

        reg = self.layout.base_result_reg
        zero_result = self._floats_to_regs(0.0, 0.0, 0.0, 0.0)
        for _ in range(self.max_detections):
            self._device().setValues(3, reg, zero_result)
            reg += self.REGS_PER_DETECTION

        reg = self.layout.base_result_reg
        for det in payload:
            packed = self._floats_to_regs(det.confidence, det.center_x, det.center_y, det.angle_deg)
            self._device().setValues(3, reg, packed)
            reg += self.REGS_PER_DETECTION

    def mark_busy(self, value: bool) -> None:
        self.set_coil(self.layout.busy_coil, value)

    def mark_ready(self, value: bool) -> None:
        self.set_coil(self.layout.ready_coil, value)

    def mark_error(self, value: bool) -> None:
        self.set_coil(self.layout.error_coil, value)

    def tick_heartbeat(self) -> int:
        self._heartbeat = (self._heartbeat + 1) % 65536
        self.write_u16(self.layout.heartbeat_reg, self._heartbeat)
        return self._heartbeat

    @staticmethod
    def _floats_to_regs(*values: float) -> list[int]:
        out: list[int] = []
        for value in values:
            b = struct.pack(">f", float(value))
            out.extend(struct.unpack(">HH", b))
        return out
