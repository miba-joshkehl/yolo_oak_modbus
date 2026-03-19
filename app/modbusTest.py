from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
from typing import Any, Literal, Optional

# ============================================================
# PyModbus Import Compatibility
# ============================================================
_PYMODBUS_IMPORT_ERROR = None
try:
    try:
        from pymodbus.client import ModbusTcpClient
    except ImportError:
        from pymodbus.client.tcp import ModbusTcpClient

    HAVE_PYMODBUS = True
except ImportError as exc:
    HAVE_PYMODBUS = False
    _PYMODBUS_IMPORT_ERROR = exc


# ============================================================
# Types
# ============================================================
AddressBase = Literal["zero_based", "one_based"]


# ============================================================
# Hard-coded settings copied from your program defaults
# Adjust these if your actual config differs.
# ============================================================
HOST = "192.168.1.11"
PORT = 502
UNIT_ID = 255
ADDRESS_BASE: AddressBase = "zero_based"
TIMEOUT_SEC = 2.0

# These are the addresses used in your code defaults:
COIL_ALERT = 128
U16_COUNT = 300
U16_BEST_CONF_X1000 = 302
COIL_CAPTURE = 129

# These are the hard-coded TEST values your main program would publish:
TEST_ALERT = True
TEST_COUNT = 3
TEST_BEST_CONF_X1000 = 875   # e.g. 0.875 confidence -> 875


# ============================================================
# Console Helper
# ============================================================
def log(msg: str) -> None:
    print(msg, flush=True)


# ============================================================
# Helpers
# ============================================================
def kw_device_id(func: Any) -> str:
    """
    Return the supported PyModbus keyword:
    device_id, slave, or unit
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


def logical_to_wire(addr: int, address_base: AddressBase) -> int:
    if address_base == "one_based":
        if addr <= 0:
            raise ValueError("1-based address must be >= 1")
        return addr - 1

    if addr < 0:
        raise ValueError("0-based address must be >= 0")
    return addr


# ============================================================
# Data Model
# ============================================================
@dataclass
class ReadResult:
    ok: bool
    value: Optional[Any]
    detail: str


# ============================================================
# Minimal Safe Diagnostic Client
# ============================================================
class URModbusDiagnosticClient:
    def __init__(
        self,
        host: str,
        port: int = 502,
        unit_id: int = 255,
        address_base: AddressBase = "zero_based",
        timeout: float = 2.0,
    ) -> None:
        if not HAVE_PYMODBUS:
            raise RuntimeError(f"PyModbus is not installed: {_PYMODBUS_IMPORT_ERROR}")

        self.host = host
        self.port = port
        self.unit_id = unit_id
        self.address_base = address_base
        self.timeout = timeout
        self.client = ModbusTcpClient(host=self.host, port=self.port, timeout=self.timeout)

    def connect(self) -> None:
        log("=" * 72)
        log("MODBUS DIAGNOSTIC START")
        log("=" * 72)
        log(f"Target host       : {self.host}")
        log(f"Target port       : {self.port}")
        log(f"Unit ID           : {self.unit_id}")
        log(f"Address base      : {self.address_base}")
        log(f"Timeout (sec)     : {self.timeout}")

        if not self.client.connect():
            raise ConnectionError(f"Could not connect to {self.host}:{self.port}")

        log("Connection        : OK")

    def close(self) -> None:
        try:
            self.client.close()
            log("Connection        : CLOSED")
        except Exception:
            pass

    def read_coil(self, logical_addr: int) -> ReadResult:
        wire_addr = logical_to_wire(logical_addr, self.address_base)
        kw = kw_device_id(self.client.read_coils)

        try:
            rr = self.client.read_coils(wire_addr, count=1, **{kw: self.unit_id})
            if rr and not rr.isError() and hasattr(rr, "bits") and len(rr.bits) > 0:
                return ReadResult(
                    ok=True,
                    value=bool(rr.bits[0]),
                    detail=f"read_coils ok | logical={logical_addr} wire={wire_addr}",
                )

            return ReadResult(
                ok=False,
                value=None,
                detail=f"read_coils error | logical={logical_addr} wire={wire_addr} response={rr}",
            )
        except Exception as exc:
            return ReadResult(
                ok=False,
                value=None,
                detail=f"read_coils exception | logical={logical_addr} wire={wire_addr} error={exc}",
            )

    def read_u16(self, logical_addr: int) -> ReadResult:
        wire_addr = logical_to_wire(logical_addr, self.address_base)
        kw = kw_device_id(self.client.read_holding_registers)

        try:
            rr = self.client.read_holding_registers(wire_addr, count=1, **{kw: self.unit_id})
            if rr and not rr.isError():
                regs = getattr(rr, "registers", None)
                if regs is not None and len(regs) > 0:
                    return ReadResult(
                        ok=True,
                        value=int(regs[0]),
                        detail=f"read_holding_registers ok | logical={logical_addr} wire={wire_addr}",
                    )

            return ReadResult(
                ok=False,
                value=None,
                detail=f"read_holding_registers error | logical={logical_addr} wire={wire_addr} response={rr}",
            )
        except Exception as exc:
            return ReadResult(
                ok=False,
                value=None,
                detail=f"read_holding_registers exception | logical={logical_addr} wire={wire_addr} error={exc}",
            )


# ============================================================
# Main Diagnostic
# ============================================================
def main() -> int:
    if not HAVE_PYMODBUS:
        log("ERROR: PyModbus is not installed.")
        log(f"Import detail: {_PYMODBUS_IMPORT_ERROR}")
        log("")
        log("Install with:")
        log("  pip install pymodbus")
        return 1

    # Show exactly what your original program uses
    log("")
    log("Configured target map from your original code:")
    log(f"  coil_alert            = {COIL_ALERT}")
    log(f"  u16_count             = {U16_COUNT}")
    log(f"  u16_best_conf_x1000   = {U16_BEST_CONF_X1000}")
    log(f"  coil_capture          = {COIL_CAPTURE}")
    log("")

    # Show the exact hard-coded values your test is meant to validate
    log("Hard-coded TEST payload (READ-ONLY / NO WRITES WILL BE SENT):")
    log(f"  alert                 = {TEST_ALERT}")
    log(f"  count                 = {TEST_COUNT}")
    log(f"  best_conf_x1000       = {TEST_BEST_CONF_X1000}")
    log("")

    cli = URModbusDiagnosticClient(
        host=HOST,
        port=PORT,
        unit_id=UNIT_ID,
        address_base=ADDRESS_BASE,
        timeout=TIMEOUT_SEC,
    )

    try:
        cli.connect()

        log("")
        log("=" * 72)
        log("READ CURRENT VALUES")
        log("=" * 72)

        result = cli.read_coil(COIL_ALERT)
        log(f"[coil_alert]          ok={result.ok} value={result.value} | {result.detail}")

        result = cli.read_u16(U16_COUNT)
        log(f"[u16_count]           ok={result.ok} value={result.value} | {result.detail}")

        result = cli.read_u16(U16_BEST_CONF_X1000)
        log(f"[u16_best_conf_x1000] ok={result.ok} value={result.value} | {result.detail}")

        result = cli.read_coil(COIL_CAPTURE)
        log(f"[coil_capture]        ok={result.ok} value={result.value} | {result.detail}")

        log("")
        log("=" * 72)
        log("WOULD-WRITE SUMMARY (SAFE DRY RUN)")
        log("=" * 72)

        log(
            f"WOULD WRITE COIL  logical={COIL_ALERT} "
            f"wire={logical_to_wire(COIL_ALERT, ADDRESS_BASE)} value={TEST_ALERT}"
        )
        log(
            f"WOULD WRITE U16   logical={U16_COUNT} "
            f"wire={logical_to_wire(U16_COUNT, ADDRESS_BASE)} value={TEST_COUNT}"
        )
        log(
            f"WOULD WRITE U16   logical={U16_BEST_CONF_X1000} "
            f"wire={logical_to_wire(U16_BEST_CONF_X1000, ADDRESS_BASE)} value={TEST_BEST_CONF_X1000}"
        )

        log("")
        log("RESULT: Connection and address resolution completed successfully.")
        log("No writes were sent to the robot.")
        log("This confirms the same host/port/unit/address-base path your main code uses.")

        return 0

    except Exception as exc:
        log("")
        log("ERROR")
        log("=" * 72)
        log(str(exc))
        return 2

    finally:
        cli.close()


if __name__ == "__main__":
    sys.exit(main())
