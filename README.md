# YOLO OBB + OAK + Modbus TCP Bridge

This project is a complete refactor of the original API into a modular, production-friendly application that:

1. Hosts a **Modbus TCP server** for robot/client read/write.
2. Accepts a **trigger coil** from Modbus to capture an image from a Luxonis OAK camera.
3. Runs **YOLO OBB inference** (`yolo11n-obb`) on that frame.
4. Publishes inference outputs to Modbus holding registers:
   - confidence
   - center X
   - center Y
   - rotation angle (degrees)
5. Provides a **web app + API** for setup, testing, monitoring, and manual triggering.

---

## Architecture

```text
UR Robot / PLC (Modbus Client)
        |
        v
+------------------------------+
|  Modbus TCP Server (pymodbus)|
|  - Trigger coil              |
|  - Busy/Ready/Error coils    |
|  - Result holding registers  |
+---------------+--------------+
                |
                v
+------------------------------+
| Vision Pipeline              |
|  - OAK camera capture        |
|  - YOLO OBB inference        |
|  - Register packing          |
+---------------+--------------+
                |
                v
+------------------------------+
| FastAPI + Web UI             |
|  - /api/status               |
|  - /api/trigger              |
|  - /api/last-image           |
|  - Browser test interface    |
+------------------------------+
```

---

## Modbus register map (default)

### Coils
- `0` = `trigger_coil` (client writes `1` to request capture)
- `1` = `busy_coil` (server sets while pipeline is running)
- `2` = `ready_coil` (server sets when fresh result is available)
- `3` = `error_coil` (server sets when latest run failed)

### Holding Registers
- `0` = `detection_count_reg`
- `1` = `max_detections_reg`
- `2` = `heartbeat_reg` (increments periodically)
- `10+` = results area

Each detection uses 8 registers (4 float32 values x 2 registers each):
1. confidence
2. center_x
3. center_y
4. angle_deg

Detection block layout:
- Detection 0 starts at register 10
- Detection 1 starts at register 18
- Detection 2 starts at register 26
- ...

---

## Quick start

### 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2) Ensure model is available

Update `config/yolo.config.json`:
- `model_path`: path to your `yolo11n-obb.pt` or custom OBB model.

### 3) Run the app

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4) Open the web UI

- `http://localhost:8000/`

### 5) Robot / PLC handshake flow

1. Write `coil 0 = 1` (trigger).
2. Wait for `coil 1` (busy) to return `0` and `coil 2` (ready) to become `1`.
3. Read `holding reg 0` for `detection_count`.
4. Read detection data from `holding reg 10+`.

---

## API endpoints

- `GET /health` → liveness
- `GET /api/config` → active model + modbus map
- `GET /api/status` → pipeline status + latest run metadata
- `POST /api/trigger` → manual capture/inference trigger
- `GET /api/last-image` → latest annotated image
- `GET /` → web app

---

## Configuration

All runtime settings are in `config/yolo.config.json`.

Typical fields:
- `model_path`
- `confidence_threshold`
- `image_size`
- `max_detections`
- `capture_dir` / `annotated_dir`
- `use_oak_camera`
- `modbus.host` / `modbus.port`
- `modbus.map.*`

---

## Notes for customization

- To add more values to Modbus (class id, width/height, etc.), extend:
  - `app/services/modbus_server.py`
  - `app/models.py`
- To add alternative trigger modes, extend `VisionPipeline.trigger(...)`.
- To integrate coordinate transforms for robot picking, add transform logic after inference and before `publish_results(...)`.

---

## Project layout

```text
app/
  main.py                      # FastAPI entrypoint + lifecycle
  config.py                    # Typed config loader
  models.py                    # API models
  services/
    camera.py                  # OAK capture + fallback camera
    inference.py               # YOLO OBB wrapper
    modbus_server.py           # Modbus TCP server + register IO helpers
    pipeline.py                # Trigger loop + orchestration
  static/
    index.html                 # Web tester UI
config/
  yolo.config.json             # Runtime config
```
