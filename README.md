# YOLO Offline Inference API (RF-DETR-Compatible Interface)

This project is a **production-focused YOLO API** for Windows 11 systems with NVIDIA GPUs (A1000 or similar), designed to be interchangeable with the RF-DETR API contract from `Example API For Reference`.

It keeps the same external endpoints and response shape, so your calling app can switch between backends with minimal/no changes.

## What is included

- FastAPI application in `app/main.py`
- JSON config in `config/yolo.config.json`
- PowerShell + BAT scripts for setup and run
- Folder layout for easy copy/deploy to other offline production PCs

## 1) Recommended project structure

```text
YoloAPI/
├─ app/
│  └─ main.py
├─ config/
│  └─ yolo.config.json
├─ models/
│  └─ weights.pt                <-- place your YOLO weights here
├─ outputs/
│  └─ annotated/
├─ scripts/
│  ├─ setup_venv.ps1
│  ├─ setup_venv.bat
│  ├─ run_api.ps1
│  └─ run_api.bat
├─ requirements.txt
└─ README.md
```

## 2) Prerequisites (Windows 11)

- Python 3.10+
- NVIDIA driver with CUDA runtime compatible with your PyTorch build
- Internet access only during initial environment setup (package install)
- Your YOLO `.pt` weights copied locally

## 3) Put model weights in the correct location

Default location expected by config:
- `models/weights.pt`

Example absolute path:
- `C:\Miba\YoloAPI\models\weights.pt`

If you use another location, update `config/yolo.config.json`.

## 4) Configure the API (`config/yolo.config.json`)

Default file:

```json
{
  "default_model": "medium",
  "default_weights_path": "models/weights.pt",
  "default_device": "auto",
  "optimize_for_inference": true,
  "annotation_output_dir": "outputs/annotated"
}
```

### Config fields

- `default_model`: `medium` or `large`
  - Maps to YOLO aliases for runtime metadata compatibility.
- `default_weights_path`: path to your local `.pt` file.
- `default_device`: `auto`, `gpu`, or `cpu`
  - `auto` = CUDA if available, else CPU.
- `optimize_for_inference`: if `true`, model fusion is attempted.
- `annotation_output_dir`: default save location when saving annotated images.

## 5) Create virtual environment

### GPU install (recommended production)

```powershell
.\scripts\setup_venv.ps1 -Device gpu
```

### CPU install (fallback/testing)

```powershell
.\scripts\setup_venv.ps1 -Device cpu
```

### Optional offline wheelhouse install

```powershell
.\scripts\setup_venv.ps1 -Device gpu -Wheelhouse C:\wheelhouse
```

## 6) Run API

```powershell
.\scripts\run_api.ps1 -BindHost 0.0.0.0 -Port 8000
```

Swagger UI:
- `http://localhost:8000/docs`

## 7) Endpoint compatibility (same as RF-DETR example)

- `GET /health`
- `GET /hardware`
- `GET /model`
- `POST /model/load`
- `POST /infer`
- `POST /infer/upload`

### Notes for interchangeability

- `/infer` request/response schema matches the RF-DETR reference style.
- `/infer/upload` form fields are the same.
- `/model` and `/model/load` follow the same model metadata/load contract.
- Detection payload includes: class id/name, confidence, and xyxy box coordinates.

## 8) Deployment checklist for additional machines

1. Copy the full project folder to target machine.
2. Copy model file(s) to `models/` (or update config path).
3. Run setup script once while internet is available.
4. Verify `config/yolo.config.json` values.
5. Start API with `scripts/run_api.ps1`.
6. Validate with `GET /health` and one `/infer` call.

## 9) Offline runtime guidance

- Keep all model weights local (do not depend on auto-download in production).
- Keep API + calling app on same machine/LAN for lowest latency.
- Prefer `default_device: "gpu"` or `auto` on GPU-capable systems.
- Store annotated output on local SSD for best I/O performance.
