from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import load_config
from app.services.camera import build_camera
from app.services.inference import YoloObbService
from app.services.modbus_server import ModbusBridge
from app.services.pipeline import VisionPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("yolo_oak_modbus")


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()

    camera = build_camera(
        preview_width=config.oak_preview_width,
        preview_height=config.oak_preview_height,
        use_oak=config.use_oak_camera,
    )
    inference = YoloObbService(
        model_path=config.model_path,
        threshold=config.confidence_threshold,
        image_size=config.image_size,
    )
    modbus = ModbusBridge(config.modbus, max_detections=config.max_detections)
    pipeline = VisionPipeline(
        camera=camera,
        inference_service=inference,
        modbus=modbus,
        capture_dir=Path(config.capture_dir),
        annotated_dir=Path(config.annotated_dir),
        poll_interval_ms=config.modbus.poll_interval_ms,
    )
    pipeline.start()

    app.state.config = config
    app.state.pipeline = pipeline
    logger.info("Service started.")
    try:
        yield
    finally:
        pipeline.stop()
        logger.info("Service stopped.")


app = FastAPI(
    title="YOLO OBB + OAK + Modbus",
    description="Capture images from OAK camera, run YOLO-OBB inference, and publish results over Modbus TCP.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/config")
def get_config() -> dict:
    cfg = app.state.config
    return {
        "model_path": cfg.model_path,
        "confidence_threshold": cfg.confidence_threshold,
        "image_size": cfg.image_size,
        "max_detections": cfg.max_detections,
        "modbus": {
            "host": cfg.modbus.host,
            "port": cfg.modbus.port,
            "map": cfg.modbus.map.__dict__,
        },
    }


@app.get("/api/status")
def get_status() -> dict:
    return app.state.pipeline.status().model_dump()


@app.post("/api/trigger")
def trigger_capture() -> dict:
    try:
        run = app.state.pipeline.trigger("api")
        return run.model_dump()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/last-image")
def get_last_image() -> FileResponse:
    status = app.state.pipeline.status()
    if not status.last_run or not status.last_run.annotated_image_path:
        raise HTTPException(status_code=404, detail="No annotated image available yet.")
    path = Path(status.last_run.annotated_image_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Annotated image file no longer exists.")
    return FileResponse(path)


@app.get("/")
def ui() -> FileResponse:
    return FileResponse("app/static/index.html")
