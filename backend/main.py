from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict
import re
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI(title="AI Vision Multi-Camera API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your frontend origin before production.
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

LINE_RATIO = 0.55
MAX_FRAME_BYTES = 5_000_000
CAMERA_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,48}$")


@dataclass
class CameraSession:
    model: object
    lock: Lock = field(default_factory=Lock)
    track_side: Dict[int, int] = field(default_factory=dict)
    track_seen: Dict[int, float] = field(default_factory=dict)
    counts: dict = field(default_factory=lambda: {"in": 0, "out": 0})
    last_seen: float = field(default_factory=time.time)
    active: bool = True


sessions: Dict[str, CameraSession] = {}
sessions_lock = Lock()


def validate_camera_id(camera_id: str) -> str:
    camera_id = (camera_id or "").strip()
    if not CAMERA_ID_RE.fullmatch(camera_id):
        raise HTTPException(
            status_code=400,
            detail="camera_id must be 1-48 characters: letters, numbers, _ or -",
        )
    return camera_id


def get_session(camera_id: str) -> CameraSession:
    with sessions_lock:
        session = sessions.get(camera_id)
        if session is None:
            # A separate YOLO object keeps tracker state isolated per camera.
            session = CameraSession(model=YOLO("yolov8n.pt"))
            sessions[camera_id] = session
        session.active = True
        session.last_seen = time.time()
        return session


def session_summary(camera_id: str, session: CameraSession) -> dict:
    inside = max(0, session.counts["in"] - session.counts["out"])
    return {
        "camera_id": camera_id,
        "in": session.counts["in"],
        "out": session.counts["out"],
        "inside": inside,
        "last_seen": session.last_seen,
        "active": session.active,
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "ai-vision-multi-camera", "version": "0.2.0"}


@app.get("/api/cameras")
def list_cameras():
    with sessions_lock:
        items = []
        for camera_id, session in sessions.items():
            with session.lock:
                # Mark idle cameras as inactive after 20 seconds.
                session.active = (time.time() - session.last_seen) < 20
                items.append(session_summary(camera_id, session))
    total_in = sum(x["in"] for x in items)
    total_out = sum(x["out"] for x in items)
    return {
        "cameras": items,
        "totals": {
            "in": total_in,
            "out": total_out,
            "inside": max(0, total_in - total_out),
            "camera_count": len(items),
            "active_cameras": sum(1 for x in items if x["active"]),
        },
    }


@app.post("/api/frame")
async def process_frame(
    camera_id: str = Form(...),
    file: UploadFile = File(...),
):
    camera_id = validate_camera_id(camera_id)
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please send a camera image.")
    raw = await file.read()
    if not raw or len(raw) > MAX_FRAME_BYTES:
        raise HTTPException(status_code=413, detail="Frame is empty or too large.")
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    session = get_session(camera_id)
    try:
        with session.lock:
            result = session.model.track(
                frame,
                persist=True,
                classes=[0],
                tracker="bytetrack.yaml",
                conf=0.35,
                imgsz=640,
                verbose=False,
            )[0]
            height, width = frame.shape[:2]
            line_y = int(height * LINE_RATIO)
            detections = []
            boxes = result.boxes

            if boxes is not None and boxes.xyxy is not None:
                coords_list = boxes.xyxy.cpu().numpy()
                ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(coords_list)
                confs = boxes.conf.cpu().tolist() if boxes.conf is not None else [0.0] * len(coords_list)

                now = time.time()
                for coords, track_id, confidence in zip(coords_list, ids, confs):
                    x1, y1, x2, y2 = [int(v) for v in coords]
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    if track_id is not None:
                        tid = int(track_id)
                        side = -1 if cy < line_y else 1
                        previous_side = session.track_side.get(tid)
                        if previous_side is not None and previous_side != side:
                            if previous_side == -1 and side == 1:
                                session.counts["in"] += 1
                            elif previous_side == 1 and side == -1:
                                session.counts["out"] += 1
                        session.track_side[tid] = side
                        session.track_seen[tid] = now
                    detections.append({
                        "id": int(track_id) if track_id is not None else None,
                        "box": [x1, y1, x2, y2],
                        "confidence": round(float(confidence), 3),
                    })

                # Keep tracker-side bookkeeping bounded.
                cutoff = time.time() - 180
                stale = [tid for tid, seen in session.track_seen.items() if seen < cutoff]
                for tid in stale:
                    session.track_seen.pop(tid, None)
                    session.track_side.pop(tid, None)

            session.last_seen = time.time()
            session.active = True
            summary = session_summary(camera_id, session)
            return {
                **summary,
                "width": width,
                "height": height,
                "line_y": line_y,
                "detections": detections,
            }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"AI processing failed: {exc}")


@app.post("/api/reset")
def reset_counts(camera_id: str = Form(...)):
    camera_id = validate_camera_id(camera_id)
    with sessions_lock:
        session = sessions.get(camera_id)
    if session is None:
        return {"camera_id": camera_id, "in": 0, "out": 0, "inside": 0}
    with session.lock:
        session.counts = {"in": 0, "out": 0}
        session.track_side.clear()
        session.track_seen.clear()
        return session_summary(camera_id, session)
