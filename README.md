# AI Vision — People Counting System V0.2

Multi-device starter: iPhone and Android browsers can act as cameras and send frames to one shared backend. A dashboard aggregates camera statistics.

## Architecture
- Each phone/browser is assigned a unique `camera_id` (for example `shop-front`, `entrance-2`).
- The browser captures frames locally and uploads JPEG frames to the API.
- The backend runs YOLO person detection/tracking separately per camera session.
- Each camera has independent IN / OUT counters; `/api/cameras` aggregates totals.
- Frames are processed in memory; this starter does not write camera images or video to disk.

## Run on a computer
Requires Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Serve the frontend in another terminal:

```bash
python -m http.server 5500 --directory frontend
```

Open `http://localhost:5500` on the computer.

## Connect phones on the same Wi-Fi
1. Find the computer's LAN IP address (example `192.168.1.20`).
2. Serve the frontend on an HTTPS origin for mobile camera access. Mobile browsers require HTTPS for `getUserMedia` except on localhost. A trusted HTTPS tunnel or local HTTPS reverse proxy is recommended.
3. Open the frontend URL on each phone.
4. Set a different Camera ID on each device.
5. Set the API address to `https://YOUR-HOST:8000` (or route frontend and API through one HTTPS reverse proxy).
6. Press Check API, then Start camera.

Do not expose the development API directly to the public internet. Before production, add authentication, HTTPS, rate limits, per-user camera permissions, secure secrets, monitoring, and retention controls.

## Cloud / Docker
Build and run:

```bash
docker compose up --build
```

The API listens on port 8000. Put it behind a TLS reverse proxy and configure the frontend's API address to the public HTTPS API URL. Cloud inference needs sufficient CPU/GPU capacity and bandwidth; use a GPU instance for multiple high-resolution streams.

## Counting behavior
- Default virtual line is at 55% of the frame height.
- Top-to-bottom crossing counts as IN.
- Bottom-to-top crossing counts as OUT.
- A track's side transitions are counted, allowing a person to cross more than once while their tracking ID remains stable.
- INSIDE is calculated as max(0, IN - OUT). Start counting when the entrance is empty or use Reset for a known baseline.

## Important MVP notes
- YOLO weights are downloaded on first run.
- One model/tracker instance is held in memory per active camera ID. Keep the number of cameras appropriate for your server.
- Tracking IDs may change after long occlusion, camera movement, or poor lighting; this can affect counting accuracy.
- This is a prototype, not a certified occupancy or life-safety system.
- Do not use face recognition. Avoid recording or retaining identifiable video unless there is a lawful, necessary basis and appropriate notice/consent.
