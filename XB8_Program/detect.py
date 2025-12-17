import time
import os
from ultralytics import YOLO

base_dir = os.path.dirname(os.path.abspath(__file__))

watch_file = os.path.join(base_dir, "ImgTemp", "Input.jpg")
watch_file_warmup = os.path.join(base_dir, "Warmup.jpg")

model = YOLO(os.path.join(base_dir, "XB8.pt"))  
last_mtime = None

if os.path.exists(watch_file):
    try:
        print("[INFO] Warming up model...", flush=True)
        model.predict(watch_file_warmup, imgsz=640, verbose=False)
    except Exception as e:
        print(f"[WARN] Warm-up skipped: {e}", flush=True)

while True:
    if os.path.exists(watch_file):
        try:
            mtime = os.path.getmtime(watch_file)
        except Exception:
            time.sleep(0.1)
            continue

        if last_mtime is None or mtime != last_mtime:
            time.sleep(0.2) 
            print(f"[INFO] Detected new image at {time.ctime(mtime)}", flush=True)
            results = model(watch_file)

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w = x2 - x1
                    h = y2 - y1
                    cls = int(box.cls[0])
                    name = model.names[cls]
                    print(f"BOX {name} {x1} {y1} {w} {h}", flush=True)

            last_mtime = mtime

    time.sleep(0.2)
