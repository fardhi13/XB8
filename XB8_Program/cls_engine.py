import time
import os
import shutil
from ultralytics import YOLO

# =============================
# CONFIG
# =============================
LEFT_MODEL_PATH = "LEFT_REGION.pt"
RIGHT_MODEL_PATH = "RIGHT_REGION.pt"

INPUT_BASE = r"D:\CLS_INPUT"
OUTPUT_BASE = r"D:\CLS_OUTPUT"

# =============================
# LOAD MODEL (sekali saja)
# =============================
print("[INFO] Loading models...")
left_model = YOLO(LEFT_MODEL_PATH)
right_model = YOLO(RIGHT_MODEL_PATH)
print("[INFO] Models loaded")

# =============================
# HELPER: tunggu file siap
# =============================
def wait_file_ready(path, retries=10):
    for _ in range(retries):
        try:
            size = os.path.getsize(path)
            if size > 0:
                return True
        except:
            pass
        time.sleep(0.05)
    return False


# =============================
# PROCESS IMAGE
# =============================
def process_image(model, img_path, region):
    try:
        results = model(img_path)

        for r in results:
            probs = r.probs

            if probs is None:
                print(f"[WARN] No probs: {img_path}")
                return

            # ambil hasil terbaik
            cls_id = probs.top1
            conf = float(probs.top1conf)
            cls_name = model.names[cls_id].upper()

            print(f"[{region}] {os.path.basename(img_path)} => {cls_name} ({conf:.3f})")

            # folder tujuan
            save_dir = os.path.join(OUTPUT_BASE, region, cls_name)
            os.makedirs(save_dir, exist_ok=True)

            dest_path = os.path.join(save_dir, os.path.basename(img_path))

            # pindahkan file
            shutil.move(img_path, dest_path)

    except Exception as e:
        print(f"[ERROR] {img_path}: {e}")


# =============================
# MAIN LOOP
# =============================
print("[INFO] Classification engine started...")

while True:
    for region in ["LEFT", "RIGHT"]:
        input_dir = os.path.join(INPUT_BASE, region)

        if not os.path.exists(input_dir):
            continue

        files = os.listdir(input_dir)

        for file in files:
            full_path = os.path.join(input_dir, file)

            # hanya proses gambar final
            if not file.lower().endswith((".jpg", ".png", ".bmp")):
                continue

            # tunggu file selesai ditulis
            if not wait_file_ready(full_path):
                continue

            # pilih model sesuai region
            if region == "LEFT":
                process_image(left_model, full_path, "LEFT")
            else:
                process_image(right_model, full_path, "RIGHT")

    time.sleep(0.1)