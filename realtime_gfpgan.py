"""Real-time GFPGAN face restoration with camera.

Usage:
    python realtime_gfpgan.py                          # USB camera 0
    python realtime_gfpgan.py 1                        # USB camera 1
    python realtime_gfpgan.py http://192.168.1.22:8080/video  # IP camera

Controls: +/- weight, r toggle restore, q quit
"""
import sys
import os
import time
import argparse

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gfpgan import GFPGANer

# ============================================================
# Config
# ============================================================
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "experiments", "pretrained_models", "GFPGANv1.4.pth")

WEIGHT = 0.5  # 0=more original, 1=more restored
UPSCALE = 1   # 1=same size (faster), 2=upscale
RESTORE_ON = True


def parse_source(source_str):
    """Parse camera source: integer for USB, string for IP/file."""
    try:
        return int(source_str)
    except ValueError:
        return source_str


def main():
    global WEIGHT, RESTORE_ON

    parser = argparse.ArgumentParser(description="Real-time GFPGAN face restoration")
    parser.add_argument("source", nargs="?", default="0",
                        help="Camera source: 0,1 for USB or http://ip:port/video for IP (default: 0)")
    args = parser.parse_args()

    camera_source = parse_source(args.source)
    source_label = f"USB camera {camera_source}" if isinstance(camera_source, int) else camera_source

    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading GFPGAN model...")
    restorer = GFPGANer(
        model_path=MODEL_PATH,
        upscale=UPSCALE,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None)
    print("[OK] Model loaded")

    print(f"Connecting to {source_label} ...")
    cap = cv2.VideoCapture(camera_source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"Error: Cannot open {source_label}")
        sys.exit(1)

    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame")
        sys.exit(1)

    h, w = frame.shape[:2]
    print(f"[OK] Stream: {w}x{h}")
    print("Keys: +/- weight | r toggle restore | q quit")

    fps_t = time.time()
    fps_n = 0
    fps_v = 0.0
    proc_ms = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize to reduce processing time
        scale = 480 / max(h, w)
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))

        if RESTORE_ON:
            t0 = time.perf_counter()
            try:
                _, _, output = restorer.enhance(
                    small,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                    weight=WEIGHT)
                if output is None:
                    output = small.copy()
            except Exception:
                output = small.copy()
            torch.cuda.synchronize()
            proc_ms = (time.perf_counter() - t0) * 1000
        else:
            output = small.copy()
            proc_ms = 0.0

        # FPS
        fps_n += 1
        if time.time() - fps_t >= 1.0:
            fps_v = fps_n / (time.time() - fps_t)
            fps_n = 0
            fps_t = time.time()

        # Display side-by-side
        dh = 480
        left = cv2.resize(small, (int(small.shape[1] * dh / small.shape[0]), dh))
        right = cv2.resize(output, (int(output.shape[1] * dh / output.shape[0]), dh))

        # Make same width
        min_w = min(left.shape[1], right.shape[1])
        left = left[:, :min_w]
        right = right[:, :min_w]

        cv2.putText(left, "Original", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        status = "ON" if RESTORE_ON else "OFF"
        cv2.putText(right, f"GFPGAN [{status}]  FPS:{fps_v:.1f}  {proc_ms:.0f}ms",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(right, f"Weight:{WEIGHT:.1f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("GFPGAN Real-time - IP Camera", np.hstack([left, right]))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in (ord('+'), ord('=')):
            WEIGHT = min(1.0, WEIGHT + 0.1)
        elif key == ord('-'):
            WEIGHT = max(0.0, WEIGHT - 0.1)
        elif key == ord('r'):
            RESTORE_ON = not RESTORE_ON

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
