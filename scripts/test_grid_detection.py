import os
import sys
from pathlib import Path

import cv2

VISION_ROOT = Path(__file__).resolve().parent.parent
if str(VISION_ROOT) not in sys.path:
    sys.path.insert(0, str(VISION_ROOT))

from src.grid_detector import GridDetector


_HERE      = os.path.dirname(__file__)
IMAGE_PATH = os.path.join(_HERE, "..", "data", "test_images", "board_001.png")
OUTPUT_DIR = os.path.join(_HERE, "..", "outputs", "grid_debug")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

    # Test image is RGBA — convert to BGR so all cv2 operations work correctly
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    detector = GridDetector(output_size=800)
    result = detector.process_image_debug(frame)

    # ── Always-available debug images ──────────────────────────────────
    cv2.imwrite(os.path.join(OUTPUT_DIR, "01_gray.jpg"),    result["gray"])
    cv2.imwrite(os.path.join(OUTPUT_DIR, "02_blurred.jpg"), result["blurred"])

    # ── Success-only outputs ────────────────────────────────────────────
    if result["success"]:
        cv2.imwrite(os.path.join(OUTPUT_DIR, "03_grid_on_original.jpg"), result["grid_original"])
        cv2.imwrite(os.path.join(OUTPUT_DIR, "04_warped_board.jpg"),     result["warped"])
        cv2.imwrite(os.path.join(OUTPUT_DIR, "05_warped_grid.jpg"),      result["warped_grid"])

        print("Board corners (TL, TR, BR, BL):")
        for name, pt in zip(("TL", "TR", "BR", "BL"), result["board_corners"]):
            print(f"  {name}: ({pt[0]:.1f}, {pt[1]:.1f})")

    print(f"\nSuccess     : {result['success']}")
    print(f"Method used : {result.get('method_used', 'n/a')}")

    if not result["success"]:
        print(f"Reason      : {result['error']}")

    print(f"\nSaved debug outputs to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
