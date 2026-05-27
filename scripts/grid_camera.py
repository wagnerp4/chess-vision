import os
import sys
import cv2
import depthai as dai
import numpy as np

import sys
from pathlib import Path

VISION_ROOT = Path(__file__).resolve().parent.parent
if str(VISION_ROOT) not in sys.path:
    sys.path.insert(0, str(VISION_ROOT))

from src.grid_detector import GridDetector


# ── Performance ───────────────────────────────────────────────────────────────
DETECT_SCALE   = 0.5
DETECT_EVERY   = 2
REDETECT_EVERY = 60   # re-anchor LK tracking with a full detection every N frames

# ── Lucas-Kanade optical flow parameters ──────────────────────────────────────
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)
MAX_LK_ERROR = 15.0
MAX_LK_JUMP  = 150.0


def main():
    detector    = GridDetector(output_size=800)
    warped_grid = None

    # LK tracking state
    prev_gray             = None
    tracked_pts           = None   # (4, 1, 2) float32
    smoothed              = None   # (4, 2)    float32 — last known good corners
    frames_since_redetect = 0

    print("Connecting to OAK-D Lite...")
    print("Press Q to quit Press R to reset")

    device = dai.Device()

    with dai.Pipeline(device) as pipeline:
        sockets   = device.getConnectedCameras()
        cam       = pipeline.create(dai.node.Camera).build(sockets[0])
        rgb_queue = cam.requestOutput(
            size=(1280, 720),
            type=dai.ImgFrame.Type.BGR888p
        ).createOutputQueue()

        pipeline.start()
        print("OAK-D Lite connected!\n")

        frame_count = 0

        while pipeline.isRunning():
            video_in = rgb_queue.get()
            frame    = video_in.getCvFrame()
            frame_count += 1

            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display   = frame.copy()
            frames_since_redetect += 1

            lk_ok = False

            # ── Step 1: try LK tracking if we have a previous frame ───────────
            if prev_gray is not None and tracked_pts is not None:
                new_pts, status, error = cv2.calcOpticalFlowPyrLK(
                    prev_gray, curr_gray, tracked_pts, None, **LK_PARAMS
                )
                all_tracked = (status.flatten() == 1).all()
                low_error   = all_tracked and (np.max(error[status.flatten() == 1]) < MAX_LK_ERROR)
                if all_tracked and low_error:
                    drift = np.max(np.linalg.norm(new_pts.reshape(4, 2) - smoothed, axis=1))
                    if drift < MAX_LK_JUMP:
                        smoothed    = new_pts.reshape(4, 2).astype(np.float32)
                        tracked_pts = new_pts
                        lk_ok       = True

            # ── Step 2: run full detection if LK failed or it's time to re-anchor
            if not lk_ok or frames_since_redetect >= REDETECT_EVERY:
                if frame_count % DETECT_EVERY == 0:
                    small  = cv2.resize(frame, (0, 0), fx=DETECT_SCALE, fy=DETECT_SCALE)
                    result = detector.process_image_debug(small)
                    if result["success"]:
                        smoothed              = (result["board_corners"] / DETECT_SCALE).astype(np.float32)
                        tracked_pts           = smoothed.reshape(4, 1, 2)
                        frames_since_redetect = 0
                    elif not lk_ok:
                        smoothed    = None
                        tracked_pts = None

            prev_gray = curr_gray.copy()

            # ── Draw ──────────────────────────────────────────────────────────
            if smoothed is not None:
                display     = detector.draw_grid_from_corners(frame, smoothed)
                warped      = detector.warp_board(frame, smoothed)[0]
                warped_grid = detector.draw_grid_on_warped(warped)
                cv2.imshow("Warped board", warped_grid)
            else:
                cv2.putText(display, "No board detected",
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 220), 2)

            cv2.imshow("OAK-D Lite — Grid Detection", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Quitting...")
                break

            if key == ord("r"):
                smoothed, tracked_pts, prev_gray = None, None, None
                frames_since_redetect = 0
                print("Reset.")

            if key == ord("s") and smoothed is not None:
                snap_dir = os.path.join(
                    os.path.dirname(__file__), "..", "outputs", "live_snapshots"
                )
                os.makedirs(snap_dir, exist_ok=True)
                cv2.imwrite(os.path.join(snap_dir, "grid_original.jpg"), display)
                cv2.imwrite(os.path.join(snap_dir, "warped_grid.jpg"),   warped_grid)
                print("Snapshot saved.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
