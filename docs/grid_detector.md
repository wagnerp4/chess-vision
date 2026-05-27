# Chess Board Grid Detector

Detects a chess board in real time using an OAK-D Lite camera, overlays a perspective-correct 8×8 grid, and warps the board to a flat top-down view. Foundation for piece detection.

```bash
pip install opencv-python numpy depthai
uv run python scripts/grid_camera.py
```

---

## How It Works

### 1. Preprocessing — `preprocess(frame)`

**Grayscale → Gaussian Blur**

- **Gaussian Blur (5×5)**: smooths out pixel noise before corner detection. *Why:* Harris and `findChessboardCorners` look at intensity gradients. Noise creates fake gradients → fake corners.

---

### 2. Board Corner Detection — `find_board_corners(frame, blurred)`

**`cv2.findChessboardCornersSB` → extrapolate → 4 corners**

#### What it finds
An 8×8 chess board has **7×7 = 49 inner corners** — the points where four squares meet. `findChessboardCornersSB` finds all 49 in one call.

#### Why this algorithm
| Approach | Problem |
|----------|---------|
| Contour detection | Needs the wooden border to be clearly visible and not fill the frame |
| Hough transform | Picks up lines from shadows, table edges, decorative border — noisy and slow |
| Harris + convex hull | Harris finds the right corners but the hull includes background corners → wrong boundary |
| **`findChessboardCornersSB`** | Directly recognises the alternating square pattern, sub-pixel accurate, built into OpenCV |

`findChessboardCornersSB` uses the **saddle point method** — at every grid intersection, intensity goes up in one direction and down in the other, forming a saddle shape. The function finds these analytically. It is more robust to colour variation (works on brown/tan boards, not just black/white) than the classic `findChessboardCorners`.

#### From 49 inner corners to 4 board corners
The 49 inner corners don't include the board edge — they're one square inward. We extrapolate using the immediate neighbours:

```
board_top_left = 3×pts[0] − pts[1] − pts[7]
```

`pts[1]` is one step right of `pts[0]`, `pts[7]` is one step down. Subtracting them and adding back `pts[0]` moves exactly one square up-left. *Why use neighbours instead of averaging?* — This preserves perspective distortion. If the board is at an angle, the step size varies across the board; using local neighbours respects that.

Falls back to `findChessboardCorners` (classic method) if SB fails.

---

### 3. Corner Ordering — `order_corners(points)`

**Sum and difference trick → TL, TR, BR, BL**

```python
s    = points.sum(axis=1)   # x+y: smallest = top-left, largest = bottom-right
diff = points[:,1] - points[:,0]  # y-x: smallest = top-right, largest = bottom-left
```

*Why:* Works for any rotation or perspective — no assumption about which raw corner is "top".

---

### 4. Perspective Warp — `warp_board(frame, corners)`

**`cv2.getPerspectiveTransform` + `cv2.warpPerspective` → 800×800 flat view**

Maps the 4 detected corners to a perfect square. *Why warp?* The original camera view has perspective distortion — squares near the top appear smaller than squares near the camera. The warp makes all 64 squares the same pixel size, which is necessary for accurate piece detection later.

---

### 5. Grid Drawing — `draw_grid_from_corners` / `draw_grid_on_warped`

**Bilinear interpolation between corners**

```python
for i in range(1, 8):
    a = i / 8.0
    point_on_top_edge    = (1-a)*tl + a*tr
    point_on_bottom_edge = (1-a)*bl + a*br
    cv2.line(...)  # vertical line
```

*Why bilinear interpolation and not evenly spaced x-coordinates?* — Because perspective makes the squares unequal in pixel width. Interpolating between the corners follows the actual geometry of the board, so the grid lines align with real square boundaries.

---

### 6. Live Tracking — `live_grid_detection.py`

**Lucas-Kanade optical flow → full re-detection fallback**

Once the board is detected, instead of running `findChessboardCornersSB` on every frame:

1. **Lucas-Kanade (LK) optical flow** (`cv2.calcOpticalFlowPyrLK`): given the 4 corner positions from the last frame and the previous grayscale frame, find where those 4 pixel patches moved to in the current frame. *Why:* LK tracks actual pixel motion — faster than re-running corner detection, and smoother.

2. **LK failure fallback**: if any corner has tracking error > 15px or jumps > 150px, LK is considered failed and a full `findChessboardCornersSB` runs instead.

3. **Periodic re-anchor** (every 60 frames): LK can accumulate small errors each frame. Every 60 frames, a full detection re-anchors the corners to prevent slow drift.

**Image pyramid** (`maxLevel=3`): LK runs on a pyramid of 3 scaled-down versions of the image. *Why:* allows it to handle both large and small movements — large movements are caught at the coarse level, small movements are refined at the fine level.

---

## File Reference

| File | Role |
|------|------|
| `src/grid_detector.py` | `GridDetector` class — all detection, drawing, and warp logic |
| `scripts/grid_camera.py` | OAK-D Lite camera loop with LK tracking (grid only) |
| `scripts/chess_pipeline.py` | Full grid + YOLO pipeline |
| `scripts/test_grid_detection.py` | Run detection on a static image, save debug outputs |

For full history (what was tried, what failed, bugs fixed): see [documentation.md](documentation.md).

# Chess Board Grid Detector — Full Documentation

Real-time chess board detection and 8×8 grid overlay using an OAK-D Lite depth camera.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Debug Output Reference](#debug-output-reference)
- [What Worked](#what-worked)
- [What Didn't Work](#what-didnt-work)
- [Bugs Fixed](#bugs-fixed)
- [Improvement Plan](#improvement-plan)
- [Related Work](#related-work)

---

## Project Structure

```
vision/
├── src/
│   └── grid_detector.py          # Core detection engine (GridDetector class)
├── scripts/
│   ├── test_grid_detection.py    # Static image test (no camera needed)
│   └── live_grid_detection.py    # Real-time OAK-D Lite camera feed
├── data/
│   └── test_images/
│       └── board_001.png         # Test image
└── outputs/
    ├── grid_debug/               # Debug images saved by test script
    └── live_snapshots/           # Snapshots saved during live session
```

---

## Requirements

### Python Packages

| Package | Purpose |
|---------|---------|
| `opencv-python` | All image processing |
| `numpy` | Array math |
| `depthai` | OAK-D Lite camera SDK |

`pynput` was removed — it conflicts with the depthai signal handler on macOS and caused hard crashes. Keys are handled via `cv2.waitKey` instead.

### Hardware

- **OAK-D Lite** camera (USB-C)

---

## Installation

```bash
pip install opencv-python numpy depthai
```

---

## Usage

### Static image test (no camera)

```bash
cd vision
python scripts/test_grid_detection.py
```

### Live camera

```bash
cd vision
uv run python scripts/grid_camera.py
```

**Controls** (click the OpenCV window first to focus it):
- `Q` — quit
- `S` — save snapshot
- `R` — reset detection

---

## Debug Output Reference

| File | Contents |
|------|----------|
| `outputs/grid_debug/01_gray.jpg` | CLAHE-enhanced grayscale |
| `outputs/grid_debug/02_blurred.jpg` | Gaussian blurred input |
| `outputs/grid_debug/03_grid_on_original.jpg` | 8×8 grid overlay on original |
| `outputs/grid_debug/04_warped_board.jpg` | Flat 800×800 top-down view |
| `outputs/grid_debug/05_warped_grid.jpg` | Grid on warped board |

---

## What Worked

- **`findChessboardCornersSB`** — OpenCV's built-in chessboard corner detector finds all 49 inner grid intersections (7×7) precisely. Extrapolating outward by one square gives the correct playing area corners. Handles perspective, color contrast, and low lighting well.
- **Lucas-Kanade optical flow** — tracks the 4 corners as pixels between frames instead of re-running full detection every frame. Falls back to full detection if tracking fails, re-anchors every 60 frames to prevent drift.
- **CLAHE preprocessing** — local contrast enhancement handles uneven lighting far better than global histogram equalization.
- **No confirmation buffer** — removing the searching/tracking state machine made the system respond instantly.
- **`cv2.waitKey` for keys** — replacing `pynput` fixed hard crashes caused by signal handler conflicts between pynput and depthai on macOS.
- **0.5× detection scale + every 2 frames** — halves computation cost with negligible accuracy loss.

---

## What Didn't Work

- **Contour-based board border detection** — looked for the wooden border as a quadrilateral contour. Failed when the board filled most of the frame (area exceeded the 85% filter) and when the border blended with the background.
- **Chess pattern validation (brightness)** — checking alternating dark/light cells failed on brown/tan boards because the contrast was too low. Threshold had to be lowered so much it stopped being useful.
- **Hough line transform for grid lines** — too slow, too noisy. Picked up lines from shadows, table edges, and the decorative border. Made the grid "go crazy" with wrong lines across the frame.
- **Harris + convex hull** — Harris corners detected all grid intersections correctly, but the convex hull also captured corners from the decorative border and background, giving wrong board boundary corners.
- **`pynput` for keyboard input** — conflicts with depthai's signal handler on macOS, causing infinite signal loop crashes when pressing keys.
- **State machine (SEARCHING → TRACKING)** — the confirmation buffer required 5–8 stable consecutive detections. With Hough-based detection, corners shifted frame to frame causing it to never lock. Removed entirely.

---

## Bugs Fixed

| Bug | File | Fix |
|-----|------|-----|
| `image_area` undefined | `grid_detector.py` | Added `image_area = h * w` after `h, w = frame.shape[:2]` |
| `raise ValueError` at class scope | `grid_detector.py` | Fixed indentation from 4 to 8 spaces |
| Hardcoded relative paths | `test_grid_detection.py` | Replaced with `os.path.dirname(__file__)` paths |
| RGBA image not converted | `test_grid_detection.py` | Added `cv2.COLOR_BGRA2BGR` conversion after imread |
| `pynput` signal crash | `live_grid_detection.py` | Removed pynput entirely, use `cv2.waitKey` |
| `np.max` on empty array (LK) | `live_grid_detection.py` | Only compute `low_error` when `all_tracked` is True |

---

## Improvement Plan

### Next (piece detection)
- **YOLO object detection** — train YOLOv8 on chess piece images to classify each piece per square. Once the board corners are known, crop each of the 64 squares and run classification.
- **Square coordinate mapping** — from the 4 board corners + perspective matrix, compute the center pixel of every square (a1–h8) so piece detection can report "white knight on a1".

### Robustness
- **Multi-scale detection** — run `findChessboardCornersSB` at both 1× and 0.5× and use whichever succeeds.
- **Lighting adaptation** — auto-adjust CLAHE parameters based on mean frame brightness.

---

## Related Work

- **[ChessboardDetect](https://github.com/Elucidation/ChessboardDetect)** — multiple algorithms for chess board detection including saddle points and CNN tile classification
- **[TensorFlow Chessbot](https://github.com/Elucidation/tensorflow_chessbot)** — piece recognition using TensorFlow
- **[YOLOv8 Chess Piece Detection](https://www.youtube.com/watch?v=RXbtSwZsoEU)** — custom YOLO training for chess pieces
