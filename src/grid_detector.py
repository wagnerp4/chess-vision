import cv2
import numpy as np


class GridDetector:

    def __init__(self, output_size: int = 800):
        """
        Detects a chess board using findChessboardCornersSB (saddle point method).

        Finds the 7×7 inner corners of an 8×8 board, extrapolates to the 4 board
        corners, perspective-warps to a flat 800×800 view, and draws an 8×8 grid.
        """
        self.output_size = output_size

    def to_gray(self, frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def find_board_corners(self, frame: np.ndarray, gray: np.ndarray) -> np.ndarray:
        """
        Find the 4 board corners using OpenCV's chessboard corner detector.

        findChessboardCornersSB locates the 7×7 inner grid intersections of the
        8×8 board (49 points).  We then extrapolate outward by one square using
        each corner's local neighbours — this handles perspective correctly.

        Falls back to the classic findChessboardCorners if SB fails.
        """
        PATTERN = (7, 7)   # 7×7 inner corners for an 8×8 board

        ret, corners = cv2.findChessboardCornersSB(
            gray, PATTERN,
            cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE,
        )

        if not ret:
            ret, corners = cv2.findChessboardCorners(
                gray, PATTERN,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
            )

        if not ret or corners is None:
            raise ValueError("Chess board pattern not found")

        pts = corners.reshape(49, 2)

        # Extrapolate from the 4 inner corner points outward by one square.
        # Using immediate neighbours preserves perspective distortion correctly:
        #   board_tl = inner_tl + step_left + step_up
        #            = pts[0] + (pts[0]-pts[1]) + (pts[0]-pts[7])
        #            = 3*pts[0] - pts[1] - pts[7]
        tl = 3*pts[0]  - pts[1]  - pts[7]
        tr = 3*pts[6]  - pts[5]  - pts[13]
        bl = 3*pts[42] - pts[43] - pts[35]
        br = 3*pts[48] - pts[47] - pts[41]

        return self.order_corners(np.array([tl, tr, bl, br], dtype=np.float32))

    def order_corners(self, points: np.ndarray) -> np.ndarray:
        """Reorder 4 points → top-left, top-right, bottom-right, bottom-left."""
        points = np.array(points, dtype=np.float32)
        s    = points.sum(axis=1)
        diff = np.diff(points, axis=1).reshape(-1)
        return np.array([
            points[np.argmin(s)],
            points[np.argmin(diff)],
            points[np.argmax(s)],
            points[np.argmax(diff)],
        ], dtype=np.float32)

    def draw_grid_from_corners(self, frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Draw perspective-correct 8×8 grid on the original image."""
        debug = frame.copy()
        tl, tr, br, bl = corners
        cv2.polylines(debug, [np.array([tl, tr, br, bl], dtype=np.int32)], True, (0, 0, 255), 3)
        for i in range(1, 8):
            a = i / 8.0
            cv2.line(debug,
                     tuple(((1-a)*tl + a*tr).astype(int)),
                     tuple(((1-a)*bl + a*br).astype(int)),
                     (255, 0, 0), 2)
            cv2.line(debug,
                     tuple(((1-a)*tl + a*bl).astype(int)),
                     tuple(((1-a)*tr + a*br).astype(int)),
                     (0, 255, 0), 2)
        return debug

    def warp_board(self, frame: np.ndarray, corners: np.ndarray):
        """Perspective-warp the board to a flat 800×800 top-down view."""
        src = np.array(corners, dtype=np.float32)
        dst = np.array([
            [0, 0],
            [self.output_size, 0],
            [self.output_size, self.output_size],
            [0, self.output_size],
        ], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(frame, matrix, (self.output_size, self.output_size))
        return warped, matrix

    def draw_grid_on_warped(self, warped: np.ndarray) -> np.ndarray:
        """Draw a perfect pixel-aligned 8×8 grid on the warped board."""
        debug = warped.copy()
        sq = self.output_size // 8
        for i in range(9):
            p = i * sq
            cv2.line(debug, (p, 0), (p, self.output_size), (255, 0, 0), 2)
            cv2.line(debug, (0, p), (self.output_size, p), (0, 255, 0), 2)
        return debug

    def process_image_debug(self, frame: np.ndarray) -> dict:
        gray = self.to_gray(frame)
        result = {
            "success": False, "error": None,
            "gray": gray,
            "method_used": None,
        }
        try:
            board_corners = self.find_board_corners(frame, gray)
            warped, matrix = self.warp_board(frame, board_corners)
            result.update({
                "success":       True,
                "method_used":   "findChessboardCornersSB",
                "board_corners": board_corners,
                "grid_original": self.draw_grid_from_corners(frame, board_corners),
                "warped":        warped,
                "warped_grid":   self.draw_grid_on_warped(warped),
                "matrix":        matrix,
            })
        except ValueError as exc:
            result["error"] = str(exc)
        return result
