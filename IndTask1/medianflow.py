import cv2
import numpy as np


class MedianFlow:
    def __init__(self, n_points=100, fb_threshold=10.0, ncc_threshold=0.8):
        self.n_points = n_points
        self.fb_threshold = fb_threshold
        self.ncc_threshold = ncc_threshold
        self.prev_gray = None
        self.prev_points = None
        self.bbox = None

    def init(self, frame, bbox):
        self.bbox = np.array(bbox, dtype=np.float32)
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        x, y, w, h = [int(v) for v in bbox]
        self.prev_points = self._generate_points(x, y, w, h)

    def update(self, frame):
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Прямой optical flow
        next_points, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, curr_gray, self.prev_points, None
        )

        # Обратный optical flow (для проверки надежности)
        prev_points_back, status_back, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, self.prev_gray, next_points, None
        )

        # Фильтрация точек по forward-backward error
        fb_error = np.linalg.norm(self.prev_points - prev_points_back, axis=1)
        good_idx = (
            (status_fwd.ravel() == 1)
            & (status_back.ravel() == 1)
            & (fb_error < self.fb_threshold)
        )

        if np.sum(good_idx) < 10:
            return False, self.bbox

        prev_good = self.prev_points[good_idx]
        next_good = next_points[good_idx]

        # Вычисление смещения и изменения масштаба через медиану
        displacement = next_good - prev_good
        dx = np.median(displacement[:, 0])
        dy = np.median(displacement[:, 1])

        # Вычисление масштаба
        if len(prev_good) >= 2:
            prev_dists = np.linalg.norm(prev_good[:, None] - prev_good[None, :], axis=2)
            next_dists = np.linalg.norm(next_good[:, None] - next_good[None, :], axis=2)

            mask = prev_dists > 1e-5
            if np.sum(mask) > 0:
                scale_ratios = next_dists[mask] / prev_dists[mask]
                scale = np.median(scale_ratios)
            else:
                scale = 1.0
        else:
            scale = 1.0

        # Ограничение изменения масштаба
        scale = np.clip(scale, 0.8, 1.2)

        # Обновление bbox
        x, y, w, h = self.bbox
        cx, cy = x + w / 2, y + h / 2

        cx += dx
        cy += dy
        w *= scale
        h *= scale

        self.bbox = np.array([cx - w / 2, cy - h / 2, w, h], dtype=np.float32)

        # Обновление состояния для следующей итерации
        self.prev_gray = curr_gray
        x_new, y_new, w_new, h_new = [int(v) for v in self.bbox]
        self.prev_points = self._generate_points(x_new, y_new, w_new, h_new)

        return True, self.bbox

    def _generate_points(self, x, y, w, h):
        # Генерация равномерной сетки точек внутри bbox
        n_side = int(np.sqrt(self.n_points))
        xx = np.linspace(x + 5, x + w - 5, n_side)
        yy = np.linspace(y + 5, y + h - 5, n_side)
        xv, yv = np.meshgrid(xx, yy)
        points = np.column_stack([xv.ravel(), yv.ravel()]).astype(np.float32)
        return points
