import numpy as np


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    xi1 = max(a[0], b[0]); yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2]); yi2 = min(a[3], b[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    if inter == 0.0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


class _Track:
    _next_id = 1

    def __init__(self, bbox: np.ndarray, is_safe):
        self.id      = _Track._next_id
        _Track._next_id += 1
        self.bbox    = bbox.copy()
        self.is_safe = is_safe
        self.missed  = 0

    def matched(self, bbox: np.ndarray, is_safe) -> None:
        self.bbox    = bbox.copy()
        self.is_safe = is_safe
        self.missed  = 0


class CentroidTracker:
    """
    Tracker IoU-based ligero para personas entre frames de inferencia.

    update()    → llamar cuando el detector produce detecciones (inference frame).
    propagate() → llamar en frames intermedios (sin inferencia).
    reset()     → reinicia IDs y tracks (llamar al inicio de cada sesión).
    """

    def __init__(self, max_missed: int = 15, min_iou: float = 0.25):
        self.max_missed = max_missed
        self.min_iou    = min_iou
        self._tracks: list[_Track] = []

    def update(self, detections: list[tuple]) -> list[tuple]:
        """
        Args:
            detections: [(bbox [x1,y1,x2,y2] array-like, is_safe bool|None), ...]
        Returns:
            [(bbox ndarray, track_id int, is_safe bool|None), ...]
        """
        if not self._tracks:
            for bbox, is_safe in detections:
                self._tracks.append(_Track(np.asarray(bbox, float), is_safe))
        elif not detections:
            for t in self._tracks:
                t.missed += 1
        else:
            self._match(detections)

        self._prune()
        return self._snapshot()

    def propagate(self) -> list[tuple]:
        """Frames sin inferencia: mantiene posiciones e IDs sin cambios."""
        return self._snapshot()

    def reset(self) -> None:
        self._tracks.clear()

    # ── private ──────────────────────────────────────────────────────────────

    def _match(self, detections: list[tuple]) -> None:
        n_t, n_d = len(self._tracks), len(detections)
        iou_mat  = np.zeros((n_t, n_d))
        for i, t in enumerate(self._tracks):
            for j, (bbox, _) in enumerate(detections):
                iou_mat[i, j] = _iou(t.bbox, np.asarray(bbox, float))

        matched_t: set[int] = set()
        matched_d: set[int] = set()
        for i, j in sorted(
            ((i, j) for i in range(n_t) for j in range(n_d)),
            key=lambda x: iou_mat[x[0], x[1]], reverse=True
        ):
            if iou_mat[i, j] < self.min_iou:
                break
            if i in matched_t or j in matched_d:
                continue
            self._tracks[i].matched(np.asarray(detections[j][0], float), detections[j][1])
            matched_t.add(i)
            matched_d.add(j)

        for i, t in enumerate(self._tracks):
            if i not in matched_t:
                t.missed += 1

        for j, (bbox, is_safe) in enumerate(detections):
            if j not in matched_d:
                self._tracks.append(_Track(np.asarray(bbox, float), is_safe))

    def _prune(self) -> None:
        self._tracks = [t for t in self._tracks if t.missed <= self.max_missed]

    def _snapshot(self) -> list[tuple]:
        return [(t.bbox, t.id, t.is_safe) for t in self._tracks]
