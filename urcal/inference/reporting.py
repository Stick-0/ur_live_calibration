"""Throttled terminal summaries shared by dual inference and control vision.

Duck-typed estimator/results and optional CalibrationFrame inputs avoid importing
models, robot drivers, camera libraries, or OpenCV for terminal formatting.
"""

import math


def _number(value, suffix=""):
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return "--"
    return f"{numeric:.1f}{suffix}" if math.isfinite(numeric) else "--"


def inference_status_line(fps, estimators, results, *, packet=None, now=None):
    """Describe this image; packet residuals are against original detections.

    EMPnP's internal residuals use corrected points, so they are intentionally
    not labelled as raw image error when no calibration packet is supplied.
    """
    fields = [f"[vision] FPS {_number(fps)}"]
    for estimator, result in zip(estimators, results):
        name = estimator.name.lower()
        solver = "PnP+plane" if result.base_plane and result.base_plane.get("applied") else "EMPnP"
        count = sum(bool(value) for value in result.valid)
        if packet is not None:
            count = getattr(packet, "valid_counts", {}).get(name, count)
        state = "OK" if result.ok else "FAIL"
        detail = f"{name.upper()} {solver} {state} keypoints={count}/{len(result.valid)}"
        if packet is not None:
            mean, maximum = packet.raw_mean_px.get(name), packet.raw_max_px.get(name)
            detail += f" raw mean/max={_number(mean)}/{_number(maximum)}px"
        if result.constraint_error:
            detail += " (" + " ".join(result.constraint_error.split()) + ")"
        fields.append(detail)
    if packet is not None:
        if now is not None:
            fields.append("capture age=" + _number(max(0., now - packet.captured_at) * 1000., "ms"))
        if packet.reason:
            fields.append("measurement: " + " ".join(packet.reason.split()))
    return " | ".join(fields)


class InferenceReporter:
    """Print at most one current-frame summary per interval (default one second).

    ``emit(now, fps, estimators, results, packet=None)`` returns the printed
    string or None when throttled. Inject ``writer(line)`` for headless tests.
    Status changes do not bypass the rate limit; existing runtime fault messages
    retain their own immediate paths.
    """

    def __init__(self, interval=1., writer=None):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("report interval must be positive and finite")
        self.interval, self.writer = float(interval), writer
        self._last_at = None

    def emit(self, now, fps, estimators, results, packet=None):
        now = float(now)
        if not math.isfinite(now):
            raise ValueError("report time must be finite")
        if self._last_at is not None and 0. <= now - self._last_at < self.interval:
            return None
        line = inference_status_line(fps, estimators, results, packet=packet, now=now)
        self._last_at = now
        if self.writer is None:
            print(line, flush=True)
        else:
            self.writer(line)
        return line
