"""anomaly_detector.py

Stage 2 of the SENTINEL pipeline: Z-score based anomaly detector that takes a
pre-fault telemetry window (list of parameter readings) and flags the anomalous
parameters, reducing thousands of possible channels down to the 5–20 most
significant.

Dependencies: math, statistics, typing, copy (stdlib only — no external packages).
"""

import math
import statistics
import copy
from typing import Optional

# ---------------------------------------------------------------------------
# Module-level nominal ranges constant
# ---------------------------------------------------------------------------

SATELLITE_NOMINAL_RANGES: dict[str, tuple[float, float]] = {
    "V_bat":                (28.0,   33.6),
    "SoC_pct":              (20.0,  100.0),
    "I_sa":                 (0.0,    12.0),
    "V_bus":                (26.6,   29.4),
    "Heater_power_W":       (0.0,    50.0),
    "RW_speed_rpm":         (-6000.0, 6000.0),
    "Gyro_rate_degs":       (0.0,     7.0),
    "Star_tracker_status":  (0.0,     0.0),
    "Sun_sensor_angle_deg": (0.0,    90.0),
    "Attitude_error_deg":   (0.0,     0.01),
    "OBC_temp_C":           (-10.0,  60.0),
    "CPU_load_pct":         (0.0,    70.0),
    "Memory_usage_MB":      (0.0,   500.0),
    "Watchdog_counter":     (0.0,  1000.0),
    "SEU_counter":          (0.0,     0.0),
    "Fault_register":       (0.0,     0.0),
    "Safe_mode_entry_count":(0.0,     5.0),
    "Transponder_lock":     (1.0,     1.0),
    "SNR_dB":               (10.0,   40.0),
    "Component_temp_C":     (-20.0,  65.0),
    "Heater_enable_flag":   (0.0,     1.0),
}


# ---------------------------------------------------------------------------
# ZScoreAnomalyDetector
# ---------------------------------------------------------------------------

class ZScoreAnomalyDetector:
    """Z-score based anomaly detector for satellite telemetry streams.

    Maintains a rolling baseline of nominal readings per parameter and flags
    readings whose Z-score exceeds a configurable threshold.  Designed to be
    used as Stage 2 of the SENTINEL diagnostic pipeline, reducing a large
    telemetry window to only the most statistically significant deviations.

    Parameters
    ----------
    z_threshold : float
        Number of standard deviations beyond which a reading is considered
        anomalous (default ``3.0``).
    window_size : int
        Maximum number of baseline readings kept per parameter.  When this
        limit is exceeded the oldest reading is dropped (FIFO, default ``10``).

    Examples
    --------
    >>> detector = ZScoreAnomalyDetector(z_threshold=3.0)
    >>> detector.fit_from_nominal_ranges({"V_bat": (28.0, 33.6)})
    >>> report = detector.detect([{"parameter": "V_bat", "value": 20.0,
    ...                            "nominal_min": 28.0, "nominal_max": 33.6}])
    >>> report["anomaly_count"]
    1
    """

    def __init__(self, z_threshold: float = 3.0, window_size: int = 10) -> None:
        """Initialise the detector with the given threshold and window size.

        Parameters
        ----------
        z_threshold : float
            Z-score threshold above which a reading is flagged (default ``3.0``).
        window_size : int
            Maximum baseline length per parameter (default ``10``).
        """
        self.z_threshold = z_threshold
        self.window_size = window_size
        # baseline_store: { parameter_name: [float, ...] }
        self._baseline: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # Baseline management
    # ------------------------------------------------------------------

    def update_baseline(self, parameter: str, value: float) -> None:
        """Add a nominal reading to the baseline for *parameter*.

        If the baseline already contains ``window_size`` entries the oldest
        entry is removed before the new value is appended (FIFO sliding window).

        Parameters
        ----------
        parameter : str
            The telemetry parameter name (e.g. ``"V_bat"``).
        value : float
            A nominal reading to add to the baseline.
        """
        if parameter not in self._baseline:
            self._baseline[parameter] = []
        buf = self._baseline[parameter]
        if len(buf) >= self.window_size:
            buf.pop(0)
        buf.append(value)

    def fit_from_nominal_ranges(
        self, nominal_ranges: dict[str, tuple[float, float]]
    ) -> None:
        """Bootstrap baselines from nominal operating ranges.

        For each parameter, ``window_size`` synthetic readings are generated
        using a Gaussian distribution whose mean is the range midpoint and
        whose standard deviation places the range boundaries at ±3 σ::

            midpoint = (min + max) / 2
            sigma    = (max - min) / 6

        When ``min == max`` (a constant-value parameter such as
        ``SEU_counter`` or ``Transponder_lock``), all synthetic readings equal
        ``midpoint`` and σ is treated as 0.

        Parameters
        ----------
        nominal_ranges : dict[str, tuple[float, float]]
            Mapping of ``parameter_name`` → ``(nominal_min, nominal_max)``.
        """
        import random as _random
        _rng = _random.Random(0)  # deterministic bootstrap; does not affect global state

        for parameter, (lo, hi) in nominal_ranges.items():
            midpoint = (lo + hi) / 2.0
            sigma    = (hi - lo) / 6.0
            self._baseline[parameter] = []
            for _ in range(self.window_size):
                if sigma == 0.0:
                    sample = midpoint
                else:
                    sample = _rng.gauss(midpoint, sigma)
                self._baseline[parameter].append(sample)

    # ------------------------------------------------------------------
    # Z-score computation
    # ------------------------------------------------------------------

    def compute_z_score(self, parameter: str, value: float) -> Optional[float]:
        """Compute the Z-score of *value* against the stored baseline.

        Returns ``None`` if fewer than 3 baseline readings are available for
        *parameter* (insufficient data for a reliable estimate).  Returns
        ``0.0`` if the baseline standard deviation is zero (constant signal).

        Parameters
        ----------
        parameter : str
            The telemetry parameter name.
        value : float
            The reading to score.

        Returns
        -------
        float or None
            The Z-score, or ``None`` when the baseline is too small.
        """
        buf = self._baseline.get(parameter, [])
        if len(buf) < 3:
            return None
        mu = statistics.mean(buf)
        try:
            sigma = statistics.stdev(buf)
        except statistics.StatisticsError:
            sigma = 0.0
        if sigma == 0.0:
            return 0.0
        return (value - mu) / sigma

    # ------------------------------------------------------------------
    # Core detection logic
    # ------------------------------------------------------------------

    def detect(self, telemetry_window: list[dict]) -> dict:
        """Detect anomalous parameters in a pre-fault telemetry window.

        Each reading dict must contain at minimum the keys ``"parameter"``,
        ``"value"``, ``"nominal_min"``, and ``"nominal_max"``.

        Algorithm
        ---------
        * Readings whose ``value`` equals the string ``"NaN"`` are
          automatically flagged with ``z_score = float("inf")``.
        * For numeric readings the Z-score is computed against the stored
          baseline.  If no baseline exists for a parameter the nominal range
          provided in the reading is used to derive a temporary one (same
          midpoint / σ = range/6 formula as ``fit_from_nominal_ranges``).
        * A reading is flagged when ``|Z| > z_threshold``.

        Parameters
        ----------
        telemetry_window : list[dict]
            The ``pre_fault_telemetry`` list from a crash dump.

        Returns
        -------
        dict
            ``{
              "anomalous_parameters": [...],
              "total_parameters_checked": int,
              "anomaly_count": int,
              "top_anomaly": str,
              "summary": str
            }``

            Each entry in ``anomalous_parameters`` contains:

            * ``parameter``       – parameter name
            * ``value``           – raw value (float or ``"NaN"``)
            * ``z_score``         – absolute Z-score (``inf`` for NaN readings)
            * ``anomaly_severity``– ``"CRITICAL"`` / ``"HIGH"`` / ``"MEDIUM"``
            * ``direction``       – ``"HIGH"``, ``"LOW"``, or ``"NaN"``
        """
        if not telemetry_window:
            return {
                "anomalous_parameters":    [],
                "total_parameters_checked": 0,
                "anomaly_count":            0,
                "top_anomaly":              "none",
                "summary":                  "No telemetry readings were provided.",
            }

        flagged: list[dict] = []
        total_checked = 0

        for reading in telemetry_window:
            parameter = reading.get("parameter", "unknown")
            value     = reading.get("value")
            nom_min   = reading.get("nominal_min")
            nom_max   = reading.get("nominal_max")

            total_checked += 1

            # ---- NaN readings are always anomalous ----
            if value == "NaN":
                flagged.append({
                    "parameter":        parameter,
                    "value":            "NaN",
                    "z_score":          float("inf"),
                    "anomaly_severity": "CRITICAL",
                    "direction":        "NaN",
                })
                continue

            # ---- Skip non-numeric values that aren't "NaN" ----
            if not isinstance(value, (int, float)):
                continue

            value = float(value)

            # ---- Derive a temporary baseline from nominal range if needed ----
            z = self.compute_z_score(parameter, value)
            if z is None and nom_min is not None and nom_max is not None:
                # Build a temporary baseline without mutating self._baseline
                lo, hi   = float(nom_min), float(nom_max)
                midpoint = (lo + hi) / 2.0
                sigma    = (hi - lo) / 6.0
                if sigma == 0.0:
                    z = 0.0
                else:
                    z = (value - midpoint) / sigma

            if z is None:
                # Still no baseline — cannot score
                continue

            abs_z = abs(z)
            if abs_z > self.z_threshold:
                severity = _severity(abs_z)
                direction = _direction(z)
                flagged.append({
                    "parameter":        parameter,
                    "value":            value,
                    "z_score":          abs_z,
                    "anomaly_severity": severity,
                    "direction":        direction,
                })

        # Sort by descending |z_score| (inf sorts naturally to the top)
        flagged.sort(key=lambda x: x["z_score"], reverse=True)

        top_anomaly = flagged[0]["parameter"] if flagged else "none"

        summary = _build_summary(flagged, total_checked)

        return {
            "anomalous_parameters":     flagged,
            "total_parameters_checked": total_checked,
            "anomaly_count":            len(flagged),
            "top_anomaly":              top_anomaly,
            "summary":                  summary,
        }

    # ------------------------------------------------------------------
    # Convenience wrapper
    # ------------------------------------------------------------------

    def filter_crash_dump(self, crash_dump: dict) -> dict:
        """Filter a crash dump to only its anomalous telemetry readings.

        Runs ``detect()`` on ``crash_dump["pre_fault_telemetry"]``, then
        returns a deep copy of the crash dump with two changes:

        * ``"pre_fault_telemetry"`` is replaced by only the anomalous readings
          (the subset whose ``parameter`` name appears in the anomaly report).
        * A new key ``"anomaly_report"`` is added containing the full output
          of ``detect()``.

        All other top-level keys are preserved unchanged.

        Parameters
        ----------
        crash_dump : dict
            A crash dump dict as returned by
            ``SatelliteFaultSimulator.generate_crash_dump``.

        Returns
        -------
        dict
            A filtered crash dump with ``"anomaly_report"`` added.
        """
        telemetry = crash_dump.get("pre_fault_telemetry", [])
        report    = self.detect(telemetry)

        # Collect the set of flagged parameter names for fast lookup
        flagged_params: set[str] = {
            entry["parameter"] for entry in report["anomalous_parameters"]
        }

        filtered_telemetry = [
            r for r in telemetry
            if r.get("parameter") in flagged_params or r.get("value") == "NaN"
        ]

        result = copy.deepcopy(crash_dump)
        result["pre_fault_telemetry"] = filtered_telemetry
        result["anomaly_report"]      = report
        return result


# ---------------------------------------------------------------------------
# Standalone helper function
# ---------------------------------------------------------------------------

def explain_anomalies(anomaly_report: dict) -> str:
    """Return a multi-line plain English explanation of an anomaly report.

    Formats the top-5 anomalous parameters and appends the one-sentence
    summary produced by ``detect()``.

    Parameters
    ----------
    anomaly_report : dict
        The dict returned by ``ZScoreAnomalyDetector.detect()``.

    Returns
    -------
    str
        A human-readable explanation string.
    """
    n_anomalies = anomaly_report.get("anomaly_count", 0)
    n_checked   = anomaly_report.get("total_parameters_checked", 0)
    top         = anomaly_report.get("top_anomaly", "none")
    summary     = anomaly_report.get("summary", "")
    params      = anomaly_report.get("anomalous_parameters", [])

    lines: list[str] = []
    lines.append(
        f"Anomaly detection found {n_anomalies} anomalous parameters "
        f"out of {n_checked} checked."
    )

    for entry in params[:5]:
        param     = entry["parameter"]
        value     = entry["value"]
        z         = entry["z_score"]
        direction = entry["direction"]
        severity  = entry["anomaly_severity"]

        if value == "NaN":
            value_str = "NaN"
        elif isinstance(value, float):
            value_str = f"{value:.3f}"
        else:
            value_str = str(value)

        z_str = "inf" if math.isinf(z) else f"{z:.1f}"
        lines.append(
            f"  - {param}: value={value_str}, Z={z_str} ({direction}, {severity})"
        )

    lines.append(f"Top anomaly: {top}")
    lines.append(summary)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _severity(abs_z: float) -> str:
    """Map an absolute Z-score to a severity label."""
    if math.isinf(abs_z) or abs_z > 6.0:
        return "CRITICAL"
    if abs_z > 4.0:
        return "HIGH"
    return "MEDIUM"


def _direction(z: float) -> str:
    """Map a signed Z-score to a direction label."""
    if z > 0:
        return "HIGH"
    if z < 0:
        return "LOW"
    return "NaN"


def _build_summary(flagged: list[dict], total: int) -> str:
    """Construct a one-sentence plain English summary."""
    if not flagged:
        return (
            f"No anomalous parameters detected across {total} telemetry readings."
        )
    n        = len(flagged)
    top      = flagged[0]["parameter"]
    top_z    = flagged[0]["z_score"]
    top_z_str = "inf" if math.isinf(top_z) else f"{top_z:.1f}"
    return (
        f"Detected {n} anomalous parameter{'s' if n != 1 else ''} out of "
        f"{total} readings; most significant deviation is {top} "
        f"(Z={top_z_str})."
    )


# ---------------------------------------------------------------------------
# __main__ demonstration block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from simulation.fault_simulator import SatelliteFaultSimulator
    import json as _json

    FAULT_TYPES = [
        "EPS_SOLAR_UNDERVOLT",
        "ADCS_GYRO_SEU",
        "OBC_WATCHDOG_OVERFLOW",
        "TCS_THERMAL_RUNAWAY",
        "COMMS_FAULT",
        "MULTI_CASCADE",
    ]

    sim = SatelliteFaultSimulator(seed=42)

    for scenario_id, fault_type in enumerate(FAULT_TYPES, start=1):
        crash_dump = sim.generate_crash_dump(fault_type, scenario_id=scenario_id)

        detector = ZScoreAnomalyDetector(z_threshold=3.0, window_size=10)
        detector.fit_from_nominal_ranges(SATELLITE_NOMINAL_RANGES)

        filtered = detector.filter_crash_dump(crash_dump)
        report   = filtered["anomaly_report"]

        print(f"Fault type: {fault_type}")
        print(explain_anomalies(report))
        print("---")
