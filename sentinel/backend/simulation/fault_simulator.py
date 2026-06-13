"""
fault_simulator.py — Synthetic Satellite Fault Crash Dump Generator
====================================================================

This module provides a single, self-contained class, ``SatelliteFaultSimulator``,
for generating realistic synthetic crash dumps across six satellite fault types.
Its primary purpose is to produce high-quality, physically consistent training data
for fine-tuning a spacecraft anomaly diagnosis model.

No external dependencies are required; the module relies only on the Python
standard library (``random``, ``math``, ``datetime``, ``json``, ``typing``).

Supported fault types
---------------------
- ``EPS_SOLAR_UNDERVOLT``      : Solar-array failure leading to battery drain.
- ``ADCS_GYRO_SEU``            : Gyroscope SEU failure causing attitude divergence.
- ``OBC_WATCHDOG_OVERFLOW``    : Memory leak / watchdog-timeout sequence.
- ``TCS_THERMAL_RUNAWAY``      : Heater control failure causing thermal runaway.
- ``COMMS_TRANSPONDER_LOSS``   : Transponder lock-loss due to antenna mispointing.
- ``MULTI_CASCADE``            : Causal chain spanning ADCS, EPS, and TCS subsystems.

Quick start
-----------
    >>> from fault_simulator import SatelliteFaultSimulator
    >>> sim = SatelliteFaultSimulator(seed=42)
    >>> dump = sim.generate_crash_dump("EPS_SOLAR_UNDERVOLT", scenario_id=1)
    >>> truth = sim.get_ground_truth("EPS_SOLAR_UNDERVOLT")
"""

import random
import math
import datetime
import json
from typing import Any, Dict, List, Optional, Tuple


class SatelliteFaultSimulator:
    """Generates synthetic satellite crash dumps for machine-learning training.

    ``SatelliteFaultSimulator`` encapsulates all fault physics, telemetry
    generation, and ground-truth labelling logic needed to produce labelled
    crash-dump dictionaries.  Each crash dump represents a complete snapshot
    of satellite state at the time of a fault, including pre-fault telemetry,
    an event log, hardware state, and operating context.

    The simulator is seeded at construction time so that identical datasets
    can be reproduced across multiple runs.

    Usage
    -----
        >>> sim = SatelliteFaultSimulator(seed=42)
        >>> dump = sim.generate_crash_dump("TCS_THERMAL_RUNAWAY", scenario_id=7)
        >>> print(dump["fault_type"])
        TCS_THERMAL_RUNAWAY

    Supported fault types
    ---------------------
    ``EPS_SOLAR_UNDERVOLT``, ``ADCS_GYRO_SEU``, ``OBC_WATCHDOG_OVERFLOW``,
    ``TCS_THERMAL_RUNAWAY``, ``COMMS_TRANSPONDER_LOSS``, ``MULTI_CASCADE``.
    """

    # The six valid fault type identifiers used throughout the class.
    _VALID_FAULT_TYPES = frozenset({
        "EPS_SOLAR_UNDERVOLT",
        "ADCS_GYRO_SEU",
        "OBC_WATCHDOG_OVERFLOW",
        "TCS_THERMAL_RUNAWAY",
        "COMMS_TRANSPONDER_LOSS",
        "MULTI_CASCADE",
    })

    def __init__(self, seed: int = 42) -> None:
        """Initialise the simulator with a fixed random seed.

        Parameters
        ----------
        seed : int, optional
            Integer seed forwarded directly to ``random.seed``.  Using the
            same seed across runs guarantees bit-for-bit identical crash dumps,
            which is essential for reproducible dataset generation.  Defaults
            to ``42``.
        """
        random.seed(seed)

        # Per-instance Random object for reproducible, isolated generation.
        # Using random.Random(seed) ensures two instances with the same seed
        # produce identical output regardless of global RNG state.
        self._rng = random.Random(seed)

        # ------------------------------------------------------------------
        # Nominal operating ranges for all 21 canonical telemetry parameters.
        # Each entry is {"nominal_min": float, "nominal_max": float, "unit": str}.
        # Values are representative of a small LEO spacecraft in nominal ops.
        # ------------------------------------------------------------------

        # EPS — Electrical Power Subsystem
        self.V_bat = {
            "nominal_min": 28.0,
            "nominal_max": 33.0,
            "unit": "V",
        }
        self.SoC_pct = {
            "nominal_min": 60.0,
            "nominal_max": 95.0,
            "unit": "%",
        }
        self.I_sa = {
            "nominal_min": 3.5,
            "nominal_max": 6.5,
            "unit": "A",
        }
        self.V_bus = {
            "nominal_min": 27.5,
            "nominal_max": 32.5,
            "unit": "V",
        }

        # TCS — Thermal Control System
        self.Heater_power_W = {
            "nominal_min": 0.0,
            "nominal_max": 10.0,
            "unit": "W",
        }

        # ADCS — Attitude Determination and Control System
        self.RW_speed_rpm = {
            "nominal_min": -5000.0,
            "nominal_max": 5000.0,
            "unit": "rpm",
        }
        self.Gyro_rate_degs = {
            "nominal_min": -0.5,
            "nominal_max": 0.5,
            "unit": "deg/s",
        }
        self.Star_tracker_status = {
            "nominal_min": 0.0,
            "nominal_max": 0.0,
            "unit": "flag",
        }
        self.Sun_sensor_angle_deg = {
            "nominal_min": 0.0,
            "nominal_max": 90.0,
            "unit": "deg",
        }
        self.Attitude_error_deg = {
            "nominal_min": 0.0,
            "nominal_max": 1.0,
            "unit": "deg",
        }

        # OBC — On-Board Computer
        self.OBC_temp_C = {
            "nominal_min": 10.0,
            "nominal_max": 50.0,
            "unit": "°C",
        }
        self.CPU_load_pct = {
            "nominal_min": 10.0,
            "nominal_max": 70.0,
            "unit": "%",
        }
        self.Memory_usage_MB = {
            "nominal_min": 50.0,
            "nominal_max": 200.0,
            "unit": "MB",
        }
        self.Watchdog_counter = {
            "nominal_min": 0.0,
            "nominal_max": 200.0,
            "unit": "count",
        }
        self.SEU_counter = {
            "nominal_min": 0.0,
            "nominal_max": 5.0,
            "unit": "count",
        }

        # Fault management
        self.Fault_register = {
            "nominal_min": 0.0,
            "nominal_max": 0.0,
            "unit": "bitmask",
        }
        self.Safe_mode_entry_count = {
            "nominal_min": 0.0,
            "nominal_max": 3.0,
            "unit": "count",
        }

        # COMMS — Communications subsystem
        self.Transponder_lock = {
            "nominal_min": 1.0,
            "nominal_max": 1.0,
            "unit": "flag",
        }
        self.SNR_dB = {
            "nominal_min": 10.0,
            "nominal_max": 25.0,
            "unit": "dB",
        }

        # Thermal (component-level)
        self.Component_temp_C = {
            "nominal_min": -10.0,
            "nominal_max": 70.0,
            "unit": "°C",
        }

        # Heater control flag
        self.Heater_enable_flag = {
            "nominal_min": 0.0,
            "nominal_max": 0.0,
            "unit": "flag",
        }


    # ------------------------------------------------------------------
    # Internal canonical parameter map used by helpers to look up ranges
    # by name (mirrors the instance attributes set in __init__).
    # ------------------------------------------------------------------
    @property
    def _param_ranges(self) -> Dict[str, Dict[str, Any]]:
        """Return a mapping of canonical parameter name → nominal-range dict."""
        return {
            "V_bat": self.V_bat,
            "SoC_pct": self.SoC_pct,
            "I_sa": self.I_sa,
            "V_bus": self.V_bus,
            "Heater_power_W": self.Heater_power_W,
            "RW_speed_rpm": self.RW_speed_rpm,
            "Gyro_rate_degs": self.Gyro_rate_degs,
            "Star_tracker_status": self.Star_tracker_status,
            "Sun_sensor_angle_deg": self.Sun_sensor_angle_deg,
            "Attitude_error_deg": self.Attitude_error_deg,
            "OBC_temp_C": self.OBC_temp_C,
            "CPU_load_pct": self.CPU_load_pct,
            "Memory_usage_MB": self.Memory_usage_MB,
            "Watchdog_counter": self.Watchdog_counter,
            "SEU_counter": self.SEU_counter,
            "Fault_register": self.Fault_register,
            "Safe_mode_entry_count": self.Safe_mode_entry_count,
            "Transponder_lock": self.Transponder_lock,
            "SNR_dB": self.SNR_dB,
            "Component_temp_C": self.Component_temp_C,
            "Heater_enable_flag": self.Heater_enable_flag,
        }

    # ------------------------------------------------------------------
    # Shared telemetry helpers
    # ------------------------------------------------------------------

    def _noisy_value(self, parameter: str) -> float:
        """Return a nominally-centred value with 2 % Gaussian noise.

        Looks up the nominal range for *parameter*, computes the midpoint,
        and adds zero-mean Gaussian noise whose standard deviation is
        approximately 2 % of that midpoint (Requirement 5.1).

        Parameters
        ----------
        parameter : str
            One of the 21 canonical parameter names.

        Returns
        -------
        float
            Midpoint of the nominal range perturbed by Gaussian noise.
        """
        rng = self._param_ranges[parameter]
        midpoint = (rng["nominal_min"] + rng["nominal_max"]) / 2.0
        sigma = abs(midpoint) * 0.02 if midpoint != 0.0 else 0.01
        return self._rng.gauss(midpoint, sigma)

    def _make_reading(
        self,
        timestamp_offset: str,
        parameter: str,
        value: Any,
        anomalous: bool,
    ) -> Dict[str, Any]:
        """Build a single telemetry reading dict.

        Parameters
        ----------
        timestamp_offset : str
            Time offset before the fault event in the format ``"T-{seconds}s"``
            (e.g. ``"T-300s"``, ``"T-0s"``).  A ``ValueError`` is raised if
            the format is not matched.
        parameter : str
            Canonical parameter name (one of the 21 defined in Requirement 4.8).
            A ``ValueError`` is raised for unknown names.
        value : float or str
            Measured value; either a ``float`` or the sentinel ``"NaN"``.
        anomalous : bool
            ``True`` if the reading represents an off-nominal / fault condition.

        Returns
        -------
        dict
            Keys: ``timestamp_offset``, ``parameter``, ``value``, ``unit``,
            ``nominal_min``, ``nominal_max``, ``anomalous``.

        Raises
        ------
        ValueError
            If *timestamp_offset* does not match ``"T-{digits}s"`` or
            *parameter* is not one of the 21 canonical names.
        """
        import re
        if not re.fullmatch(r"T-\d+s", timestamp_offset):
            raise ValueError(
                f"timestamp_offset must match 'T-{{seconds}}s' (e.g. 'T-300s'), "
                f"got: {timestamp_offset!r}"
            )
        ranges = self._param_ranges
        if parameter not in ranges:
            raise ValueError(
                f"Unknown parameter {parameter!r}. "
                f"Must be one of: {sorted(ranges.keys())}"
            )
        rng = ranges[parameter]
        return {
            "timestamp_offset": timestamp_offset,
            "parameter": parameter,
            "value": value,
            "unit": rng["unit"],
            "nominal_min": rng["nominal_min"],
            "nominal_max": rng["nominal_max"],
            "anomalous": anomalous,
        }

    def _apply_nan_dropout(self, readings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply independent 5 % NaN dropout to non-anomalous readings.

        For each reading that is *not* already marked anomalous, there is a
        5 % probability that its ``value`` is replaced with ``"NaN"`` and its
        ``anomalous`` field is set to ``False`` (Requirements 5.2, 5.3).
        Anomalous readings are never modified.

        Parameters
        ----------
        readings : list of dict
            Telemetry reading dicts as produced by :meth:`_make_reading`.

        Returns
        -------
        list of dict
            The same list (mutated in-place) with dropout applied.
        """
        for reading in readings:
            if not reading["anomalous"]:
                if random.random() < 0.05:
                    reading["value"] = "NaN"
                    reading["anomalous"] = False
        return readings

    def _make_event(self, time_offset: str, source: str, message: str) -> Dict[str, Any]:
        """Build a single event log entry dict.

        Parameters
        ----------
        time_offset : str
            Time offset in the format ``"T-HH:MM:SS"`` (e.g. ``"T-00:04:21"``).
            A ``ValueError`` is raised if the format is not matched.
        source : str
            Name of the onboard software component that generated the event.
        message : str
            Human-readable event description.

        Returns
        -------
        dict
            Keys: ``time_offset``, ``source``, ``message``.

        Raises
        ------
        ValueError
            If *time_offset* does not match ``"T-HH:MM:SS"``.
        """
        import re
        if not re.fullmatch(r"T-\d{2}:\d{2}:\d{2}", time_offset):
            raise ValueError(
                f"time_offset must match 'T-HH:MM:SS' (e.g. 'T-00:04:21'), "
                f"got: {time_offset!r}"
            )
        return {
            "time_offset": time_offset,
            "source": source,
            "message": message,
        }

    def _make_hardware_state(
        self,
        last_reset_cause: str,
        seu_count: int,
        processes: List[str],
        memory_MB: float,
    ) -> Dict[str, Any]:
        """Build the hardware_state snapshot dict.

        Parameters
        ----------
        last_reset_cause : str
            Human-readable cause of the last system reset (e.g.
            ``"WATCHDOG_TIMEOUT"``).
        seu_count : int
            Total number of Single Event Upsets recorded since boot.
        processes : list of str
            Names of currently running OBC processes.
        memory_MB : float
            Current memory allocation in megabytes.

        Returns
        -------
        dict
            Keys: ``last_reset_cause``, ``SEU_event_count_since_boot``,
            ``running_processes``, ``memory_allocation_MB``.
        """
        return {
            "last_reset_cause": last_reset_cause,
            "SEU_event_count_since_boot": seu_count,
            "running_processes": processes,
            "memory_allocation_MB": memory_MB,
        }

    def _make_operating_context(
        self,
        eclipse_fraction: float,
        sun_angle_deg: float,
        mission_phase: str,
        minutes_since_contact: int,
        safe_mode_count: int,
    ) -> Dict[str, Any]:
        """Build the operating_context metadata dict.

        Parameters
        ----------
        eclipse_fraction : float
            Fraction of the current orbit spent in eclipse (0.0 = full sun,
            1.0 = full eclipse).
        sun_angle_deg : float
            Angle between the spacecraft body axis and the sun vector, in
            degrees.
        mission_phase : str
            Current mission phase.  Must be one of ``"nominal_science"``,
            ``"maneuver"``, or ``"commissioning"``; a ``ValueError`` is raised
            otherwise (Requirement 4.16).
        minutes_since_contact : int
            Minutes elapsed since the last successful ground-station contact.
        safe_mode_count : int
            Total number of safe-mode entries since launch.

        Returns
        -------
        dict
            Keys: ``orbital_position``, ``sun_angle_deg``, ``mission_phase``,
            ``minutes_since_last_ground_contact``, ``safe_mode_entry_count_total``.

        Raises
        ------
        ValueError
            If *mission_phase* is not one of the three valid values.
        """
        valid_phases = {"nominal_science", "maneuver", "commissioning"}
        if mission_phase not in valid_phases:
            raise ValueError(
                f"mission_phase must be one of {sorted(valid_phases)}, "
                f"got: {mission_phase!r}"
            )
        return {
            "orbital_position": f"eclipse_fraction: {eclipse_fraction:.1f}",
            "sun_angle_deg": sun_angle_deg,
            "mission_phase": mission_phase,
            "minutes_since_last_ground_contact": minutes_since_contact,
            "safe_mode_entry_count_total": safe_mode_count,
        }


    def generate_crash_dump(self, fault_type: str, scenario_id: int) -> dict:
        """Generate a synthetic satellite crash dump for the given fault type.

        Validates the fault type, generates a random timestamp in 2026, dispatches
        to the appropriate private fault generator, and assembles the final crash
        dump dictionary.

        Parameters
        ----------
        fault_type : str
            One of the six supported fault type identifiers:

            - ``"EPS_SOLAR_UNDERVOLT"``      : Solar-array failure leading to battery drain.
            - ``"ADCS_GYRO_SEU"``            : Gyroscope SEU failure causing attitude divergence.
            - ``"OBC_WATCHDOG_OVERFLOW"``    : Memory leak / watchdog-timeout sequence.
            - ``"TCS_THERMAL_RUNAWAY"``      : Heater control failure causing thermal runaway.
            - ``"COMMS_TRANSPONDER_LOSS"``   : Transponder lock-loss due to antenna mispointing.
            - ``"MULTI_CASCADE"``            : Causal chain spanning ADCS, EPS, and TCS.

        scenario_id : int
            An integer identifier that uniquely labels this crash dump within a dataset.

        Returns
        -------
        dict
            A crash dump dictionary with exactly the following 8 top-level keys:

            - ``scenario_id``        : The ``scenario_id`` argument (int).
            - ``timestamp``          : ISO 8601 datetime string in 2026 (e.g. ``"2026-03-15T14:22:09Z"``).
            - ``fault_type``         : The ``fault_type`` argument (str).
            - ``fault_register``     : Hex bitmask string (e.g. ``"0x00000002"``).
            - ``pre_fault_telemetry``: List of 8–15 telemetry reading dicts.
            - ``event_log``          : List of 4–8 event dicts.
            - ``hardware_state``     : Dict with reset cause, SEU count, processes, memory.
            - ``operating_context``  : Dict with orbital position, sun angle, mission phase, etc.

        Raises
        ------
        ValueError
            If ``fault_type`` is not one of the six valid fault type identifiers.
        NotImplementedError
            If the fault generator for ``fault_type`` has not yet been implemented.
        """
        if fault_type not in self._VALID_FAULT_TYPES:
            valid_sorted = sorted(self._VALID_FAULT_TYPES)
            raise ValueError(
                f"Invalid fault_type {fault_type!r}. "
                f"Must be one of: {valid_sorted}"
            )

        # Generate a random ISO 8601 timestamp within calendar year 2026.
        # Pick a random second within the year: 365 days * 24h * 60m * 60s
        seconds_in_2026 = 365 * 24 * 60 * 60
        random_second = self._rng.randint(0, seconds_in_2026 - 1)
        base = datetime.datetime(2026, 1, 1, 0, 0, 0)
        dt = base + datetime.timedelta(seconds=random_second)
        timestamp = dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Dispatch to the appropriate private fault generator.
        _dispatch = {
            "EPS_SOLAR_UNDERVOLT":      self._generate_eps_fault,
            "ADCS_GYRO_SEU":            self._generate_adcs_fault,
            "OBC_WATCHDOG_OVERFLOW":    self._generate_obc_fault,
            "TCS_THERMAL_RUNAWAY":      self._generate_tcs_fault,
            "COMMS_TRANSPONDER_LOSS":   self._generate_comms_fault,
            "MULTI_CASCADE":            self._generate_cascade_fault,
        }
        generator = _dispatch[fault_type]
        telemetry, event_log, hardware_state, operating_context, fault_register = generator()

        return {
            "scenario_id":        scenario_id,
            "timestamp":          timestamp,
            "fault_type":         fault_type,
            "fault_register":     fault_register,
            "pre_fault_telemetry": telemetry,
            "event_log":          event_log,
            "hardware_state":     hardware_state,
            "operating_context":  operating_context,
        }

    # ------------------------------------------------------------------
    # Private fault generators (stubs — implemented in tasks 6–11)
    # ------------------------------------------------------------------

    def _generate_eps_fault(self) -> Tuple[list, list, dict, dict, str]:
        """Generate EPS power fault data (solar-array failure / battery drain).

        Returns
        -------
        tuple
            ``(telemetry, event_log, hardware_state, operating_context, fault_register)``
        """
        fault_register = "0x00000002"

        # Build telemetry readings (8–15 readings covering a 300-second window)
        readings: List[Dict[str, Any]] = []

        # Pre-fault nominal readings at T-300s to T-181s
        for offset in ["T-300s", "T-240s", "T-210s"]:
            readings.append(self._make_reading(offset, "V_bat", self._noisy_value("V_bat"), False))
            readings.append(self._make_reading(offset, "SoC_pct", self._noisy_value("SoC_pct"), False))

        # I_sa drops to ~0 A at T-180s (solar array failure)
        readings.append(self._make_reading("T-180s", "I_sa", self._rng.gauss(0.0, 0.05), True))

        # V_bat drifting from ~31 V to ~24 V over 3 minutes
        v_bat_values = [31.0, 29.5, 27.5, 25.5, 24.2]
        time_offsets = ["T-180s", "T-120s", "T-90s", "T-60s", "T-0s"]
        for ts, vval in zip(time_offsets, v_bat_values):
            readings.append(self._make_reading(ts, "V_bat", self._rng.gauss(vval, 0.1), True))

        # SoC_pct falling from ~80 % to ~45 %
        soc_values = [80.0, 70.0, 60.0, 53.0, 45.5]
        for ts, sval in zip(time_offsets, soc_values):
            readings.append(self._make_reading(ts, "SoC_pct", self._rng.gauss(sval, 0.5), True))

        # Apply NaN dropout to non-anomalous readings
        self._apply_nan_dropout(readings)

        # Trim to 15 if we went over
        readings = readings[:15]

        # Event log (4–8 events reflecting EPS fault sequence)
        event_log = [
            self._make_event("T-00:03:00", "EPS_MONITOR", "Solar array current drop detected: I_sa near zero"),
            self._make_event("T-00:02:30", "EPS_MONITOR", "Low voltage warning: V_bat below 30 V threshold"),
            self._make_event("T-00:02:00", "POWER_MGMT", "Non-essential loads shed to conserve power"),
            self._make_event("T-00:01:30", "EPS_MONITOR", "Battery SoC falling below 60 %"),
            self._make_event("T-00:01:00", "POWER_MGMT", "Critical load shedding initiated"),
            self._make_event("T-00:00:30", "EPS_MONITOR", "Low voltage critical: V_bat below 25 V"),
            self._make_event("T-00:00:05", "OBC", "Safe mode entry triggered by EPS undervoltage"),
        ]

        hardware_state = self._make_hardware_state(
            last_reset_cause="UNDERVOLTAGE_RESET",
            seu_count=0,
            processes=["task_scheduler", "eps_monitor", "telemetry_mgr"],
            memory_MB=round(self._noisy_value("Memory_usage_MB"), 1),
        )

        operating_context = self._make_operating_context(
            eclipse_fraction=0.0,
            sun_angle_deg=round(self._noisy_value("Sun_sensor_angle_deg"), 1),
            mission_phase="nominal_science",
            minutes_since_contact=self._rng.randint(10, 45),
            safe_mode_count=self._rng.randint(0, 3),
        )

        return readings, event_log, hardware_state, operating_context, fault_register

    def _generate_adcs_fault(self) -> Tuple[list, list, dict, dict, str]:
        """Generate ADCS sensor fault data (gyroscope failure / attitude divergence).

        Returns
        -------
        tuple
            ``(telemetry, event_log, hardware_state, operating_context, fault_register)``
        """
        fault_register = "0x00000042"

        readings: List[Dict[str, Any]] = []

        # Pre-fault nominal readings
        for offset in ["T-120s", "T-90s"]:
            readings.append(self._make_reading(offset, "Gyro_rate_degs", self._noisy_value("Gyro_rate_degs"), False))
            readings.append(self._make_reading(offset, "Attitude_error_deg", self._noisy_value("Attitude_error_deg"), False))

        # SEU spike at ~T-62s causes gyro failure
        seu_spike = self._rng.randint(2, 4)
        readings.append(self._make_reading("T-62s", "SEU_counter", float(seu_spike), True))

        # Gyro NaN at/after T-60s
        for offset in ["T-60s", "T-30s", "T-0s"]:
            readings.append(self._make_reading(offset, "Gyro_rate_degs", "NaN", True))

        # Attitude divergence at fault time
        attitude_error = self._rng.uniform(5.0, 10.0)
        readings.append(self._make_reading("T-0s", "Attitude_error_deg", attitude_error, True))

        # RW saturation at fault time (~6000 rpm)
        rw_speed = self._rng.gauss(6000.0, 50.0)
        readings.append(self._make_reading("T-0s", "RW_speed_rpm", rw_speed, True))

        # Star tracker nominal pre-fault, nominal background readings
        readings.append(self._make_reading("T-120s", "RW_speed_rpm", self._noisy_value("RW_speed_rpm"), False))
        readings.append(self._make_reading("T-90s", "OBC_temp_C", self._noisy_value("OBC_temp_C"), False))

        self._apply_nan_dropout(readings)
        readings = readings[:15]

        event_log = [
            self._make_event("T-00:01:02", "ADCS_CTRL", "SEU burst detected: gyroscope register corruption"),
            self._make_event("T-00:01:00", "ADCS_CTRL", "Gyroscope telemetry invalid — switching to star tracker only"),
            self._make_event("T-00:00:45", "ADCS_CTRL", "Attitude error growing: star tracker rate extrapolation diverging"),
            self._make_event("T-00:00:30", "ADCS_CTRL", "Reaction wheel saturation imminent"),
            self._make_event("T-00:00:15", "ADCS_CTRL", "Reaction wheels saturated — attitude control lost"),
            self._make_event("T-00:00:05", "OBC", "Safe mode entry triggered by ADCS failure"),
        ]

        hardware_state = self._make_hardware_state(
            last_reset_cause="ADCS_SAFE_MODE_ENTRY",
            seu_count=seu_spike,
            processes=["task_scheduler", "adcs_ctrl", "telemetry_mgr"],
            memory_MB=round(self._noisy_value("Memory_usage_MB"), 1),
        )

        operating_context = self._make_operating_context(
            eclipse_fraction=round(self._rng.uniform(0.0, 0.6), 1),
            sun_angle_deg=round(self._noisy_value("Sun_sensor_angle_deg"), 1),
            mission_phase="nominal_science",
            minutes_since_contact=self._rng.randint(5, 60),
            safe_mode_count=self._rng.randint(0, 2),
        )

        return readings, event_log, hardware_state, operating_context, fault_register

    def _generate_obc_fault(self) -> Tuple[list, list, dict, dict, str]:
        """Generate OBC software fault data (memory leak / watchdog-timeout sequence).

        Returns
        -------
        tuple
            ``(telemetry, event_log, hardware_state, operating_context, fault_register)``
        """
        fault_register = "0x00000040"

        readings: List[Dict[str, Any]] = []

        # CPU load ramping toward 100 % from T-300s
        cpu_values = [45.0, 58.0, 72.0, 85.0, 94.0, 98.5]
        cpu_offsets = ["T-300s", "T-240s", "T-180s", "T-120s", "T-60s", "T-0s"]
        for offset, cpu in zip(cpu_offsets, cpu_values):
            readings.append(self._make_reading(offset, "CPU_load_pct", self._rng.gauss(cpu, 0.5), True))

        # Memory usage increasing monotonically (memory leak)
        mem_values = [80.0, 100.0, 128.0, 162.0, 198.0, 245.0]
        for offset, mem in zip(cpu_offsets, mem_values):
            readings.append(self._make_reading(offset, "Memory_usage_MB", mem, True))

        # Watchdog counter near overflow in final reading
        readings.append(self._make_reading("T-0s", "Watchdog_counter", 255.0, True))

        # Some nominal readings
        readings.append(self._make_reading("T-300s", "OBC_temp_C", self._noisy_value("OBC_temp_C"), False))
        readings.append(self._make_reading("T-180s", "V_bat", self._noisy_value("V_bat"), False))

        self._apply_nan_dropout(readings)
        readings = readings[:15]

        event_log = [
            self._make_event("T-00:05:00", "OBC_MONITOR", "Memory usage above nominal threshold: possible leak"),
            self._make_event("T-00:04:00", "OBC_MONITOR", "CPU load elevated: task scheduling impacted"),
            self._make_event("T-00:03:00", "OBC_MONITOR", "Memory allocation failure — heap fragmentation detected"),
            self._make_event("T-00:02:00", "WATCHDOG", "Watchdog counter approaching overflow"),
            self._make_event("T-00:01:00", "OBC_MONITOR", "CPU load critical: telemetry processing degraded"),
            self._make_event("T-00:00:10", "WATCHDOG", "Watchdog timeout — initiating system reset"),
        ]

        hardware_state = self._make_hardware_state(
            last_reset_cause="WATCHDOG_TIMEOUT",
            seu_count=0,
            processes=["task_scheduler", "obc_monitor", "watchdog", "telemetry_mgr"],
            memory_MB=245.0,
        )

        operating_context = self._make_operating_context(
            eclipse_fraction=round(self._rng.uniform(0.0, 1.0), 1),
            sun_angle_deg=round(self._noisy_value("Sun_sensor_angle_deg"), 1),
            mission_phase="nominal_science",
            minutes_since_contact=self._rng.randint(15, 90),
            safe_mode_count=self._rng.randint(0, 3),
        )

        return readings, event_log, hardware_state, operating_context, fault_register

    def _generate_tcs_fault(self) -> Tuple[list, list, dict, dict, str]:
        """Generate TCS thermal fault data (heater control failure / thermal runaway).

        Returns
        -------
        tuple
            ``(telemetry, event_log, hardware_state, operating_context, fault_register)``
        """
        fault_register = "0x00000010"

        readings: List[Dict[str, Any]] = []

        # Pre-fault nominal temperatures
        for offset in ["T-300s", "T-240s", "T-180s"]:
            readings.append(self._make_reading(offset, "Component_temp_C", self._noisy_value("Component_temp_C"), False))
            readings.append(self._make_reading(offset, "Heater_power_W", self._noisy_value("Heater_power_W"), False))

        # Thermal runaway — component temp exceeds 75 °C at fault time
        over_temp = self._rng.uniform(76.0, 95.0)
        readings.append(self._make_reading("T-0s", "Component_temp_C", over_temp, True))

        # Heater stuck on — elevated power
        elevated_heater_power = self._rng.uniform(15.0, 25.0)
        readings.append(self._make_reading("T-0s", "Heater_power_W", elevated_heater_power, True))

        # Heater enable flag stuck True (anomalous)
        readings.append(self._make_reading("T-0s", "Heater_enable_flag", 1.0, True))

        # Rising temp readings leading to fault
        temps = [40.0, 55.0, 67.0]
        for i, offset in enumerate(["T-120s", "T-60s", "T-30s"]):
            readings.append(self._make_reading(offset, "Component_temp_C", self._rng.gauss(temps[i], 0.5), True))

        # Nominal background readings
        readings.append(self._make_reading("T-300s", "OBC_temp_C", self._noisy_value("OBC_temp_C"), False))
        readings.append(self._make_reading("T-300s", "V_bat", self._noisy_value("V_bat"), False))

        self._apply_nan_dropout(readings)
        readings = readings[:15]

        event_log = [
            self._make_event("T-00:05:00", "TCS_MONITOR", "Component temperature rising above nominal band"),
            self._make_event("T-00:04:00", "TCS_MONITOR", "Heater control loop anomaly detected"),
            self._make_event("T-00:03:00", "TCS_MONITOR", "Heater enable flag stuck high — manual override attempted"),
            self._make_event("T-00:02:00", "TCS_MONITOR", "Component temperature critical: 67 C"),
            self._make_event("T-00:01:00", "TCS_MONITOR", "Thermal runaway — temperature exceeding 75 C limit"),
            self._make_event("T-00:00:10", "OBC", "Safe mode entry triggered by TCS over-temperature"),
        ]

        hardware_state = self._make_hardware_state(
            last_reset_cause="THERMAL_PROTECTION_RESET",
            seu_count=0,
            processes=["task_scheduler", "tcs_monitor", "telemetry_mgr"],
            memory_MB=round(self._noisy_value("Memory_usage_MB"), 1),
        )

        operating_context = self._make_operating_context(
            eclipse_fraction=round(self._rng.uniform(0.0, 1.0), 1),
            sun_angle_deg=round(self._noisy_value("Sun_sensor_angle_deg"), 1),
            mission_phase="nominal_science",
            minutes_since_contact=self._rng.randint(10, 60),
            safe_mode_count=self._rng.randint(0, 3),
        )

        return readings, event_log, hardware_state, operating_context, fault_register

    def _generate_comms_fault(self) -> Tuple[list, list, dict, dict, str]:
        """Generate COMMS fault data (transponder lock-loss / antenna mispointing).

        Returns
        -------
        tuple
            ``(telemetry, event_log, hardware_state, operating_context, fault_register)``
        """
        fault_register = "0x00000008"

        readings: List[Dict[str, Any]] = []

        # Pre-fault nominal COMMS readings
        for offset in ["T-300s", "T-240s", "T-180s"]:
            readings.append(self._make_reading(offset, "SNR_dB", self._noisy_value("SNR_dB"), False))
            readings.append(self._make_reading(offset, "Transponder_lock", 1.0, False))

        # SNR degrading and dropping below 5 dB at fault time
        snr_values = [12.0, 8.0, 4.5]
        for offset, snr in zip(["T-120s", "T-60s", "T-0s"], snr_values):
            anomalous = snr < 5.0
            readings.append(self._make_reading(offset, "SNR_dB", snr, anomalous))

        # Transponder lock lost at fault time
        readings.append(self._make_reading("T-0s", "Transponder_lock", 0.0, True))

        # Star tracker blinded (antenna attitude issue)
        readings.append(self._make_reading("T-0s", "Star_tracker_status", 1.0, True))

        # Nominal background
        readings.append(self._make_reading("T-300s", "OBC_temp_C", self._noisy_value("OBC_temp_C"), False))
        readings.append(self._make_reading("T-300s", "V_bat", self._noisy_value("V_bat"), False))

        self._apply_nan_dropout(readings)
        readings = readings[:15]

        event_log = [
            self._make_event("T-00:05:00", "COMMS_MGMT", "SNR degradation detected on primary link"),
            self._make_event("T-00:04:00", "ADCS_CTRL", "Antenna pointing error growing — star tracker blinded"),
            self._make_event("T-00:03:00", "COMMS_MGMT", "SNR below 8 dB threshold — switching to backup link"),
            self._make_event("T-00:02:00", "COMMS_MGMT", "Backup link also degraded — possible mispointing"),
            self._make_event("T-00:01:00", "COMMS_MGMT", "Transponder lock lost — no uplink"),
            self._make_event("T-00:00:10", "OBC", "Communication blackout — entering contingency mode"),
        ]

        hardware_state = self._make_hardware_state(
            last_reset_cause="COMMS_CONTINGENCY_RESET",
            seu_count=0,
            processes=["task_scheduler", "comms_mgmt", "telemetry_mgr"],
            memory_MB=round(self._noisy_value("Memory_usage_MB"), 1),
        )

        operating_context = self._make_operating_context(
            eclipse_fraction=round(random.uniform(0.0, 1.0), 1),
            sun_angle_deg=round(self._noisy_value("Sun_sensor_angle_deg"), 1),
            mission_phase="nominal_science",
            minutes_since_contact=random.randint(20, 120),
            safe_mode_count=random.randint(0, 2),
        )

        return readings, event_log, hardware_state, operating_context, fault_register

    def _generate_cascade_fault(self) -> Tuple[list, list, dict, dict, str]:
        """Generate multi-system cascade fault data (ADCS → EPS → TCS causal chain).

        The fault unfolds in three causally ordered phases:

        * **Phase 1 — ADCS** (~T-120s): An SEU burst corrupts gyroscope registers,
          causing invalid gyroscope telemetry and attitude divergence which leads to
          solar-array off-pointing.
        * **Phase 2 — EPS** (~T-45s): Solar-array off-pointing reduces array current
          to near zero, initiating battery drain and voltage decline.
        * **Phase 3 — TCS** (~T-30s): Loss of heater control leads to thermal runaway.

        Returns
        -------
        tuple
            ``(telemetry, event_log, hardware_state, operating_context, fault_register)``
        """
        # Bits 1, 2, and 6 set: 0x00000002 | 0x00000004 | 0x00000040 = 0x00000046
        fault_register = "0x00000046"

        readings: List[Dict[str, Any]] = []

        # ------------------------------------------------------------------
        # Phase 1 — ADCS (earliest in timeline, ~T-120s onwards)
        # ------------------------------------------------------------------

        # SEU spike at T-62s causes gyroscope register corruption
        seu_spike = self._rng.randint(2, 4)
        readings.append(self._make_reading("T-62s", "SEU_counter", float(seu_spike), True))

        # Pre-fault nominal Gyro_rate_degs at T-120s and T-90s
        readings.append(self._make_reading("T-120s", "Gyro_rate_degs", self._noisy_value("Gyro_rate_degs"), False))
        readings.append(self._make_reading("T-90s", "Gyro_rate_degs", self._noisy_value("Gyro_rate_degs"), False))

        # Gyro NaN from T-60s onwards (sensor failed)
        for offset in ["T-60s", "T-30s", "T-0s"]:
            readings.append(self._make_reading(offset, "Gyro_rate_degs", "NaN", True))

        # Attitude divergence at T-60s
        attitude_error = self._rng.uniform(5.0, 10.0)
        readings.append(self._make_reading("T-60s", "Attitude_error_deg", attitude_error, True))

        # Reaction wheel saturation at T-60s (~6000 rpm)
        rw_speed = self._rng.gauss(6000.0, 50.0)
        readings.append(self._make_reading("T-60s", "RW_speed_rpm", rw_speed, True))

        # ------------------------------------------------------------------
        # Phase 2 — EPS (after ADCS anomaly in timeline, ~T-45s)
        # ------------------------------------------------------------------

        # Pre-fault nominal solar-array current at T-90s
        readings.append(self._make_reading("T-90s", "I_sa", self._noisy_value("I_sa"), False))

        # I_sa drops to ~0 A at T-45s (solar array off-pointed)
        readings.append(self._make_reading("T-45s", "I_sa", self._rng.gauss(0.05, 0.02), True))

        # V_bat declining: ~29 V at T-45s, ~25.5 V at T-0s
        readings.append(self._make_reading("T-45s", "V_bat", self._rng.gauss(29.0, 0.1), True))
        readings.append(self._make_reading("T-0s", "V_bat", self._rng.gauss(25.5, 0.1), True))

        # SoC_pct fallen to ~52% at T-0s
        readings.append(self._make_reading("T-0s", "SoC_pct", self._rng.gauss(52.0, 0.5), True))

        # ------------------------------------------------------------------
        # Phase 3 — TCS (at ~T-30s)
        # ------------------------------------------------------------------

        # Component temperature exceeding 75 °C
        over_temp = self._rng.uniform(76.0, 90.0)
        readings.append(self._make_reading("T-30s", "Component_temp_C", over_temp, True))

        # ------------------------------------------------------------------
        # Background nominal readings (pad to 8+ total)
        # ------------------------------------------------------------------
        readings.append(self._make_reading("T-120s", "OBC_temp_C", self._noisy_value("OBC_temp_C"), False))
        readings.append(self._make_reading("T-120s", "V_bat", self._noisy_value("V_bat"), False))

        self._apply_nan_dropout(readings)
        readings = readings[:15]

        # ------------------------------------------------------------------
        # Event log — 7 entries reflecting the causal chain
        # ------------------------------------------------------------------
        event_log = [
            self._make_event("T-00:02:02", "ADCS_CTRL", "SEU burst detected — gyroscope register corruption"),
            self._make_event("T-00:02:00", "ADCS_CTRL", "Gyroscope telemetry invalid — attitude divergence beginning"),
            self._make_event("T-00:01:30", "ADCS_CTRL", "Solar array off-pointing due to attitude loss"),
            self._make_event("T-00:01:00", "EPS_MONITOR", "Solar array current dropping — battery drain initiated"),
            self._make_event("T-00:00:45", "EPS_MONITOR", "Low voltage warning — non-essential loads shed"),
            self._make_event("T-00:00:30", "TCS_MONITOR", "Heater control failure — thermal runaway beginning"),
            self._make_event("T-00:00:10", "OBC", "Multi-system cascade failure — safe mode entry"),
        ]

        hardware_state = self._make_hardware_state(
            last_reset_cause="MULTI_SYSTEM_SAFE_MODE",
            seu_count=seu_spike,
            processes=["task_scheduler", "adcs_ctrl", "eps_monitor", "tcs_monitor", "telemetry_mgr"],
            memory_MB=round(self._noisy_value("Memory_usage_MB"), 1),
        )

        operating_context = self._make_operating_context(
            eclipse_fraction=0.0,
            sun_angle_deg=round(self._noisy_value("Sun_sensor_angle_deg"), 1),
            mission_phase="nominal_science",
            minutes_since_contact=self._rng.randint(10, 60),
            safe_mode_count=self._rng.randint(1, 4),
        )

        return readings, event_log, hardware_state, operating_context, fault_register


    def get_ground_truth(self, fault_type: str) -> dict:
        """Return expert-labelled ground truth for the given fault type.

        Provides the canonical diagnosis data for a fault type, including the root
        cause classification, the responsible subsystem, an ordered causal chain,
        a diagnostic confidence score, a recommended recovery action sequence, and
        an overall risk level.

        Parameters
        ----------
        fault_type : str
            One of the six supported fault type identifiers:

            - ``"EPS_SOLAR_UNDERVOLT"``      : Solar-array failure leading to battery drain.
            - ``"ADCS_GYRO_SEU"``            : Gyroscope SEU failure causing attitude divergence.
            - ``"OBC_WATCHDOG_OVERFLOW"``    : Memory leak / watchdog-timeout sequence.
            - ``"TCS_THERMAL_RUNAWAY"``      : Heater control failure causing thermal runaway.
            - ``"COMMS_TRANSPONDER_LOSS"``   : Transponder lock-loss due to antenna mispointing.
            - ``"MULTI_CASCADE"``            : Causal chain spanning ADCS, EPS, and TCS.

        Returns
        -------
        dict
            A ground truth dictionary with exactly the following keys:

            - ``root_cause_classification`` (str): Equal to ``fault_type``.
            - ``root_cause_subsystem`` (str): The primary subsystem responsible
              (``"EPS"``, ``"ADCS"``, ``"OBC"``, ``"TCS"``, ``"COMMS"``,
              or ``"MULTI"``).
            - ``causal_chain`` (list of str): Ordered sequence of fault events
              from initial trigger to final system state.
            - ``confidence`` (float): Diagnostic confidence in [0.0, 1.0].
              ``"MULTI_CASCADE"`` returns ~0.65 to reflect ambiguity.
            - ``recovery_action_sequence`` (list of str): Ordered recovery steps
              recommended by ground operators.
            - ``risk_level`` (str): One of ``"LOW"``, ``"MEDIUM"``, ``"HIGH"``.

        Raises
        ------
        ValueError
            If ``fault_type`` is not one of the six valid fault type identifiers.
        """
        if fault_type not in self._VALID_FAULT_TYPES:
            valid_sorted = sorted(self._VALID_FAULT_TYPES)
            raise ValueError(
                f"Invalid fault_type {fault_type!r}. "
                f"Must be one of: {valid_sorted}"
            )

        _ground_truth_data = {
            "EPS_SOLAR_UNDERVOLT": {
                "root_cause_classification": "EPS_SOLAR_UNDERVOLT",
                "root_cause_subsystem": "EPS",
                "causal_chain": [
                    "Solar array output current drops to near zero due to panel damage or shadowing",
                    "Battery begins discharging to compensate for missing generation",
                    "Bus voltage declines as battery state of charge falls",
                    "Non-essential loads shed by power management software",
                    "Battery SoC falls below critical threshold triggering undervoltage protection",
                    "OBC enters safe mode via undervoltage reset",
                ],
                "confidence": 0.95,
                "recovery_action_sequence": [
                    "Verify solar array deployment and orientation via attitude telemetry",
                    "Command spacecraft to sun-pointing attitude to maximise array illumination",
                    "Monitor V_bat and SoC recovery over next orbit",
                    "Re-enable non-essential loads once SoC exceeds 70 %",
                    "Inspect solar array health and perform I-V curve characterisation",
                    "Resume nominal operations after confirming stable power balance",
                ],
                "risk_level": "HIGH",
            },
            "ADCS_GYRO_SEU": {
                "root_cause_classification": "ADCS_GYRO_SEU",
                "root_cause_subsystem": "ADCS",
                "causal_chain": [
                    "Ionising radiation causes an SEU burst in gyroscope memory registers",
                    "Gyroscope rate telemetry becomes invalid (reads NaN)",
                    "ADCS control loop switches to star-tracker-only attitude estimation",
                    "Rate extrapolation from star tracker alone diverges over time",
                    "Attitude error grows beyond reaction wheel authority",
                    "Reaction wheels saturate at maximum speed (~6000 rpm)",
                    "Attitude control is lost and OBC enters safe mode",
                ],
                "confidence": 0.92,
                "recovery_action_sequence": [
                    "Power-cycle gyroscope unit to clear SEU-corrupted registers",
                    "Verify gyroscope telemetry validity after reset",
                    "Desaturate reaction wheels using magnetorquers",
                    "Restore attitude control using full sensor suite",
                    "Re-point spacecraft to nominal science attitude",
                    "Monitor SEU counter for further radiation events",
                ],
                "risk_level": "HIGH",
            },
            "OBC_WATCHDOG_OVERFLOW": {
                "root_cause_classification": "OBC_WATCHDOG_OVERFLOW",
                "root_cause_subsystem": "OBC",
                "causal_chain": [
                    "A software process begins leaking heap memory due to unfreed allocations",
                    "Memory usage grows monotonically over several minutes",
                    "CPU load increases as the scheduler contends for shrinking resources",
                    "Memory allocation failures cause task scheduling delays",
                    "Watchdog counter approaches overflow as critical tasks miss deadlines",
                    "Watchdog timer expires and triggers a system reset",
                ],
                "confidence": 0.90,
                "recovery_action_sequence": [
                    "Allow watchdog reset to complete and verify system boot",
                    "Confirm all critical tasks restart and watchdog counter returns to nominal",
                    "Review OBC process logs to identify the leaking software component",
                    "Upload patched flight software to address the memory leak",
                    "Monitor memory usage and CPU load over subsequent orbits",
                    "Re-enable science operations after 24 hours of stable OBC health",
                ],
                "risk_level": "MEDIUM",
            },
            "TCS_THERMAL_RUNAWAY": {
                "root_cause_classification": "TCS_THERMAL_RUNAWAY",
                "root_cause_subsystem": "TCS",
                "causal_chain": [
                    "Heater control loop fails, leaving heater enable flag stuck high",
                    "Heater continues to apply power above the component operating temperature limit",
                    "Component temperature rises into the anomalous range",
                    "Thermal runaway ensues as heat cannot be rejected fast enough",
                    "Component temperature exceeds 75 °C critical limit",
                    "Thermal protection circuit triggers a safe mode reset",
                ],
                "confidence": 0.88,
                "recovery_action_sequence": [
                    "Immediately command heater disable override to cut heater power",
                    "Monitor Component_temp_C until it returns below 60 °C",
                    "Inspect heater control software for stuck-flag bug",
                    "Upload corrected heater control parameters or software patch",
                    "Verify thermal model predictions against recovered telemetry",
                    "Resume nominal heating schedule after component temperature stabilises",
                ],
                "risk_level": "HIGH",
            },
            "COMMS_TRANSPONDER_LOSS": {
                "root_cause_classification": "COMMS_TRANSPONDER_LOSS",
                "root_cause_subsystem": "COMMS",
                "causal_chain": [
                    "Antenna pointing error develops due to an ADCS anomaly",
                    "Star tracker is blinded, further degrading attitude knowledge",
                    "Signal-to-noise ratio on the primary link begins to fall",
                    "SNR drops below the 5 dB acquisition threshold",
                    "Transponder loses carrier lock — uplink communication severed",
                    "Spacecraft enters contingency mode awaiting ground contact recovery",
                ],
                "confidence": 0.85,
                "recovery_action_sequence": [
                    "Wait for scheduled ground station pass in contingency mode",
                    "Send emergency uplink on omni-directional antenna to restore contact",
                    "Command spacecraft to Earth-pointing attitude to recover transponder lock",
                    "Verify SNR recovers above 10 dB on primary link",
                    "Investigate root cause of the antenna pointing error",
                    "Clear star tracker blind flag and restore nominal ADCS operation",
                ],
                "risk_level": "MEDIUM",
            },
            "MULTI_CASCADE": {
                "root_cause_classification": "MULTI_CASCADE",
                "root_cause_subsystem": "MULTI",
                "causal_chain": [
                    "SEU burst corrupts gyroscope registers, initiating ADCS failure",
                    "Invalid gyroscope telemetry causes attitude divergence",
                    "Loss of attitude control causes solar arrays to off-point from the Sun",
                    "Solar array current drops to near zero, initiating battery drain",
                    "Declining bus voltage reduces heater control authority",
                    "Loss of heater control causes thermal runaway in a component",
                    "Simultaneous EPS undervoltage and TCS over-temperature trigger safe mode",
                ],
                "confidence": 0.65,
                "recovery_action_sequence": [
                    "Power-cycle gyroscope to clear SEU corruption and restore ADCS",
                    "Command sun-pointing attitude to recover solar array power generation",
                    "Monitor battery SoC recovery and re-enable loads progressively",
                    "Command heater disable override and monitor component temperature",
                    "Desaturate reaction wheels and restore full attitude control",
                    "Perform comprehensive subsystem health check before resuming science",
                    "Review cascade timeline to identify primary root cause for corrective action",
                ],
                "risk_level": "HIGH",
            },
        }

        return _ground_truth_data[fault_type]


if __name__ == "__main__":
    # Full validation: instantiate the simulator, generate one crash dump per fault
    # type, pretty-print each to stdout, and assert all required top-level keys are
    # present (Requirements 13.1–13.5).

    REQUIRED_KEYS = [
        "scenario_id",
        "timestamp",
        "fault_type",
        "fault_register",
        "pre_fault_telemetry",
        "event_log",
        "hardware_state",
        "operating_context",
    ]

    FAULT_TYPES = [
        "EPS_SOLAR_UNDERVOLT",
        "ADCS_GYRO_SEU",
        "OBC_WATCHDOG_OVERFLOW",
        "TCS_THERMAL_RUNAWAY",
        "COMMS_TRANSPONDER_LOSS",
        "MULTI_CASCADE",
    ]

    sim = SatelliteFaultSimulator(seed=42)

    for scenario_id, fault_type in enumerate(FAULT_TYPES, start=1):
        dump = sim.generate_crash_dump(fault_type, scenario_id=scenario_id)
        print(json.dumps(dump, indent=2))

        for key in REQUIRED_KEYS:
            if key not in dump:
                raise AssertionError(
                    f"Required key {key!r} is missing from the crash dump "
                    f"for fault type {fault_type!r}."
                )
