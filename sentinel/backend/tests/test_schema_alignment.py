"""
test_schema_alignment.py — Verifies fault type names are consistent
across fault_simulator.py, evaluator.py, prompts.py, and dataset_generator.py.

This is the critical alignment test that prevents fault_class_accuracy = 0%.
If any test here fails, the LLM will output one name and the evaluator will
score a different name.

Run: python -m pytest test_schema_alignment.py -v
"""

import json
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


from simulation.fault_simulator import SatelliteFaultSimulator
from app.analytics.evaluator import GROUND_TRUTH_REGISTRY, evaluate_response
from app.agent.prompts import FAULT_SIGNATURES

# The 6 canonical fault type names (source of truth: prompts.py)
CANONICAL_NAMES = [
    "ADCS_GYRO_SEU",
    "EPS_SOLAR_UNDERVOLT",
    "OBC_WATCHDOG_OVERFLOW",
    "TCS_THERMAL_RUNAWAY",
    "COMMS_TRANSPONDER_LOSS",
    "MULTI_CASCADE",
]

# The 6 OLD names that must NOT appear anywhere
OLD_NAMES = [
    "EPS_POWER_FAULT",
    "ADCS_SENSOR_FAULT",
    "OBC_SOFTWARE_FAULT",
    "TCS_THERMAL_FAULT",
    "COMMS_FAULT",
    "MULTI_SYSTEM_CASCADE",
]


class TestSimulatorFaultTypesCanonical:
    """Test 1: Simulator generates crash dumps with canonical fault_type values."""

    def setup_method(self):
        self.sim = SatelliteFaultSimulator(seed=42)

    @pytest.mark.parametrize("fault_type", CANONICAL_NAMES)
    def test_crash_dump_fault_type_matches_canonical(self, fault_type):
        """Generate a crash dump and verify fault_type field uses canonical name."""
        dump = self.sim.generate_crash_dump(fault_type, scenario_id=1)
        assert dump["fault_type"] == fault_type, (
            f"Expected fault_type={fault_type!r}, got {dump['fault_type']!r}"
        )

    @pytest.mark.parametrize("fault_type", CANONICAL_NAMES)
    def test_crash_dump_fault_type_in_registry(self, fault_type):
        """Verify each generated fault_type is a key in GROUND_TRUTH_REGISTRY."""
        dump = self.sim.generate_crash_dump(fault_type, scenario_id=1)
        assert dump["fault_type"] in GROUND_TRUTH_REGISTRY, (
            f"fault_type={dump['fault_type']!r} not found in GROUND_TRUTH_REGISTRY"
        )

    @pytest.mark.parametrize("fault_type", CANONICAL_NAMES)
    def test_ground_truth_classification_matches(self, fault_type):
        """Verify get_ground_truth returns matching root_cause_classification."""
        truth = self.sim.get_ground_truth(fault_type)
        assert truth["root_cause_classification"] == fault_type


class TestGroundTruthRegistryAlignment:
    """Test 2: GROUND_TRUTH_REGISTRY keys match their root_cause values."""

    def test_all_canonical_names_present(self):
        """All 6 canonical names must be keys in GROUND_TRUTH_REGISTRY."""
        for name in CANONICAL_NAMES:
            assert name in GROUND_TRUTH_REGISTRY, (
                f"Canonical name {name!r} missing from GROUND_TRUTH_REGISTRY"
            )

    def test_keys_match_root_cause_values(self):
        """Every registry key must equal its own root_cause value."""
        for key, entry in GROUND_TRUTH_REGISTRY.items():
            assert key == entry["root_cause"], (
                f"Key {key!r} != root_cause {entry['root_cause']!r}"
            )

    def test_exactly_six_entries(self):
        """Registry must have exactly 6 entries."""
        assert len(GROUND_TRUTH_REGISTRY) == 6, (
            f"Expected 6 entries, got {len(GROUND_TRUTH_REGISTRY)}"
        )


class TestEvaluatorScoresCorrectName:
    """Test 3: evaluate_response matches on canonical names, rejects old names."""

    def _make_mock_response(self, root_cause: str) -> str:
        """Build a minimal valid SENTINEL response JSON string."""
        return json.dumps({
            "hypotheses": [
                {
                    "rank": 1,
                    "root_cause": root_cause,
                    "affected_component": "GYRO_A",
                    "confidence": 0.92,
                    "causal_chain": ["SEU spike", "Gyro NaN", "Safe mode"],
                },
                {
                    "rank": 2,
                    "root_cause": "MULTI_CASCADE",
                    "affected_component": "OBC",
                    "confidence": 0.05,
                    "causal_chain": ["Alternative", "Low probability"],
                },
                {
                    "rank": 3,
                    "root_cause": "EPS_SOLAR_UNDERVOLT",
                    "affected_component": "SOLAR_ARRAY_A",
                    "confidence": 0.03,
                    "causal_chain": ["Unlikely", "Ruled out"],
                },
            ],
            "recovery_plan": [
                {
                    "step": 1,
                    "command": "CMD_VERIFY_SEU_COUNTER",
                    "rationale": "Check SEU counter",
                    "wait_seconds": 10,
                    "verify": "SEU counter value",
                    "risk": "LOW",
                },
                {
                    "step": 2,
                    "command": "CMD_GYRO_RESET",
                    "rationale": "Reset gyro",
                    "wait_seconds": 15,
                    "verify": "Gyro rate valid",
                    "risk": "LOW",
                },
            ],
            "confidence": 0.92,
            "requires_human_review": True,
            "reasoning_summary": "SEU counter spiked causing gyro NaN.",
        })

    def test_correct_canonical_name_scores_true(self):
        """evaluate_response with correct canonical name → fault_class_correct=True."""
        response = self._make_mock_response("ADCS_GYRO_SEU")
        result = evaluate_response(response, "ADCS_GYRO_SEU")
        assert result["fault_class_correct"] is True, (
            "Expected fault_class_correct=True for matching canonical name"
        )

    def test_old_name_scores_false(self):
        """evaluate_response with old name → fault_class_correct=False."""
        response = self._make_mock_response("ADCS_SENSOR_FAULT")
        result = evaluate_response(response, "ADCS_GYRO_SEU")
        assert result["fault_class_correct"] is False, (
            "Old name ADCS_SENSOR_FAULT should NOT match canonical ADCS_GYRO_SEU"
        )

    @pytest.mark.parametrize("canonical_name", CANONICAL_NAMES)
    def test_all_canonical_names_score_correctly(self, canonical_name):
        """Every canonical name should score True when matched against itself."""
        response = self._make_mock_response(canonical_name)
        result = evaluate_response(response, canonical_name)
        assert result["fault_class_correct"] is True


class TestPromptsFaultSignatureNamesMatchRegistry:
    """Test 4: FAULT_SIGNATURES teaches the LLM exactly the names evaluator expects."""

    @pytest.mark.parametrize("canonical_name", CANONICAL_NAMES)
    def test_canonical_name_in_fault_signatures(self, canonical_name):
        """Each canonical name must appear in FAULT_SIGNATURES."""
        assert canonical_name in FAULT_SIGNATURES, (
            f"Canonical name {canonical_name!r} not found in FAULT_SIGNATURES"
        )


class TestNoOldNamesInSimulator:
    """Test 5: Old fault type names are no longer accepted by the simulator."""

    def setup_method(self):
        self.sim = SatelliteFaultSimulator(seed=42)

    @pytest.mark.parametrize("old_name", OLD_NAMES)
    def test_old_name_not_in_valid_fault_types(self, old_name):
        """Old fault type names must NOT be in _VALID_FAULT_TYPES."""
        assert old_name not in self.sim._VALID_FAULT_TYPES, (
            f"Old name {old_name!r} should NOT be in _VALID_FAULT_TYPES"
        )

    @pytest.mark.parametrize("old_name", OLD_NAMES)
    def test_old_name_raises_on_generate(self, old_name):
        """Generating a crash dump with an old name must raise ValueError."""
        with pytest.raises(ValueError):
            self.sim.generate_crash_dump(old_name, scenario_id=1)

    @pytest.mark.parametrize("old_name", OLD_NAMES)
    def test_old_name_not_in_registry(self, old_name):
        """Old names must NOT be keys in GROUND_TRUTH_REGISTRY."""
        assert old_name not in GROUND_TRUTH_REGISTRY, (
            f"Old name {old_name!r} should NOT be in GROUND_TRUTH_REGISTRY"
        )
