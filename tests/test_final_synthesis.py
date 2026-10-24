import copy

import pytest

from marketpulse.experiments.final_synthesis import evaluate_claim, verify_claim
from marketpulse.experiments.recovery import file_hash


def test_evidence_trace_rejects_source_and_claim_tampering(tmp_path):
    path = tmp_path / "measured.csv"
    path.write_text("model,effect,p\nrf,-0.01,0.03\nlstm,0.02,0.04\n")
    claim = {
        "id": "C01",
        "artifact": "measured.csv",
        "sha256": file_hash(path),
        "operation": "count",
        "filters": {"effect": {"lt": 0}, "p": {"lt": 0.05}},
        "value": 1,
    }
    verify_claim(tmp_path, claim)
    changed = copy.deepcopy(claim)
    changed["value"] = 2
    with pytest.raises(ValueError, match="value mismatch"):
        verify_claim(tmp_path, changed)
    path.write_text(path.read_text().replace("-0.01", "0.01"))
    with pytest.raises(ValueError, match="source changed"):
        verify_claim(tmp_path, claim)


def test_evidence_selectors_do_not_execute_arbitrary_expressions(tmp_path):
    path = tmp_path / "measured.csv"
    path.write_text("model,effect\nrf,-0.01\nlstm,0.02\n")
    claim = {
        "id": "C02",
        "artifact": "measured.csv",
        "sha256": file_hash(path),
        "operation": "minmax",
        "column": "effect",
        "filters": {"model": ["rf"]},
    }
    assert evaluate_claim(tmp_path, claim) == [-0.01, -0.01]
    claim["filters"] = {"effect": {"__import__": "bad"}}
    with pytest.raises(ValueError, match="unsupported filter"):
        evaluate_claim(tmp_path, claim)
