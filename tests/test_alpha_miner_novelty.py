from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))

from alpha_miner import FactorCandidate, _novelty_score, _structure_signature


def test_structure_signature_collapses_window_and_winsor_params():
    left = "winsorize(-ts_return(close, 5), lower=0.01, upper=0.99)"
    right = " winsorize( -ts_return(close, 20), lower=0.02, upper=0.98 ) "
    assert _structure_signature(left) == _structure_signature(right)


def test_novelty_penalizes_parameter_only_change():
    base = FactorCandidate(
        name="base",
        code="",
        code_hash="",
        generation=0,
        family="momentum",
        expression="rank(ts_return(close, 20))",
        required_fields=["close"],
    )
    tweak = FactorCandidate(
        name="tweak",
        code="",
        code_hash="",
        generation=1,
        family="momentum",
        expression="rank(ts_return(close, 60))",
        required_fields=["close"],
    )
    assert _novelty_score(tweak, [base]) < 0.25


def test_novelty_rewards_new_field_and_operator_mix():
    base = FactorCandidate(
        name="base",
        code="",
        code_hash="",
        generation=0,
        family="momentum",
        expression="rank(ts_return(close, 20))",
        required_fields=["close"],
    )
    candidate = FactorCandidate(
        name="volume_interaction",
        code="",
        code_hash="",
        generation=1,
        family="volume",
        expression="rank(delta(close, 10) / (ts_mean(volume, 10) + 1))",
        required_fields=["close", "volume"],
    )
    assert _novelty_score(candidate, [base]) >= 0.25
