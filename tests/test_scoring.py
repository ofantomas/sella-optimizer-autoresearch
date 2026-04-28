from validate import ENERGY_VALIDITY_FLOOR, INVALID_FITNESS, score_results


def test_valid_score_is_mean_relative_steps():
    result = score_results(
        [
            {"n_steps": 5, "max_steps": 10, "converged": True, "rel_steps": 0.5, "rel_energy": ENERGY_VALIDITY_FLOOR},
            {"n_steps": 10, "max_steps": 10, "converged": False, "rel_steps": 1.0, "rel_energy": 1.1},
        ],
        num_errors=0,
    )
    assert result["is_valid"] == 1
    assert result["fitness"] == result["mean_rel_steps"] == 0.75
    assert result["lower_is_better"] is True


def test_energy_floor_is_hard_validity_gate():
    result = score_results(
        [
            {"n_steps": 1, "max_steps": 10, "converged": True, "rel_steps": 0.1, "rel_energy": ENERGY_VALIDITY_FLOOR - 0.001},
        ],
        num_errors=0,
    )
    assert result["is_valid"] == 0
    assert result["fitness"] == INVALID_FITNESS
    assert "mean_rel_energy" in result["invalid_reason"]
