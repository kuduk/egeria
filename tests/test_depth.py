import numpy as np

from egeria.depth import fit_exit_temperatures, simulate, sweep


def _record(exit_logits, label, qtype="choice"):
    n = len(exit_logits[0])
    return {
        "type": qtype,
        "exit_layers": [3, 7, 11],
        "exit_logits": exit_logits,
        "option_logits": exit_logits[-1],
        "gold": list(np.eye(n)[label]),
        "label": label,
    }


def test_confident_early_layer_exits_early():
    easy = _record([[10.0, 0.0], [10.0, 0.0], [10.0, 0.0]], 0)
    hard = _record([[0.1, 0.0], [0.2, 0.0], [0.0, 5.0]], 1)
    result = simulate([easy, hard], {}, threshold=0.9, num_layers=12)
    assert result["exit_histogram"] == {3: 1, 7: 0, 11: 1}
    assert result["compute"] == (4 / 12 + 12 / 12) / 2
    assert result["accuracy"] == 1.0


def test_threshold_above_one_is_full_depth():
    records = [_record([[5.0, 0.0], [5.0, 0.0], [5.0, 0.0]], 0)]
    [full] = [r for r in sweep(records, {}, 12, [1.01])]
    assert full["compute"] == 1.0


def test_exit_temperatures_per_layer():
    rng = np.random.default_rng(1)
    records = []
    for _ in range(300):
        label = int(rng.integers(2))
        signal = np.eye(2)[label] * 2.0
        noise = rng.normal(size=2)
        # layer 0 molto sovraconfidente, layer 2 ben calibrato
        records.append(_record([list((signal + noise) * 10), list(signal + noise), list(signal + noise)], label))
    temps = fit_exit_temperatures(records)["choice"]
    assert temps[0] > 5 * temps[2]


def test_oracle_exits_where_argmax_stabilizes():
    from egeria.depth import oracle

    flips_late = _record([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], 1)  # stabile dal layer 7
    always = _record([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], 1)  # stabile dal layer 3
    result = oracle([flips_late, always], num_layers=12)
    assert result["stable_from"] == {3: 1, 7: 1, 11: 0}
    assert result["compute"] == (8 / 12 + 4 / 12) / 2
