import numpy as np

from occlubench.metrics import bootstrap_ci, format_table, summarize


def test_bootstrap_ci_brackets_mean():
    v = np.r_[np.ones(70), np.zeros(30)]
    point, lo, hi = bootstrap_ci(v, n_boot=2000)
    assert point == 0.7 and lo < 0.7 < hi and hi - lo < 0.25


def test_bootstrap_empty_is_nan():
    assert all(np.isnan(x) for x in bootstrap_ci([]))


def _rec(clean, pert, cs=50, ps=60):
    return dict(
        condition="c",
        camera="wrist",
        severity=1.0,
        onset=0.1,
        seed=0,
        clean_success=clean,
        pert_success=pert,
        clean_step=cs,
        pert_step=ps,
        coverage=1.0,
        coverage_other=0.0,
    )


def test_summarize_paired_metrics():
    recs = [_rec(True, False)] * 4 + [_rec(True, True)] * 4 + [_rec(False, False)] * 2
    (row,) = summarize(recs, n_boot=500)
    assert row["clean_sr"] == 0.8 and row["pert_sr"] == 0.4
    assert row["paired_drop"][0] == 0.4
    assert row["conditional_success"][0] == 0.5
    assert row["recovery_steps"][0] == 10
    assert "paired drop" in format_table([row])
