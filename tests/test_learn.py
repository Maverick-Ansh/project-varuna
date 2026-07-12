"""Learning loop: label assembly, band-tolerant loss, fine-tune smoke, reward-gate guards."""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from varuna.learn.labels import collect_labels, split_holdout
from varuna.learn.finetune import band_huber


def _reports(day, cells, band="knee", depth=0.4):
    return [dict(id=f"{day}{i}", area="t", ts=f"{day}T12:00:00+00:00",
                 cell=list(c), depth_band=band, depth_m=depth)
            for i, c in enumerate(cells)]


def _recent_days(n):
    import datetime as dt
    today = dt.datetime.now(dt.timezone.utc).date()
    return [(today - dt.timedelta(days=i + 1)).isoformat() for i in range(n)]


def test_collect_labels_filters_and_dedupes():
    d1, d2 = _recent_days(2)
    reps = (_reports(d1, [(1, 1), (1, 1), (2, 2)])                 # duplicate cell on d1
            + _reports(d2, [(3, 3)]))
    rain = {d1: 40.0, d2: 2.0}                                     # d2 is a dry day
    out = collect_labels(reps, (19, 72), rain_fn=lambda d: rain[d])
    assert len(out) == 1 and out[0]["date"] == d1
    assert len(out[0]["points"]) == 2                              # (1,1) deduped to median


def test_split_holdout_by_day():
    days = [dict(date=d, rain_mm=50, points=[(1, 1, 0.4, "knee")]) for d in _recent_days(6)]
    tr, ho = split_holdout(days, seed=0)
    assert len(tr) + len(ho) == 6 and len(ho) >= 1
    assert not {d["date"] for d in tr} & {d["date"] for d in ho}
    tr2, ho2 = split_holdout(days, seed=0)
    assert [d["date"] for d in ho] == [d["date"] for d in ho2]     # deterministic


def test_band_huber_tolerance():
    p = torch.tensor(0.45)
    assert float(band_huber(p, 0.4, "knee")) == 0.0                # inside +-0.2 band
    assert float(band_huber(torch.tensor(1.5), 0.4, "knee")) > 0.0


@pytest.fixture
def learn_bundle(tmp_path):
    """Tiny 32x32 emulator + replay buffer so learn-loop tests run in seconds.

    The net is biased shallow (softplus(x-2) ~ 0.1-0.2 m) so chest-deep (1.3 m) report
    points have a robustly NONZERO error for the serving model, and replay targets are
    self-consistent with that shallow net."""
    from varuna.build.twin import UNet
    torch.manual_seed(0)
    net = UNet()
    with torch.no_grad():
        net.out.bias -= 2.0
    n = 32
    terrain = torch.randn(n, n) * 0.1
    X = torch.stack([torch.stack([terrain, torch.full((n, n), r / 100.0), torch.zeros(n, n)])
                     for r in (30.0, 60.0, 100.0, 140.0)])
    with torch.no_grad():
        Y = net(X).clamp(max=0.5)                                  # self-consistent targets
    torch.save(net.state_dict(), tmp_path / "emulator.pt")
    torch.save({"X": X.half(), "Y": Y.half()}, tmp_path / "replay_buffer.pt")
    return str(tmp_path)


def _days(points, dates=None):
    dates = dates or ["2026-07-01", "2026-07-02"]
    per = max(1, len(points) // len(dates))
    return [dict(date=d, rain_mm=60.0, points=points[i * per:(i + 1) * per] or points[:1])
            for i, d in enumerate(dates)]


def test_finetune_writes_candidate_only(learn_bundle):
    from varuna.learn.finetune import finetune_emulator
    before = open(os.path.join(learn_bundle, "emulator.pt"), "rb").read()
    path = finetune_emulator(learn_bundle, _days([(5, 5, 0.8, "waist"), (10, 10, 0.4, "knee")]),
                             epochs=3)
    assert os.path.basename(path) == "emulator_candidate.pt" and os.path.exists(path)
    assert open(os.path.join(learn_bundle, "emulator.pt"), "rb").read() == before


def test_gate_insufficient_data(learn_bundle):
    from varuna.learn.reward import gate
    entry = gate(learn_bundle, os.path.join(learn_bundle, "emulator.pt"),
                 _days([(1, 1, 0.4, "knee")] * 3))
    assert not entry["accepted"] and "insufficient" in entry["reason"]


def _many_points(k=20):
    g = np.random.default_rng(1)
    return [(int(r), int(c), 1.3, "chest")
            for r, c in zip(g.integers(2, 30, k), g.integers(2, 30, k))]


def test_gate_rejects_all_wet_candidate(learn_bundle):
    """A candidate that predicts ~1.3 m everywhere ACES the chest-deep report points but
    wrecks the simulated grid — the replay guard must kill it (the anti-gaming test)."""
    from varuna.build.twin import UNet
    from varuna.learn.reward import gate
    net = UNet()
    net.load_state_dict(torch.load(os.path.join(learn_bundle, "emulator.pt"),
                                   weights_only=False))
    with torch.no_grad():
        net.out.bias += 3.1                                        # softplus(~1.1) ~ 1.3 m everywhere
    cand = os.path.join(learn_bundle, "emulator_candidate.pt")
    torch.save(net.state_dict(), cand)
    entry = gate(learn_bundle, cand, _days(_many_points()))
    assert entry["point_err_before"] > 0                           # serving model misses the points
    assert not entry["accepted"]
    assert "replay" in entry["reason"] or "did not improve" in entry["reason"]


def test_gate_accepts_genuine_improvement(learn_bundle, monkeypatch):
    """Gate mechanics: better points + intact replay + robust bootstrap -> accepted."""
    import varuna.learn.reward as R
    calls = {"n": 0}

    def fake_errors(net, terrain, days, device="cpu"):
        calls["n"] += 1
        good = calls["n"] % 2 == 0                                 # old, new, old, new...
        base = 0.05 if good else 0.30
        g = np.random.default_rng(calls["n"])
        return base + 0.01 * g.random(sum(len(d["points"]) for d in days))
    monkeypatch.setattr(R, "_point_errors", fake_errors)
    monkeypatch.setattr(R, "replay_rmse", lambda net, X, Y: 0.1)
    entry = R.gate(learn_bundle, os.path.join(learn_bundle, "emulator.pt"),
                   _days(_many_points()))
    assert entry["accepted"] and entry["bootstrap_win_rate"] >= 0.7


def test_append_log(learn_bundle):
    from varuna.learn.reward import append_log
    from varuna.io import load_json
    append_log(learn_bundle, dict(date="2026-07-11", accepted=False, reason="x"))
    append_log(learn_bundle, dict(date="2026-07-12", accepted=True, reason="y"))
    logl = load_json(os.path.join(learn_bundle, "learning_log.json"))
    assert len(logl) == 2 and logl[-1]["accepted"]
