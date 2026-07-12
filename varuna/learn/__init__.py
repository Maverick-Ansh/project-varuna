"""Nightly self-improvement: citizen reports -> emulator fine-tune -> reward gate.

The loop (scripts/nightly_update.py) is deliberately conservative: a candidate emulator is
trained on report points anchored by the committed replay buffer, and REPLACES the serving
emulator only when the reward gate passes (held-out report error improves AND the simulated
grid does not degrade AND there is enough data to mean anything). "Skipped: insufficient
data" is the designed common case early on.
"""
from .labels import collect_labels, split_holdout, rain_for_day
from .finetune import finetune_emulator, band_huber
from .reward import gate, point_error, replay_rmse, append_log

__all__ = ["collect_labels", "split_holdout", "rain_for_day", "finetune_emulator",
           "band_huber", "gate", "point_error", "replay_rmse", "append_log"]
