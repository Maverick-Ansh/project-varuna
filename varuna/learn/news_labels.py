"""The quarantine boundary: news NEVER becomes training signal in V3.

Why this module exists at all, when its main function returns an empty list: the quarantine
is a scientific position that deserves an enforcement point and a test, not a comment. News
volume tracks media attention, not water depth — ten outlets covering one flood are ten
*correlated* points, and the reward gate's bootstrap (varuna.learn.reward) assumes i.i.d.
evidence. A coordinated burst of coverage would sail past the `min_points` bar while adding
zero independent information, which is precisely how a learner gets gamed by a news cycle.

Defense in depth:
  1. Structurally, news lives in a parallel store under the `news/` prefix
     (varuna.serve.news.NewsStore) and is never read by the learning pipeline.
  2. `filter_reports_for_learning` is the belt-and-braces gate the nightly job applies to
     the report list itself, so even a record that somehow lands in `reports/` with a
     non-citizen `source` is dropped before labelling.
  3. `news_to_labels` — the function a future V4 would have to *rewrite*, not merely call —
     returns [] unconditionally.

If V4 ever lifts the quarantine it must solve correlation first (cluster coverage into
events, one point per event, credibility-weighted) and re-derive the gate's evidence bar.
"""
from __future__ import annotations

import logging

log = logging.getLogger("varuna.learn.news_labels")

# Records the learner may see. Citizen reports written before the `source` field existed
# have no key at all -> None is deliberately trainable; anything else is not.
TRAINABLE_SOURCES = {None, "citizen"}


def filter_reports_for_learning(reports):
    """Drop every record whose provenance is not a citizen pin. Applied by the nightly job
    BEFORE labelling, so no news (or any future non-citizen source) can reach the learner."""
    kept = [r for r in reports if r.get("source") in TRAINABLE_SOURCES]
    dropped = len(reports) - len(kept)
    if dropped:
        log.warning("quarantine: dropped %d non-citizen record(s) before labelling "
                    "(news is displayed, never trained on)", dropped)
    return kept


def news_to_labels(items):  # noqa: ARG001 - the argument is the point of the signature
    """News items -> supervision labels. In V3 this is the quarantine: always [].

    Correlated media coverage must not masquerade as independent observations. See the
    module docstring for what a V4 would have to establish before changing this.
    """
    return []
