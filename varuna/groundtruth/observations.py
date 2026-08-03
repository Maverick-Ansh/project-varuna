"""The canonical observed-depth record, and a JSON-backed set of them.

One record = one person, at one place, at one moment, reporting how deep the water was. Sources
differ wildly (a phone app, a sensor, a news line) so the record keeps provenance on every row:
you should always be able to ask an unflattering number where it came from.

PRIVACY: observation sources carry personal data — the Mumbai app ships the reporter's NAME and
free-text feedback on every row. `Observation` has nowhere to put either, deliberately. Adapters
must drop them at parse time, so nothing personal reaches a file, a commit, or a metric.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Observation:
    """A single observed depth of standing water.

    lat/lon  : WGS84
    ts       : ISO-8601 timestamp WITH offset (the source's local time is preserved)
    depth_m  : observed depth of standing water, metres
    source   : short id of where it came from ("mumbaiflood_cs", "citizen_report", "news", ...)
    source_id: the source's own row id, so a row can be traced back and de-duplicated
    place    : free-text locality from the source, if any (a place name, never a person)
    area     : Varuna area id this point falls in — filled in by the scorer, not the adapter
    """
    lat: float
    lon: float
    ts: str
    depth_m: float
    source: str
    source_id: Optional[str] = None
    place: Optional[str] = None
    area: Optional[str] = None
    meta: dict = field(default_factory=dict)

    @property
    def date(self) -> str:
        return self.ts[:10]

    def dt(self) -> _dt.datetime:
        """Timezone-aware datetime. Naive timestamps are read as UTC rather than guessed at."""
        s = self.ts.replace("Z", "+00:00")
        d = _dt.datetime.fromisoformat(s)
        return d if d.tzinfo else d.replace(tzinfo=_dt.timezone.utc)

    def with_area(self, area: str) -> "Observation":
        return Observation(**{**asdict(self), "area": area})


class ObservationSet:
    """A list of observations plus the QC ledger that produced it.

    `rejected` is kept and persisted on purpose. A validation set that quietly discards two
    thirds of its input is making a claim, and the claim should be auditable: every dropped row
    is counted under the rule that dropped it.
    """

    def __init__(self, observations=None, rejected=None, meta=None):
        self.observations = list(observations or [])
        self.rejected = dict(rejected or {})
        self.meta = dict(meta or {})

    def __len__(self):
        return len(self.observations)

    def __iter__(self):
        return iter(self.observations)

    def extend(self, other: "ObservationSet") -> "ObservationSet":
        self.observations.extend(other.observations)
        for k, v in other.rejected.items():
            self.rejected[k] = self.rejected.get(k, 0) + v
        return self

    def filter(self, fn) -> "ObservationSet":
        return ObservationSet([o for o in self.observations if fn(o)], self.rejected, self.meta)

    def to_dict(self):
        return {"observations": [asdict(o) for o in self.observations],
                "rejected": self.rejected, "meta": self.meta,
                "n": len(self.observations)}

    @classmethod
    def from_dict(cls, d):
        return cls([Observation(**o) for o in d.get("observations", [])],
                   d.get("rejected"), d.get("meta"))

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
