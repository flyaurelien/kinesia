"""Joining SAM 3.1 masklets that belong to the same person.

SAM 3.1 keeps an identity through occlusions and crossings with its own
memory, but a player who leaves the picture (or is hidden for long) comes back
as a new masklet. Those fragments are joined here, conservatively:

* two fragments that are visible at the same time are different people,
* the later fragment must start after the earlier one ends, within ``max_gap``,
* the person could have covered the distance at running speed, and
* they look alike: mask-pooled image features, compared at the ends that
  face each other.

The similarity threshold is not a guess. Fragments that overlap in time are
certainly different people, and the two halves of one fragment are certainly
the same person; the threshold is set between those two distributions for
each video.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

RUN_SPEED = 9.0  # m/s, a sprint; bounds how far someone can move while unseen
END_FRAMES = 20  # frames at each end used to describe a fragment's appearance


@dataclass
class Fragment:
    track: int
    frames: np.ndarray  # sorted frame indices with a body
    embeddings: np.ndarray  # (F, D) unit vectors
    positions: np.ndarray  # (F, 3) world positions of the body root (nan when unknown)

    @property
    def start(self) -> int:
        return int(self.frames[0])

    @property
    def end(self) -> int:
        return int(self.frames[-1])

    def head(self) -> np.ndarray:
        return _unit(self.embeddings[:END_FRAMES].mean(axis=0))

    def tail(self) -> np.ndarray:
        return _unit(self.embeddings[-END_FRAMES:].mean(axis=0))


@dataclass
class Linking:
    groups: list[list[int]]  # tracks per person, in time order
    threshold: float
    same_person: list[float] = field(default_factory=list)
    different_people: list[float] = field(default_factory=list)
    links: list[dict] = field(default_factory=list)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-9)


def _calibrate(fragments: list[Fragment]) -> tuple[float, list[float], list[float]]:
    same: list[float] = []
    for f in fragments:
        n = len(f.frames)
        if n >= 2 * END_FRAMES:
            first = _unit(f.embeddings[: n // 2].mean(axis=0))
            second = _unit(f.embeddings[n // 2 :].mean(axis=0))
            same.append(float(first @ second))
    different: list[float] = []
    for i, a in enumerate(fragments):
        for b in fragments[i + 1 :]:
            overlap = np.intersect1d(a.frames, b.frames)
            if len(overlap) >= 5:
                ia = np.searchsorted(a.frames, overlap)
                ib = np.searchsorted(b.frames, overlap)
                different.append(float(_unit(a.embeddings[ia].mean(0)) @ _unit(b.embeddings[ib].mean(0))))
    if not same or not different:
        return 0.9, same, different
    impostor = float(np.percentile(different, 99))
    genuine = float(np.percentile(same, 10))
    # Midway between the worst genuine match and the best impostor, never below 0.8.
    return max(0.8, 0.5 * (impostor + genuine) if genuine > impostor else impostor + 0.02), same, different


def link_fragments(fragments: list[Fragment], rate: float, max_gap_seconds: float = 20.0) -> Linking:
    """Group fragments into people; each group is a list of track ids."""
    threshold, same, different = _calibrate(fragments)
    order = sorted(range(len(fragments)), key=lambda i: fragments[i].start)
    person_of = {i: i for i in order}
    members: dict[int, list[int]] = {i: [i] for i in order}

    def frames_of(person: int) -> np.ndarray:
        return np.concatenate([fragments[m].frames for m in members[person]])

    candidates = []
    for i in order:
        a = fragments[i]
        for j in order:
            b = fragments[j]
            gap = b.start - a.end
            if gap <= 0 or gap > max_gap_seconds * rate:
                continue
            similarity = float(a.tail() @ b.head())
            if similarity < threshold:
                continue
            pa, pb = a.positions[-1], b.positions[0]
            if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
                reach = RUN_SPEED * gap / rate + 1.0
                if np.linalg.norm((pb - pa)[:2]) > reach:
                    continue
            candidates.append((similarity, i, j, gap))
    candidates.sort(reverse=True)

    links = []
    for similarity, i, j, gap in candidates:
        pi, pj = person_of[i], person_of[j]
        if pi == pj:
            continue
        # j must be the first fragment of its person and i the last of its own,
        # and the joined person may never be in two places at once.
        if members[pj][0] != j or members[pi][-1] != i:
            continue
        if np.intersect1d(frames_of(pi), frames_of(pj)).size:
            continue
        members[pi].extend(members.pop(pj))
        for m in members[pi]:
            person_of[m] = pi
        links.append({"from": fragments[i].track, "to": fragments[j].track, "similarity": round(similarity, 3), "gap_frames": int(gap)})

    groups = sorted(
        ([fragments[m].track for m in sorted(ms, key=lambda m: fragments[m].start)] for ms in members.values()),
        key=lambda g: min(fragments[[f.track for f in fragments].index(t)].start for t in g),
    )
    return Linking(groups, threshold, same, different, links)
