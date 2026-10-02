"""Who is who: cutting masklets where they change person, and joining each person's pieces.

SAM 3.1 follows people with its own memory, but two things still go wrong.
During an occlusion a masklet can slide onto someone else (two players swap,
or the masklet ends on a passer-by), and a player who leaves the picture, or
stays hidden too long, comes back as a new masklet. Both are settled here by
appearance, measured on the person alone: DINOv3 features of the person's
pixels (the background greyed out before the model sees them) and the colours
of the upper and lower body.

Nothing is tuned to a particular video; each video calibrates itself:

* a masklet shows one person in many poses and lights, so the features are
  whitened against that variation, which leaves what tells people apart;
* the same masklet seconds apart is the same person, and people seen at the
  same time are different people, so a logistic model fitted on those pairs
  turns similarities into a score: the log-likelihood ratio of "same person"
  over "different people".

Appearance is always compared between windows of half a second of clear
views, the unit the score is calibrated on; longer stretches are compared by
the median score of their windows.

A masklet is cut where it lastingly changes person. The pieces are then
joined most-confident first: never two people seen at the same time, never
faster than a sprint, and never when a rival candidate scores nearly as well,
so an ambiguous return becomes a new person rather than a wrong one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

RUN_SPEED = 9.0  # m/s, a sprint; bounds how far someone can move while unseen
WINDOW_SECONDS = 0.5  # appearance is compared between windows of this many seconds of clear views
APART_SECONDS = 2.0  # "same person" examples: windows of one masklet at least this far apart
MAX_GAP_SECONDS = 20.0  # the longest absence bridged
TOGETHER_FRAMES = 5  # masklets seen together this often are different people
MIN_EXAMPLES = 20  # calibration pairs needed of each kind; with fewer, nothing is cut or joined
MIN_CLEAR = 3  # clear views a piece needs before it can be joined to another
MAX_WINDOWS = 24  # windows per piece compared when joining (evenly spread)
WHITEN_DIMS = 64
CUT_SCORE = 0.0  # cut where the two sides are more likely different people than the same
JOIN_SCORE = 3.0  # join when "same person" is e^3 = 20 times more likely than "different people"
MARGIN = 2.0  # ... and e^2 = 7 times more likely than with the best rival
CONTINUITY = 3.0  # in favour of rejoining the two sides of a cut masklet (SAM 3.1 saw one person)
LEFTOVER_SECONDS = 1.5  # shorter remains of a masklet that changed person are dropped
MAX_PAIRS = 200_000  # calibration pairs kept of each kind


@dataclass
class Observations:
    """One row per person and frame."""

    frame: np.ndarray  # (N,) int
    track: np.ndarray  # (N,) the SAM 3.1 masklet of each row
    piece: np.ndarray  # (N,) the piece of its masklet (masklets already cut where they jump)
    position: np.ndarray  # (N, 3) world position of the body root, nan when unknown
    clear: np.ndarray  # (N,) bool: a clear view (wholly in the picture, large enough, not hidden)
    appearance: np.ndarray  # (N, D) image features of the masked person (whitened here)
    colour: np.ndarray | None = None  # (N, C) colour histograms, compared as they are


@dataclass
class Identities:
    person: np.ndarray  # (N,) the person of each row; -1 for rows dropped at a change of person
    pieces: np.ndarray  # (N,) the piece of each row once masklets are cut; -1 likewise
    cuts: list[dict] = field(default_factory=list)
    links: list[dict] = field(default_factory=list)
    leftovers: list[dict] = field(default_factory=list)
    ambiguous: list[dict] = field(default_factory=list)
    calibration: dict = field(default_factory=dict)


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-9)


def whiten(x: np.ndarray, groups: np.ndarray, use: np.ndarray, dims: int = WHITEN_DIMS, shrink: float = 0.1) -> np.ndarray:
    """Unit vectors in which the spread within each group (one masklet: one person) is white."""
    x = x.astype(np.float64)
    sample, members = x[use], groups[use]
    if len(sample) < 2:
        return _unit(x)
    mean = sample.mean(axis=0)
    _, _, vt = np.linalg.svd(sample - mean, full_matrices=False)
    basis = vt[:dims].T
    z = (sample - mean) @ basis
    scatter, count = np.zeros((basis.shape[1],) * 2), 0
    for g in np.unique(members):
        part = z[members == g]
        if len(part) >= 5:
            part = part - part.mean(axis=0)
            scatter += part.T @ part
            count += len(part)
    if count == 0:
        return _unit(x)
    scatter /= count
    size = len(scatter)
    scatter = (1 - shrink) * scatter + shrink * np.trace(scatter) / size * np.eye(size)
    values, vectors = np.linalg.eigh(scatter)
    return _unit((x - mean) @ basis @ (vectors / np.sqrt(values)) @ vectors.T)


@dataclass
class Scorer:
    """Similarities (one per feature) to the log-likelihood ratio of "same person"."""

    weights: np.ndarray
    bias: float

    def __call__(self, similarities: np.ndarray) -> np.ndarray:
        return similarities @ self.weights + self.bias


def fit_scorer(same: np.ndarray, different: np.ndarray, l2: float = 1e-2) -> Scorer:
    """Logistic regression with both classes weighted equally, so that its logit is a likelihood ratio."""
    x = np.vstack([same, different])
    y = np.r_[np.ones(len(same)), np.zeros(len(different))]
    weight = np.where(y > 0, 0.5 / len(same), 0.5 / len(different)) * len(y)
    a = np.hstack([x, np.ones((len(x), 1))])
    ridge = l2 * np.diag(np.r_[np.ones(x.shape[1]), 0.0])
    theta = np.zeros(a.shape[1])
    for _ in range(50):
        p = 1.0 / (1.0 + np.exp(-(a @ theta)))
        step = np.linalg.solve((a * (weight * p * (1 - p))[:, None]).T @ a + ridge, a.T @ (weight * (p - y)) + ridge @ theta)
        theta -= step
        if np.abs(step).max() < 1e-8:
            break
    return Scorer(theta[:-1], float(theta[-1]))


def _sorted_rows(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask)
    return rows[np.argsort(frame[rows], kind="stable")]


def _together(frame: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The group ids, and in how many frames each pair of groups is seen together."""
    ids = np.unique(groups[groups >= 0])
    present = np.zeros((len(ids), int(frame.max()) + 1))
    seen = groups >= 0
    present[np.searchsorted(ids, groups[seen]), frame[seen]] = 1.0
    return ids, present @ present.T


class _Views:
    """The clear views of each group of rows, as features and as windows of ``size`` views."""

    def __init__(self, observations: Observations, groups: np.ndarray, size: int):
        use = observations.clear & (groups >= 0)
        self.values = [whiten(observations.appearance, groups, use)]
        if observations.colour is not None:
            self.values.append(_unit(observations.colour.astype(np.float64)))
        self.frame, self.size = observations.frame, size
        self.clear = {int(g): _sorted_rows(self.frame, use & (groups == g)) for g in np.unique(groups[groups >= 0])}

    def windows(self, rows: np.ndarray, at_most: int | None = None) -> list[np.ndarray]:
        """Descriptors of consecutive windows of ``rows`` (one window when there are fewer rows)."""
        count = max(1, len(rows) // self.size)
        chosen = range(count) if at_most is None or count <= at_most else np.linspace(0, count - 1, at_most).round().astype(int)
        return self.describe([rows[k * self.size : (k + 1) * self.size] if count > 1 else rows for k in chosen])

    def describe(self, sets: list[np.ndarray]) -> list[np.ndarray]:
        return [_unit(np.stack([v[rows].mean(axis=0) for rows in sets])) for v in self.values]

    @staticmethod
    def similarities(a: list[np.ndarray], b: list[np.ndarray]) -> np.ndarray:
        """``(len(a), len(b), features)`` cosine similarities between two lists of descriptors."""
        return np.stack([x @ y.T for x, y in zip(a, b)], axis=-1)

    def compare(self, scorer: Scorer, a: np.ndarray | list, b: np.ndarray | list, at_most: int | None = None) -> float:
        """The median score between the windows of two sets of clear rows (or their window descriptors)."""
        a = self.windows(a, at_most) if isinstance(a, np.ndarray) else a
        b = self.windows(b, at_most) if isinstance(b, np.ndarray) else b
        return float(np.median(scorer(self.similarities(a, b))))


def _calibrate(views: _Views, groups: np.ndarray, rate: float, adjacent: bool) -> tuple[Scorer | None, dict]:
    """Fit the scorer on windows: of one masklet ("same"), of masklets seen together ("different").

    ``adjacent`` picks the "same" examples: back-to-back windows (to cut a
    masklet), or windows at least ``APART_SECONDS`` apart (to join pieces).
    """
    frame, size = views.frame, views.size
    sets, owner = [], []
    for g, rows in views.clear.items():
        for k in range(len(rows) // size):
            sets.append(rows[k * size : (k + 1) * size])
            owner.append(g)
    report: dict = {"windows": len(sets)}
    if len(sets) < 2:
        return None, report
    owner = np.array(owner)
    first = np.array([frame[s[0]] for s in sets])
    last = np.array([frame[s[-1]] for s in sets])
    described = views.describe(sets)
    sims = views.similarities(described, described)
    i, j = np.triu_indices(len(sets), 1)
    same_group = owner[i] == owner[j]
    if adjacent:  # with no gap in between, where the masklet could have changed person
        same = same_group & (j == i + 1) & (first[j] - last[i] <= size)
    else:
        same = same_group & (first[j] - last[i] >= APART_SECONDS * rate)
    ids, together = _together(frame, groups)
    different = ~same_group & (together[np.searchsorted(ids, owner[i]), np.searchsorted(ids, owner[j])] >= TOGETHER_FRAMES)
    rng = np.random.default_rng(0)
    pairs = []
    for chosen in (np.flatnonzero(same), np.flatnonzero(different)):
        if len(chosen) > MAX_PAIRS:
            chosen = rng.choice(chosen, MAX_PAIRS, replace=False)
        pairs.append(sims[i[chosen], j[chosen]])
    report.update(same=len(pairs[0]), different=len(pairs[1]))
    if min(len(pairs[0]), len(pairs[1])) < MIN_EXAMPLES:
        return None, report
    scorer = fit_scorer(*pairs)
    report.update(
        weights=np.round(scorer.weights, 2).tolist(),
        bias=round(scorer.bias, 2),
        same_p5=round(float(np.percentile(scorer(pairs[0]), 5)), 1),
        different_p99=round(float(np.percentile(scorer(pairs[1]), 99)), 1),
    )
    return scorer, report


def _change_points(views: _Views, scorer: Scorer, clear: np.ndarray) -> list[tuple[int, float]]:
    """Where, along one masklet's clear views, the person changes: ``(index into clear, score)``.

    Windows on either side of every position are compared; each dip below
    ``CUT_SCORE`` gives a candidate. A candidate stands only if the stretches
    on either side of it, up to the neighbouring candidates, still look like
    different people: someone passing in front makes a dip, not a lasting
    change.
    """
    size = views.size
    sums = [np.vstack([np.zeros(v.shape[1]), np.cumsum(v[clear], axis=0)]) for v in views.values]
    at = np.arange(size, len(clear) - size + 1)
    sims = np.stack([np.sum(_unit(s[at] - s[at - size]) * _unit(s[at + size] - s[at]), axis=1) for s in sums], axis=-1)
    score = scorer(sims)
    edges = np.flatnonzero(np.diff(np.r_[0, (score < CUT_SCORE).astype(np.int8), 0]))
    bounds = [0, *(int(at[lo + np.argmin(score[lo:hi])]) for lo, hi in zip(edges[::2], edges[1::2])), len(clear)]
    while len(bounds) > 2:
        scores = [views.compare(scorer, clear[bounds[k - 1] : bounds[k]], clear[bounds[k] : bounds[k + 1]]) for k in range(1, len(bounds) - 1)]
        best = int(np.argmax(scores))
        if scores[best] < CUT_SCORE:
            return [(bounds[k + 1], s) for k, s in enumerate(scores)]
        del bounds[best + 1]
    return []


@dataclass
class _Seam:
    """Where a masklet was cut: the pieces on either side and the rows of the change itself."""

    before: int
    after: int
    rows: np.ndarray


def cut_masklets(observations: Observations, rate: float, size: int) -> tuple[np.ndarray, list[_Seam], list[dict], dict]:
    """New piece ids, cut wherever a masklet seems to change person; -1 for the rows of the change.

    This errs on the side of cutting: joining the pieces afterwards weighs the
    appearance of each side against all other candidates, and puts a masklet
    back together when its sides are the same person after all.
    """
    frame, piece = observations.frame, observations.piece
    views = _Views(observations, piece, size)
    scorer, report = _calibrate(views, piece, rate, adjacent=True)
    out = np.full(len(frame), -1, dtype=np.int64)
    seams: list[_Seam] = []
    cuts: list[dict] = []
    next_id = 0
    for p in np.unique(piece[piece >= 0]):
        rows = _sorted_rows(frame, piece == p)
        clear = views.clear[int(p)]
        changes = _change_points(views, scorer, clear) if scorer is not None and len(clear) >= 2 * size else []
        start = -1
        for k, score in changes:
            before, after = frame[clear[k - 1]], frame[clear[k]]
            out[rows[(frame[rows] >= start) & (frame[rows] <= before)]] = next_id
            seams.append(_Seam(next_id, next_id + 1, rows[(frame[rows] > before) & (frame[rows] < after)]))
            next_id += 1
            start = after
            cuts.append({"piece": int(p), "last_before": int(before), "first_after": int(after), "score": round(score, 1)})
        out[rows[frame[rows] >= start]] = next_id
        next_id += 1
    return out, seams, cuts, report


def join_pieces(
    observations: Observations, pieces: np.ndarray, seams: list[_Seam], rate: float, size: int
) -> tuple[np.ndarray, list[dict], list[dict], dict]:
    """Group pieces into people; returns a person per row (-1 where the piece is -1).

    The two sides of a seam start with ``CONTINUITY`` in their favour: SAM 3.1
    followed them as one person, and appearance has to overturn that.
    """
    frame = observations.frame
    views = _Views(observations, pieces, size)
    scorer, report = _calibrate(views, pieces, rate, adjacent=False)
    ids = np.unique(pieces[pieces >= 0])
    n = len(ids)
    rows_of = [_sorted_rows(frame, pieces == t) for t in ids]
    first = np.array([frame[r[0]] for r in rows_of])
    last = np.array([frame[r[-1]] for r in rows_of])

    def end_position(rows: np.ndarray, at_end: bool) -> np.ndarray:
        known = rows[np.all(np.isfinite(observations.position[rows]), axis=1)]
        return observations.position[known[-1 if at_end else 0]] if len(known) else np.full(3, np.nan)

    start_at = np.array([end_position(r, False) for r in rows_of]).reshape(n, 3)
    end_at = np.array([end_position(r, True) for r in rows_of]).reshape(n, 3)
    _, together = _together(frame, pieces)
    conflict = together > 0  # seen at the same time: two different people
    gap = first[None, :] - last[:, None]  # [i, j] > 0 when i ends before j starts
    distance = np.linalg.norm(start_at[None, :, :2] - end_at[:, None, :2], axis=-1)
    too_far = (gap > 0) & (distance > RUN_SPEED * gap / rate + 1.0)
    blocked = conflict | too_far | too_far.T
    np.fill_diagonal(blocked, True)

    pair_score = np.full((n, n), np.nan)
    if scorer is not None:
        bridged = np.maximum(gap, gap.T) <= MAX_GAP_SECONDS * rate
        clear = [views.clear[int(t)] for t in ids]
        windows = [views.windows(rows, MAX_WINDOWS) if len(rows) >= MIN_CLEAR else None for rows in clear]
        for a in range(n):
            for b in range(a + 1, n):
                if not blocked[a, b] and bridged[a, b] and windows[a] is not None and windows[b] is not None:
                    pair_score[a, b] = pair_score[b, a] = views.compare(scorer, windows[a], windows[b])
    for seam in seams:
        a, b = np.searchsorted(ids, [seam.before, seam.after])
        pair_score[a, b] = pair_score[b, a] = CONTINUITY + (pair_score[a, b] if np.isfinite(pair_score[a, b]) else 0.0)

    # Average linkage over the pieces of two people, most confident pair first.
    total = np.where(np.isfinite(pair_score), pair_score, 0.0)
    count = np.isfinite(pair_score).astype(np.float64)
    active = np.ones(n, dtype=bool)
    members = {k: [k] for k in range(n)}
    links: list[dict] = []

    def people_score() -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            score = np.where((count > 0) & ~blocked & active[:, None] & active[None, :], total / count, np.nan)
        np.fill_diagonal(score, np.nan)
        return score

    while True:
        score = people_score()
        seen_with = {k: conflict[ms].any(axis=0) for k, ms in members.items()}  # pieces seen with each person
        candidates = sorted(zip(*np.nonzero(np.triu(np.nan_to_num(score, nan=-np.inf) >= JOIN_SCORE, 1))), key=lambda ab: -score[ab])
        chosen = None
        for a, b in candidates:
            # A rival: someone else who could be a's match (or b's), but only if b (or a) is not.
            rival = -np.inf
            for x, y in ((a, b), (b, a)):
                others = np.array([k in members and k not in (a, b) and seen_with[y][members[k]].any() for k in range(n)])
                if others.any():
                    rival = max(rival, float(np.nanmax(np.where(others, score[x], -np.inf))))
            if score[a, b] - rival >= MARGIN:
                chosen = (a, b, rival)
                break
        if chosen is None:
            break
        a, b, rival = chosen
        links.append(
            {
                "pieces": [int(ids[m]) for m in members[a]],
                "joined": [int(ids[m]) for m in members[b]],
                "score": round(float(score[a, b]), 1),
                "rival": round(rival, 1) if np.isfinite(rival) else None,
            }
        )
        for matrix in (total, count):
            matrix[a] += matrix[b]
            matrix[:, a] += matrix[:, b]
        blocked[a] |= blocked[b]
        blocked[:, a] |= blocked[:, b]
        active[b] = False
        members[a] = sorted(members[a] + members.pop(b), key=lambda m: first[m])

    final = people_score()
    ambiguous = [
        {"pieces": [int(ids[m]) for m in members[a]], "other": [int(ids[m]) for m in members[b]], "score": round(float(final[a, b]), 1)}
        for a, b in zip(*np.nonzero(np.triu(np.nan_to_num(final, nan=-np.inf) >= JOIN_SCORE, 1)))
    ]
    person = np.full(len(frame), -1, dtype=np.int64)
    for number, key in enumerate(sorted(members, key=lambda k: first[members[k][0]])):
        for m in members[key]:
            person[rows_of[m]] = number
    return person, links, ambiguous, report


def identify(observations: Observations, rate: float) -> Identities:
    """Who is who in every row: masklets cut where they change person, pieces joined into people."""
    size = max(6, int(round(WINDOW_SECONDS * rate)))
    pieces, seams, cuts, cut_report = cut_masklets(observations, rate, size)
    person, links, ambiguous, join_report = join_pieces(observations, pieces, seams, rate, size)
    for seam, cut in zip(seams, cuts):  # a masklet put back together keeps the rows of its seam
        before, after = person[pieces == seam.before], person[pieces == seam.after]
        cut["rejoined"] = bool(len(before) and len(after) and before[0] == after[0])
        if cut["rejoined"]:
            person[seam.rows], pieces[seam.rows] = before[0], seam.before
    # What is left of a masklet after it changed person, if too short to tell who it is, is dropped.
    leftovers = []
    for p in np.unique(pieces[pieces >= 0]):
        rows = np.flatnonzero(pieces == p)
        who, track = person[rows[0]], observations.track[rows[0]]
        alone = np.count_nonzero(person == who) == len(rows)
        if alone and len(rows) < LEFTOVER_SECONDS * rate and np.any((observations.track == track) & (person >= 0) & (person != who)):
            person[rows] = -1
            frames = observations.frame[rows]
            leftovers.append({"track": int(track), "first": int(frames.min()), "last": int(frames.max()), "rows": len(rows)})
    dropped = bool((person < 0).any())
    person = np.unique(person, return_inverse=True)[1].reshape(-1) - int(dropped)  # numbered 0.. again
    cuts = [c for c in cuts if not c.pop("rejoined")]
    return Identities(person, pieces, cuts, links, leftovers, ambiguous, {"cut": cut_report, "join": join_report})
