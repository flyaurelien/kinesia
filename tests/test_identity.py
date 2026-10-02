"""Cutting masklets where they change person, and joining each person's pieces."""

import unittest

import numpy as np

from kinesia.scene.identity import Observations, fit_scorer, identify, whiten

RATE = 24.0


class Video:
    """Synthetic people seen through masklets.

    Everyone has their own look; on top of it, every view varies along a few
    directions shared by all (pose, light), more than people differ, which is
    what the whitening is for.
    """

    def __init__(self, seed: int = 0, dims: int = 24):
        self.rng = np.random.default_rng(seed)
        self.dims = dims
        self.pose = self.rng.normal(size=(3, dims)) * 1.5
        self.rows: list[tuple] = []

    def look(self) -> np.ndarray:
        return self.rng.normal(size=self.dims)

    def show(self, piece: int, frames, look: np.ndarray, where=(0.0, 0.0)) -> None:
        for frame in frames:
            view = look + self.rng.normal(size=3) @ self.pose + 0.3 * self.rng.normal(size=self.dims)
            self.rows.append((frame, piece, (where[0], where[1], 0.0), view))

    def observations(self) -> Observations:
        frame, piece, position, appearance = (np.array(v) for v in zip(*self.rows))
        return Observations(frame, piece, piece, position.astype(float), np.ones(len(frame), bool), appearance)


def people_of(found, observations, piece: int, frames=None) -> set[int]:
    rows = observations.piece == piece
    if frames is not None:
        rows &= np.isin(observations.frame, list(frames))
    return set(found.person[rows].tolist()) - {-1}


class IdentityTest(unittest.TestCase):
    def crowd(self, video: Video, frames=range(0, 480)) -> None:
        """Three other people, always in view: they calibrate what "different people" looks like."""
        for k, where in enumerate(((6, 0), (0, 6), (-6, 0))):
            video.show(10 + k, frames, video.look(), where)

    def test_a_player_who_leaves_and_comes_back_gets_their_identity_back(self):
        video = Video(1)
        self.crowd(video)
        look = video.look()
        video.show(0, range(0, 200), look)
        video.show(1, range(300, 480), look, (1.0, 0.0))  # back 4 s later
        observations = video.observations()
        found = identify(observations, RATE)
        self.assertEqual(people_of(found, observations, 0), people_of(found, observations, 1))
        self.assertEqual(len(set(found.person.tolist())), 4)
        self.assertEqual(len(found.links), 1)

    def test_people_seen_at_the_same_time_are_never_joined(self):
        video = Video(2)
        self.crowd(video)
        look = video.look()
        video.show(0, range(0, 200), look)
        video.show(1, range(150, 400), look, (2.0, 0.0))  # the same look, but both in view at once
        observations = video.observations()
        found = identify(observations, RATE)
        self.assertNotEqual(people_of(found, observations, 0), people_of(found, observations, 1))

    def test_a_return_too_far_away_to_be_reached_is_someone_else(self):
        video = Video(3)
        self.crowd(video)
        look = video.look()
        video.show(0, range(0, 200), look, (0.0, 0.0))
        video.show(1, range(204, 400), look, (40.0, 0.0))  # 40 m in a sixth of a second
        observations = video.observations()
        found = identify(observations, RATE)
        self.assertNotEqual(people_of(found, observations, 0), people_of(found, observations, 1))

    def test_two_masklets_that_swap_people_are_cut_and_rejoined(self):
        video = Video(4)
        self.crowd(video)
        a, b = video.look(), video.look()
        video.show(0, range(0, 240), a, (0.0, 0.0))
        video.show(0, range(240, 480), b, (1.0, 0.0))
        video.show(1, range(0, 240), b, (1.0, 0.0))
        video.show(1, range(240, 480), a, (0.0, 0.0))
        observations = video.observations()
        found = identify(observations, RATE)
        self.assertEqual(sorted(c["piece"] for c in found.cuts), [0, 1])
        person_a = people_of(found, observations, 0, range(0, 230))
        self.assertEqual(len(person_a), 1)
        self.assertEqual(person_a, people_of(found, observations, 1, range(250, 480)))
        self.assertEqual(people_of(found, observations, 1, range(0, 230)), people_of(found, observations, 0, range(250, 480)))
        self.assertNotEqual(person_a, people_of(found, observations, 1, range(0, 230)))

    def test_a_masklet_that_keeps_one_person_is_not_left_cut(self):
        video = Video(5)
        self.crowd(video)
        video.show(0, range(0, 480), video.look())
        observations = video.observations()
        found = identify(observations, RATE)
        self.assertFalse([c for c in found.cuts if c["piece"] == 0])
        self.assertEqual(len(people_of(found, observations, 0)), 1)
        self.assertTrue((found.person[observations.piece == 0] >= 0).all())

    def test_an_ambiguous_return_stays_a_new_person(self):
        video = Video(6)
        self.crowd(video)
        twin = video.look()
        video.show(0, range(0, 200), twin, (0.0, 0.0))
        video.show(1, range(0, 200), twin + 0.05 * video.look(), (1.0, 0.0))  # look-alikes, side by side
        video.show(2, range(300, 480), twin, (0.5, 0.0))  # one of them is back: which one?
        observations = video.observations()
        found = identify(observations, RATE)
        back = people_of(found, observations, 2)
        self.assertFalse(back & (people_of(found, observations, 0) | people_of(found, observations, 1)))
        self.assertTrue(found.ambiguous)

    def test_without_people_seen_together_nothing_is_joined(self):
        video = Video(7)
        look = video.look()
        video.show(0, range(0, 200), look)
        video.show(1, range(300, 480), look)
        observations = video.observations()
        found = identify(observations, RATE)
        self.assertNotIn("weights", found.calibration["join"])
        self.assertNotEqual(people_of(found, observations, 0), people_of(found, observations, 1))


class CalibrationTest(unittest.TestCase):
    def test_the_score_is_a_likelihood_ratio(self):
        rng = np.random.default_rng(0)
        same = rng.normal(0.6, 0.1, size=(20000, 1))
        different = rng.normal(0.4, 0.1, size=(5000, 1))  # fewer: the classes are weighted equally
        scorer = fit_scorer(same, different)
        # Two normal classes of equal spread: the log-ratio is 0 midway and
        # grows by (0.2 / 0.1**2) per unit of similarity.
        self.assertAlmostEqual(float(scorer(np.array([0.5]))), 0.0, delta=0.1)
        self.assertAlmostEqual(float(scorer.weights[0]), 20.0, delta=1.0)

    def test_whitening_discounts_what_varies_within_a_person(self):
        rng = np.random.default_rng(1)
        looks = rng.normal(size=(6, 8))
        pose = np.zeros(8)
        pose[0] = 1.0
        groups = np.repeat(np.arange(6), 200)
        x = looks[groups] + 5.0 * rng.normal(size=(1200, 1)) * pose + 0.1 * rng.normal(size=(1200, 8))
        z = whiten(x, groups, np.ones(1200, bool), dims=8)
        same = np.mean([z[groups == g][:100] @ z[groups == g][100:].T for g in range(6)])
        different = np.mean([z[groups == 0] @ z[groups == g].T for g in range(1, 6)])
        self.assertGreater(same, 0.8)
        self.assertLess(different, 0.3)


if __name__ == "__main__":
    unittest.main()
