"""Turn the GPU results of one analysis into the 3D scene the viewer plays.

Input  (``raw/``): SAM 3.1 tracks, SAM 3D Body estimates, camera intrinsics.
Output (``scene/``): ``scene.json``, ``mesh.bin``, ``people.bin``,
``overlay.json`` and ``metrics.json``.

Runs locally once the GPU results are downloaded, and can be re-run at any
time (``kinesia scene <run>``): it only needs the CPU and the body model files.
"""

from __future__ import annotations

import colorsys
import json
import time
from pathlib import Path

import numpy as np

from .. import masks as mask_ops
from ..paths import models_root
from ..tracks import TracksFile, read_tracks
from . import export, filters, quat
from .body_model import BodyModel
from .ground import WorldFrame, find_floor, floor_factors, lowest_foot, smooth_scales
from .identity import Observations, identify
from .keypoints import BONES, LEFT_HIP, RIGHT_HIP
from .metrics import SERIES, person_metrics
from .motion import PersonFrames, animate

SCENE_VERSION = 1
MIN_PERSON_FRAMES = 8
MAX_ASPECT = 2.5  # box width / height: anything wider is a bench, a banner or a group
TELEPORT_SPEED = 15.0  # m/s; a masklet moving faster between sightings has changed person
SPRINT_HEIGHTS = 7.0  # the same across the picture, in person heights per second (crouched sprint)
OFF_FLOOR_SHARE = 0.3  # pieces whose feet miss the floor this often are not on it
SHRINK = 0.6  # a box this much smaller than usual around that moment shows only part of its person
# Views that tell who someone is: the person is this tall, shows most of their
# box and is not hidden behind someone else.
SEEN_HEIGHT = 60  # pixels
SEEN_SHARE = 0.25  # of the box covered by the person's own mask
HIDDEN_SHARE = 0.3  # of the box covered by other people's masks


def palette(count: int) -> list[str]:
    """Distinct, saturated colours that read well on a light background."""
    colours = []
    for i in range(count):
        hue = (0.58 + i * 0.618034) % 1.0
        lightness = 0.46 if i % 2 == 0 else 0.56
        r, g, b = colorsys.hls_to_rgb(hue, lightness, 0.72)
        colours.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
    return colours


def _pieces(rows: dict, points_rc: np.ndarray, rate: float) -> np.ndarray:
    """A piece id per row (-1 for glitches).

    Two things go wrong inside a masklet. For a frame or two its mask can
    swallow a neighbour: the box balloons and the body model's estimate jumps,
    then comes back. Those frames are glitches and are dropped. Occasionally
    SAM 3.1 hands the masklet over to someone else for good: to a spectator
    behind the player (the distance jumps, e.g. from 20 to 40 metres, and
    stays), or to a player nearby (the body jumps sideways in the picture).
    The masklet is cut there. Sideways motion is judged in the image, where it
    is precise, in person heights; distance on the body model's own estimate,
    with a tolerance that grows with distance, where monocular depth is noisy.
    """
    pelvis = 0.5 * (points_rc[:, LEFT_HIP] + points_rc[:, RIGHT_HIP])
    pelvis_2d = 0.5 * (rows["kp2d"][:, LEFT_HIP] + rows["kp2d"][:, RIGHT_HIP]).astype(np.float64)
    heights = (rows["box"][:, 3] - rows["box"][:, 1]).astype(np.float64).clip(1.0)
    piece = np.full(len(pelvis), -1, dtype=np.int64)
    window = max(3, int(round(0.25 * rate)))  # glitch test: each side of a frame
    side = max(2, int(round(0.125 * rate)))  # change test: each side of a boundary
    next_id = 0
    for track in np.unique(rows["track"]):
        index = np.flatnonzero(rows["track"] == track)
        index = index[np.argsort(rows["frame"][index], kind="stable")]
        p, q, h = pelvis[index], pelvis_2d[index], heights[index]
        scale = np.median(h)
        depth_tolerance = 1.0 + 0.2 * np.linalg.norm(p, axis=1)
        bad = (np.linalg.norm(p - filters.rolling_median(p, 2 * window + 1), axis=1) > depth_tolerance) | (
            np.linalg.norm(q - filters.rolling_median(q, 2 * window + 1), axis=1) > 0.75 * scale
        )
        index, p, q, depth_tolerance = index[~bad], p[~bad], q[~bad], depth_tolerance[~bad]
        frames = rows["frame"][index]
        excess = np.full(max(0, len(index) - 1), -np.inf)
        for j in range(len(index) - 1):
            b, a = slice(max(0, j - side + 1), j + 1), slice(j + 1, j + 1 + side)
            seconds = (frames[j + 1] - frames[j] + side) / rate
            person = max(float(np.max(heights[index[b]])), float(np.max(heights[index[a]])))
            sideways = np.linalg.norm(np.median(q[a], axis=0) - np.median(q[b], axis=0)) / person
            distance = abs(np.linalg.norm(np.median(p[a], axis=0)) - np.linalg.norm(np.median(p[b], axis=0)))
            excess[j] = max(
                sideways - (SPRINT_HEIGHTS * seconds + 0.5),
                distance - (TELEPORT_SPEED * seconds + depth_tolerance[j]),
            )
        cuts = np.zeros(len(excess), dtype=bool)
        for lo, hi in _runs(excess > 0):  # one cut per change, where it is clearest
            cuts[lo + int(np.argmax(excess[lo:hi]))] = True
        ids = next_id + np.concatenate([[0], np.cumsum(cuts)]).astype(np.int64)
        piece[index] = ids
        next_id = int(ids[-1]) + 1 if len(ids) else next_id
    return piece


def _partial_views(rows: dict, piece: np.ndarray, rate: float) -> np.ndarray:
    """Rows whose box suddenly shrinks well below its size around that moment.

    When someone passes in front, a mask can cover only the head and
    shoulders for a moment; the body model then guesses the rest of the body,
    and its distance with it (one such guess put a player 1.3 m away). Those
    rows still tell who the person is, but their bodies are left out of the
    motion, which fills short gaps.
    """
    box = rows["box"].astype(np.float64)
    diagonal = np.hypot(box[:, 2] - box[:, 0], box[:, 3] - box[:, 1])
    partial = np.zeros(len(piece), dtype=bool)
    window = max(3, int(round(2 * rate)) | 1)
    for p in np.unique(piece[piece >= 0]):
        index = np.flatnonzero(piece == p)
        index = index[np.argsort(rows["frame"][index], kind="stable")]
        partial[index] = diagonal[index] < SHRINK * filters.rolling_median(diagonal[index], window)
    return partial


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """``(start, end)`` of each run of True values (end exclusive)."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], flags.astype(np.int8), [0]])))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def _keep_on_floor(rows: dict, piece: np.ndarray, depth: np.ndarray, reachable: np.ndarray, cut: np.ndarray) -> np.ndarray:
    """Rows to keep; fills in the floor factor where it could not be measured (in place).

    A piece that mostly cannot reach the floor is someone off the playing
    surface and is left out. Elsewhere, frames whose factor is unknown (feet
    cut off by the picture, or hidden) take it from the neighbouring frames.
    """
    keep = piece >= 0
    reliable = reachable & ~cut
    for p in np.unique(piece[keep]):
        index = np.flatnonzero(piece == p)
        index = index[np.argsort(rows["frame"][index], kind="stable")]
        seen = ~cut[index]
        if seen.any() and np.mean(~reachable[index][seen]) > OFF_FLOOR_SHARE:
            keep[index] = False
            continue
        good = reliable[index]
        if good.all():
            continue
        frames = rows["frame"][index]
        depth[index[~good]] = np.interp(frames[~good], frames[good], depth[index[good]]) if good.any() else 1.0
    return keep


def _positions(rows: dict, points_rc: np.ndarray, depth: np.ndarray, world: WorldFrame, rate: float, piece: np.ndarray) -> np.ndarray:
    """World position of each row's body root (between the hips)."""
    hips = 0.5 * (points_rc[:, LEFT_HIP] + points_rc[:, RIGHT_HIP])
    positions = np.full((len(piece), 3), np.nan)
    for p in np.unique(piece):
        index = np.flatnonzero(piece == p)
        index = index[np.argsort(rows["frame"][index], kind="stable")]
        # A piece's ends are where its person enters or leaves the picture,
        # often with the feet cut off: the raw depth factor is least reliable
        # exactly there, so positions use the smoothed one.
        positions[index] = world.apply(hips[index] * smooth_scales(depth[index], rate)[:, None])
    return positions


def _appearance(raw: Path, rows: dict, width: int, height: int) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Each row's appearance features, colour histograms and whether it is a clear view.

    They come from ``appearance.npz`` (DINOv3 and colours of the masked
    person, from the GPU stages). Analyses made before that stage existed fall
    back on SAM 3D Body's own image features, which tell people apart less
    well, and on the box alone to judge the view.
    """
    box = rows["box"].astype(np.float64)
    clear = (box[:, 3] - box[:, 1] >= SEEN_HEIGHT) & (box[:, 0] > 2) & (box[:, 1] > 2)
    clear &= (box[:, 2] < width - 3) & (box[:, 3] < height - 3)  # wholly in the picture
    path = raw / "appearance.npz"
    if not path.is_file():
        if "embed" not in rows:
            raise FileNotFoundError(f"{path} is missing: run `kinesia appearance` on this analysis")
        return rows["embed"].astype(np.float64), None, clear
    with np.load(path) as data:
        found = {(int(f), int(t)): i for i, (f, t) in enumerate(zip(data["frame"], data["track"]))}
        index = np.array([found.get((int(f), int(t)), -1) for f, t in zip(rows["frame"], rows["track"])])
        if (index < 0).any():
            raise ValueError(f"appearance.npz lacks {int((index < 0).sum())} of the bodies; delete it to compute it again")
        clear &= (data["visible"][index] >= SEEN_SHARE) & (data["hidden"][index] < HIDDEN_SHARE)
        return data["dino"][index].astype(np.float64), data["colour"][index].astype(np.float64), clear


def _overlay(tracks: TracksFile, spans: dict[int, list[tuple[int, int, int]]]) -> list[list]:
    """Per frame: ``[person, x0, y0, x1, y1, polygon, ...]`` for the video overlay.

    ``spans`` maps a SAM 3.1 track to the ``(first, last, person)`` frame spans
    during which it shows that person.
    """
    frames = []
    for record in tracks.frames:
        entries = []
        for i, track in enumerate(record.ids):
            person = next((who for first, last, who in spans.get(track, ()) if first <= record.frame <= last), None)
            if person is None:
                continue
            mask = record.mask(i, tracks.height, tracks.width)
            box = [round(v) for v in record.boxes[i]]
            entries.append([person, *box, mask_ops.outline(mask)])
        frames.append(entries)
    return frames


def build_scene(folder: Path, body_dir: Path | None = None, log=print) -> dict:
    started = time.monotonic()
    raw, out = folder / "raw", folder / "scene"
    out.mkdir(exist_ok=True)
    request = json.loads((folder / "request.json").read_text())
    rate = float(request["video"]["fps"])
    tracks = read_tracks(raw / "tracks.jsonl.gz")
    with np.load(raw / "bodies.npz") as data:
        rows = {key: data[key] for key in data.files}
    camera = json.loads((raw / "camera.json").read_text())
    focal = float(camera["K"][0][0])
    model = BodyModel(body_dir or models_root() / "sam-3d-body-dinov3")

    log("floor")
    box = rows["box"]
    rows = {key: value[box[:, 2] - box[:, 0] <= MAX_ASPECT * (box[:, 3] - box[:, 1])] for key, value in rows.items()}
    points_rc = rows["kp3d"].astype(np.float64) + rows["cam_t"].astype(np.float64)[:, None, :]
    cut = rows["box"][:, 3] >= tracks.height - 2  # feet below the picture: their position is a guess
    plane = find_floor(points_rc[~cut])
    lowest = lowest_foot(points_rc, -plane.normal)
    depth, reachable = floor_factors(lowest, plane)
    piece = _pieces(rows, points_rc, rate)
    keep = _keep_on_floor(rows, piece, depth, reachable, cut)
    off_floor = len(np.unique(piece[~keep & (piece >= 0)]))
    rows, points_rc, lowest, depth, piece = ({k: v[keep] for k, v in rows.items()}, points_rc[keep], lowest[keep], depth[keep], piece[keep])
    world = WorldFrame.from_floor(plane, np.median(lowest * depth[:, None], axis=0))

    log("identities")
    appearance, colour, clear = _appearance(raw, rows, tracks.width, tracks.height)
    position = _positions(rows, points_rc, depth, world, rate, piece)
    found = identify(Observations(rows["frame"], rows["track"], piece, position, clear, appearance, colour), rate)

    log("motion")
    partial = _partial_views(rows, found.pieces, rate)  # who they are, yes; their 3D body, no
    motions, people = [], []
    for who in range(int(found.person.max()) + 1 if len(found.person) else 0):
        index = np.flatnonzero((found.person == who) & ~partial)
        index = index[np.argsort(rows["frame"][index], kind="stable")]
        frames = rows["frame"][index]
        keep = np.concatenate([[True], np.diff(frames) > 0])  # one estimate per frame
        index, frames = index[keep], frames[keep]
        if len(index) < MIN_PERSON_FRAMES:
            continue
        person = PersonFrames(
            frames=frames,
            model_params=rows["model_params"][index],
            shape=rows["shape"][index],
            cam_t=rows["cam_t"][index].astype(np.float64),
            boxes=rows["box"][index],
            camera=np.tile([0.0, 0.0, 0.0, 1.0], (len(index), 1)),  # fixed camera: one camera frame
            depth=depth[index],
            image_height=tracks.height,
        )
        motions.append(animate(person, model, rate, world.rotation, world.origin, world.camera_position))
        people.append(index)

    colours = palette(len(motions))
    spans: dict[int, list[tuple[int, int, int]]] = {}
    for person, index in enumerate(people):
        for p in np.unique(found.pieces[index]):
            frames = rows["frame"][index][found.pieces[index] == p]
            track = int(rows["track"][index][found.pieces[index] == p][0])
            spans.setdefault(track, []).append((int(frames.min()), int(frames.max()), person))
    metrics = [person_metrics(motion, rate) for motion in motions]

    log("export")
    mesh_layout = export.write_mesh(out / "mesh.bin", model)
    people_layout = export.write_people(out / "people.bin", motions)
    export.write_json(out / "overlay.json", {"frames": _overlay(tracks, spans)})
    export.write_json(
        out / "metrics.json",
        {"series_labels": SERIES, "people": [{"summary": m["summary"], "series": m["series"]} for m in metrics]},
    )

    scene = {
        "version": SCENE_VERSION,
        "fps": rate,
        "frames": len(tracks.frames),
        "width": tracks.width,
        "height": tracks.height,
        "camera": {
            "focal": focal,
            "focal_source": camera.get("source", "default"),
            "position": np.round(world.camera_position, 4).tolist(),
            "rotation": np.round(quat.from_matrix(world.rotation), 6).tolist(),
        },
        "floor": {"camera_height": round(float(plane.offset), 3)},
        "mesh": {"file": "mesh.bin", "layout": mesh_layout, "vertices": int(model.faces.max()) + 1},
        "people_file": "people.bin",
        "encoding": {
            "quat_scale": export.QUAT_SCALE,
            "local_t_unit": export.LOCAL_T_UNIT,
            "keypoint_unit": export.KEYPOINT_UNIT,
            "flags": {"measured": export.FLAG_MEASURED, "left_contact": export.FLAG_LEFT_CONTACT, "right_contact": export.FLAG_RIGHT_CONTACT},
            "pelvis_joint": 1,
        },
        "bones": [list(b) for b in BONES],
        "people": [
            {
                "id": i,
                "label": f"Person {i + 1}",
                "color": colours[i],
                "tracks": sorted({int(t) for t in rows["track"][index]}),
                "segments": [list(s) for s in motion.segments],
                "first": int(motion.frames[0]),
                "last": int(motion.frames[-1]),
                "measured": int(motion.measured.sum()),
                "summary": metrics[i]["summary"],
                "layout": people_layout[i],
            }
            for i, (index, motion) in enumerate(zip(people, motions))
        ],
        "identity": {
            "features": "dinov3+colour" if colour is not None else "sam-3d-body",
            "pieces": int(len(np.unique(found.pieces[found.pieces >= 0]))),
            "cuts": found.cuts,
            "links": found.links,
            "ambiguous": found.ambiguous,
            "leftovers": found.leftovers,
            "calibration": found.calibration,
            "off_floor": off_floor,
        },
        "built_seconds": round(time.monotonic() - started, 1),
    }
    export.write_json(out / "scene.json", scene)
    log(
        f"scene: {len(motions)} people from {scene['identity']['pieces']} pieces of masklets "
        f"({len(found.cuts)} cut at a change of person, {off_floor} off the floor) in {scene['built_seconds']} s"
    )
    return scene

