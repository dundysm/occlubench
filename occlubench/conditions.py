"""Perturbation conditions: which camera is occluded, how much, and when."""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass

from occlubench.scene import CAMERAS


@dataclass(frozen=True)
class Condition:
    """One cell of the camera x severity x onset grid.

    camera:   camera whose view the occluder enters ("third_person" or "wrist"),
              or "none" for the clean reference rollout.
    severity: target fraction of the task object's pixels hidden in that camera,
              in (0, 1]. Calibrated per episode from segmentation renders.
    onset:    fraction of the episode horizon at which the occluder starts moving in.
    duration: fraction of the horizon the occluder stays in place before leaving.
    anchor:   "camera" (default): the occluder keeps re-aligning to the camera->object ray every
              step, so target-camera severity stays roughly constant even as the wrist camera moves.
              "world": the pose is frozen in the world at onset, like an object that stops in place.
    """

    camera: str = "none"
    severity: float = 0.0
    onset: float = 0.0
    duration: float = 0.25
    anchor: str = "camera"

    def __post_init__(self):
        if self.camera not in ("none", *CAMERAS):
            raise ValueError(f"unknown camera {self.camera!r}")
        if self.camera != "none" and not 0.0 < self.severity <= 1.0:
            raise ValueError("severity must be in (0, 1]")
        if not 0.0 <= self.onset < 1.0:
            raise ValueError("onset must be in [0, 1)")
        if not 0.0 < self.duration <= 1.0:
            raise ValueError("duration must be in (0, 1]")
        if self.anchor not in ("camera", "world"):
            raise ValueError("anchor must be 'camera' or 'world'")

    @property
    def is_clean(self) -> bool:
        return self.camera == "none"

    @property
    def name(self) -> str:
        if self.is_clean:
            return "clean"
        tag = "" if self.anchor == "camera" else "_world"
        return f"{self.camera}_sev{self.severity:.2f}_on{self.onset:.2f}_dur{self.duration:.2f}{tag}"

    def to_dict(self) -> dict:
        return asdict(self)


CLEAN = Condition()


def condition_grid(
    cameras=CAMERAS,
    severities=(0.5, 1.0),
    onsets=(0.15, 0.45),
    duration: float = 0.25,
    anchor: str = "camera",
) -> list[Condition]:
    """Full factorial grid. Defaults give a 2 x 2 x 2 grid (8 conditions)."""
    return [
        Condition(camera=c, severity=s, onset=o, duration=duration, anchor=anchor)
        for c, s, o in itertools.product(cameras, severities, onsets)
    ]
