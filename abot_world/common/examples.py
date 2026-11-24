"""Built-in scene assets and the per-stream skill trigger."""

import json
import threading
from dataclasses import dataclass
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
FIRST_FRAMES_DIR = EXAMPLES_DIR / "first_frames"
REFERENCE_IMAGES_DIR = EXAMPLES_DIR / "references"
SKILLS_PER_EXAMPLE = 2
SKILL_BLOCKS = 4


@dataclass(frozen=True)
class Skill:
    name: str
    prompt: str


@dataclass(frozen=True)
class ExampleScene:
    image_hash: str
    label: str
    scene_prompt: str
    skills: tuple[Skill, ...]

    @property
    def first_frame(self) -> Path:
        return FIRST_FRAMES_DIR / f"{self.image_hash}.png"

    @property
    def reference_dir(self) -> Path:
        return REFERENCE_IMAGES_DIR / self.image_hash


EXAMPLE_SCENES = {}
for row in json.loads((EXAMPLES_DIR / "scenes.json").read_text(encoding="utf-8")):
    example = ExampleScene(
        image_hash=row["image_hash"],
        label=row["label"],
        scene_prompt=row["scene_prompt"],
        skills=tuple(Skill(**skill) for skill in row["skills"]),
    )
    if len(example.skills) != SKILLS_PER_EXAMPLE:
        raise ValueError(f"{example.image_hash} requires {SKILLS_PER_EXAMPLE} demo skills")
    EXAMPLE_SCENES[str(example.first_frame)] = example

DEFAULT_EXAMPLE = next(iter(EXAMPLE_SCENES.values()))


class ExampleSkill:
    """Play one preset skill for the configured number of generation blocks."""

    def __init__(self, scene: ExampleScene):
        self.scene = scene
        self._lock = threading.Lock()
        self._remaining = 0
        self._prompt = ""

    def request(self, slot: int):
        with self._lock:
            if self._remaining or not 0 <= slot < len(self.scene.skills):
                return False
            self._prompt = self.scene.skills[slot].prompt
            self._remaining = SKILL_BLOCKS
            return True

    def caption(self, *, interrupt: bool = False) -> str:
        with self._lock:
            if interrupt:
                self._remaining = 0
                self._prompt = ""
            return self._prompt if self._remaining else ""

    def advance(self):
        with self._lock:
            self._remaining = max(0, self._remaining - 1)
