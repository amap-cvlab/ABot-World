import ast
from html.parser import HTMLParser
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

from abot_world.common.examples import EXAMPLE_SCENES
from abot_world.iris.config import load_interactive_config, load_iris_config

ROOT = Path(__file__).resolve().parents[1]
LEGACY_NAMES = {"wan", "utils", "pipeline", "quantizer", "fantasydecoder", "web_client"}


class StaticAssets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in {"src", "href", "poster"} and value:
                self.paths.append(value)


class LayoutTest(unittest.TestCase):
    def test_runtime_imports_do_not_cross_boundaries(self):
        for name, other in (("iris", "bit"), ("bit", "iris"), ("common", None)):
            for path in (ROOT / "abot_world" / name).rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and not node.level:
                        modules = [node.module]
                    elif isinstance(node, ast.Import):
                        modules = [alias.name for alias in node.names]
                    else:
                        continue
                    for module in modules:
                        with self.subTest(path=path.relative_to(ROOT), module=module):
                            self.assertNotIn(module.split(".")[0], LEGACY_NAMES)
                            self.assertFalse(other and module.startswith(f"abot_world.{other}."))

    def test_config_and_example_compatibility_paths(self):
        for name in ("default_config", "long_forcing_dmd", "model_inference"):
            self.assertEqual((ROOT / f"configs/{name}.yaml").resolve(),
                             ROOT / f"configs/iris/{name}.yaml")
        with patch.dict(os.environ, {"ABOTWORLD_CONFIG": "configs/model_inference.yaml"}):
            self.assertEqual(load_interactive_config(),
                             load_iris_config(ROOT / "configs/iris/model_inference.yaml"))
        for scene in EXAMPLE_SCENES.values():
            self.assertEqual((ROOT / "web_client/datasets/images" / scene.first_frame.name).resolve(),
                             scene.first_frame)
            self.assertEqual((ROOT / "outputs/ref_image_cache" / scene.image_hash).resolve(),
                             scene.reference_dir)

    def test_cli_help_works_from_new_and_legacy_entry_points(self):
        for target, flag in (("iris.stream_cli", "--first-frame"), ("bit.cli", "--example"),
                             ("iris.benchmark_cli", "--action-json")):
            legacy = {"iris.stream_cli": "stream_inference", "bit.cli": "infer_teacher",
                      "iris.benchmark_cli": "inference"}[target]
            for command in (["-m", f"abot_world.{target}"], [f"scripts/{legacy}.py"]):
                with self.subTest(command=command):
                    run = subprocess.run([sys.executable, *command, "--help"], cwd=ROOT,
                                         capture_output=True, text=True, timeout=30)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    self.assertIn(flag, run.stdout)

    def test_bit_defaults_to_absolute_rope_and_has_no_relative_option(self):
        command = [sys.executable, "-m", "abot_world.bit.cli", "--example", "1",
                   "--frames", "321", "--output", str(ROOT)]
        for flag in ([], ["--accept-teacher-absolute-rope"]):
            run = subprocess.run(command + flag, cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 2, run.stderr)
            self.assertIn("Output directory already exists", run.stderr)
        run = subprocess.run(command + ["--use-relative-rope"], cwd=ROOT,
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 2, run.stderr)
        self.assertIn("unrecognized arguments: --use-relative-rope", run.stderr)

    def test_website_can_be_served_without_repository_root(self):
        website = ROOT / "website"

        def check_local_path(source, url):
            parts = urlsplit(url)
            if not parts.scheme and not parts.netloc and parts.path:
                path = (source.parent / parts.path).resolve()
                self.assertTrue(path.is_relative_to(website), url)
                self.assertTrue((path / "index.html" if path.is_dir() else path).is_file(), url)

        for name in ("index.html", "abot-world-0.html", "world-0.html"):
            source = website / name
            parser = StaticAssets()
            parser.feed(source.read_text(encoding="utf-8"))
            for url in parser.paths:
                with self.subTest(source=name, url=url):
                    check_local_path(source, url)
        for source in (website / "css").glob("*.css"):
            for url in re.findall(r"""url\(\s*['"]?([^'"\s)]+)""", source.read_text(encoding="utf-8")):
                with self.subTest(source=source.name, url=url):
                    check_local_path(source, url)
        self.assertIn("url=website/", (ROOT / "index.html").read_text())
