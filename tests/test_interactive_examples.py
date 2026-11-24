import os
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.infer_teacher import read_example
from web_client.example_scenes import EXAMPLE_SCENES, ExampleSkill

ROOT = Path(__file__).resolve().parents[1]


class ExamplesTest(unittest.TestCase):
    def test_six_confirmed_scenes_have_two_skills_and_images(self):
        scenes = list(EXAMPLE_SCENES.values())
        self.assertEqual(len(scenes), 6)
        self.assertEqual([scene.image_hash for scene in scenes], [
            '9810c7b50f90eb92', 'e5d3ac82c9a561d5', '37ff0564314c12d4',
            'fe7d590734555ed0', '8a0f99ae390546c4', 'a25f518da9964a39',
        ])
        for scene in scenes:
            self.assertEqual(len(scene.skills), 2)
            self.assertTrue(scene.first_frame.is_file())
            self.assertIn(b'PNG', scene.first_frame.read_bytes()[:8])
            self.assertIn('third-person', scene.scene_prompt.lower())
            for slot in ('head', 'left', 'right', 'front', 'back'):
                self.assertEqual(len(list(scene.reference_dir.glob(slot + '.*'))), 1, (scene.label, slot))

    def test_skill_covers_four_blocks_and_interrupt_releases_it(self):
        scene = next(iter(EXAMPLE_SCENES.values()))
        skill = ExampleSkill(scene)
        self.assertFalse(skill.request(2))
        self.assertTrue(skill.request(1))
        self.assertFalse(skill.request(0))
        for _ in range(4):
            self.assertEqual(skill.caption(), scene.skills[1].prompt)
            skill.advance()
        self.assertEqual(skill.caption(), '')
        self.assertTrue(skill.request(0))
        self.assertEqual(skill.caption(interrupt=True), '')
        self.assertTrue(skill.request(1))

    def test_teacher_example_uses_same_skills_and_action_timeline(self):
        scenes = list(EXAMPLE_SCENES.values())
        for i, scene in enumerate(scenes, 1):
            for slot in (1, 2):
                image, refs, action, prompt = read_example(i, slot, 121)
                self.assertEqual(image, scene.first_frame)
                self.assertEqual(refs, scene.reference_dir)
                self.assertEqual(action, [[1, 0, 0, 0, 0, 0, 0, 0]] * 121)
                self.assertEqual(prompt['frame_ranges'], [[0, 13], [13, 61], [61, 121]])
                self.assertEqual(prompt['action_prompts'][1], scene.skills[slot - 1].prompt)

    def test_gradio_configs_keep_iris_paths_and_rope(self):
        from web_client.config import load_ui_config

        with patch.dict(os.environ, {'ABOTWORLD_CONFIG': ''}):
            base = load_ui_config()
        with patch.dict(os.environ, {'ABOTWORLD_CONFIG': 'configs/model_inference.yaml'}):
            fantasy = load_ui_config()
        self.assertEqual(base.vae_type, 'taew2_2')
        self.assertEqual(fantasy.vae_type, 'fantasydecoder')
        self.assertEqual(fantasy.fantasydecoder_checkpoint,
                         'checkpoints/ABot-World-0-5-5B-Iris/FantasyDecoder/model.pt')
        self.assertEqual(fantasy.model_kwargs.model_name, 'checkpoints/ABot-World-0-5-5B-Iris')
        self.assertTrue(fantasy.use_relative_rope and fantasy.model_kwargs.use_relative_rope)


if __name__ == '__main__':
    unittest.main()
