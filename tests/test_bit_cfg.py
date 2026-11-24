import unittest
from unittest.mock import Mock

import torch

from abot_world.bit.cli import REFERENCE_SLOTS, guided_flow, negative_conditions


class BitCFGTest(unittest.TestCase):
    def test_negative_uses_unrouted_text_and_drops_references_and_action_injection(self):
        positive = {
            'prompt_embeds': torch.ones(1, 3, 2, 4),
            'act_context': torch.ones(1, 32, 81, 1, 1),
            'first_frame_latents': object(),
            'ref_latents': object(),
            'ref_mask': object(),
            'cross_attn_frame_ranges': object(),
        }
        negative_text = torch.arange(8).reshape(1, 2, 4)
        zero_action = torch.zeros_like(positive['act_context'])
        negative = negative_conditions(positive, negative_text, zero_action)
        self.assertIs(negative['prompt_embeds'], negative_text)
        self.assertIs(negative['act_context'], zero_action)
        self.assertEqual(negative['act_context_scale'], 0.0)
        self.assertIs(negative['first_frame_latents'], positive['first_frame_latents'])
        for name in ('ref_latents', 'ref_mask', 'cross_attn_frame_ranges', 'cross_attn_num_frames'):
            self.assertNotIn(name, negative)
        self.assertEqual(torch.count_nonzero(positive['act_context']).item(), 32 * 81)
        self.assertEqual(REFERENCE_SLOTS, ('head', 'front', 'left', 'right', 'back'))

    def test_guidance_formula_and_conditional_only_path(self):
        cond = torch.tensor([5.0], dtype=torch.bfloat16)
        uncond = torch.tensor([2.0], dtype=torch.bfloat16)
        teacher = Mock(side_effect=[(cond, None), (uncond, None)])
        torch.testing.assert_close(guided_flow(teacher, None, None, {}, {}, 3),
                                   torch.tensor([11.0], dtype=torch.bfloat16))
        self.assertEqual(teacher.call_count, 2)
        teacher.reset_mock(side_effect=True)
        teacher.return_value = (cond, None)
        self.assertIs(guided_flow(teacher, None, None, {}, None, 1), cond)
        teacher.assert_called_once()
