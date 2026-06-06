import random
import unittest

from SFT_stage.mix_hsp_sft import mix_examples


class MixHSPSFTTest(unittest.TestCase):
    def test_caps_replay_fraction_while_retaining_protocol_data(self):
        protocol = [{"id": f"protocol_{index}"} for index in range(4)]
        replay = [{"id": f"replay_{index}"} for index in range(20)]
        mixed, selected_replay_count = mix_examples(protocol, replay, 0.25, random.Random(0))
        self.assertEqual(selected_replay_count, 1)
        self.assertEqual(len(mixed), 5)
        self.assertEqual({item["id"] for item in mixed if item["id"].startswith("protocol_")}, {
            "protocol_0", "protocol_1", "protocol_2", "protocol_3"
        })

    def test_rejects_full_replay_replacement(self):
        with self.assertRaisesRegex(ValueError, "max_replay_fraction"):
            mix_examples([{"id": "protocol"}], [{"id": "replay"}], 1.0, random.Random(0))


if __name__ == "__main__":
    unittest.main()
