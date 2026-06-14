import unittest

from scripts.token_probe_hsp import build_probe_prefix, collect_probes


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        content = "\n".join(message["content"] for message in messages)
        return f"USER:{content}\nASSISTANT:"

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))


class TokenProbeTest(unittest.TestCase):
    def test_build_prefix_stops_before_action_token(self):
        example = {
            "id": "ex1",
            "sample_type": "ask_help",
            "segments": [
                {"source": "user", "text": "Question", "loss": False},
                {"source": "student", "text": "I need help.\n<ASK>64</ASK>", "loss": True},
            ],
        }
        probe = build_probe_prefix(FakeTokenizer(), example, "<ASK>")
        self.assertIsNotNone(probe)
        self.assertTrue(probe["prefix"].endswith("I need help.\n"))
        self.assertNotIn("<ASK>", probe["prefix"])
        self.assertEqual(probe["expected_token"], "<ASK>")

    def test_collects_one_probe_per_action(self):
        examples = [
            {
                "id": "ask",
                "sample_type": "ask_help",
                "segments": [
                    {"source": "user", "text": "Q", "loss": False},
                    {"source": "student", "text": "<ASK>64</ASK>", "loss": True},
                ],
            },
            {
                "id": "verify",
                "sample_type": "verify_confirm",
                "segments": [
                    {"source": "user", "text": "Q", "loss": False},
                    {"source": "student", "text": "draft\n<VERIFY>96</VERIFY>", "loss": True},
                ],
            },
        ]
        probes = collect_probes(
            FakeTokenizer(),
            examples,
            actions=["<ASK>", "<VERIFY>"],
            probes_per_action=1,
            max_prompt_tokens=4096,
        )
        self.assertEqual([probe["expected_token"] for probe in probes], ["<ASK>", "<VERIFY>"])


if __name__ == "__main__":
    unittest.main()
