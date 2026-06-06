import unittest

from SFT_stage.hsp_collator import HSPDataCollator, IGNORE_INDEX


class CharacterTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    padding_side = "right"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        content = "\n".join(message["content"] for message in messages)
        return f"USER:{content}\nASSISTANT:"

    def __call__(self, text, **kwargs):
        max_length = kwargs.get("max_length", len(text))
        limited_text = text[:max_length]
        return {
            "input_ids": [ord(char) + 10 for char in limited_text],
            "offset_mapping": [(index, index + 1) for index in range(len(limited_text))],
        }


class HSPCollatorMaskTest(unittest.TestCase):
    def test_only_student_spans_have_labels(self):
        example = {
            "segments": [
                {"source": "user", "text": "Question", "loss": False},
                {"source": "student", "text": "draft <VERIFY>", "loss": True},
                {"source": "teacher", "text": "<TEACHER_REVIEW>wrong</TEACHER_REVIEW>", "loss": False},
                {"source": "student", "text": "<ACCEPT> final", "loss": True},
            ]
        }
        collator = HSPDataCollator(CharacterTokenizer(), max_length=4096, append_eos_token=False)
        text, spans, _ = collator._render_example(example)
        encoded = collator.encode_example(example)

        teacher_start = text.index("<TEACHER_REVIEW>")
        teacher_end = text.index("</TEACHER_REVIEW>") + len("</TEACHER_REVIEW>")
        self.assertTrue(all(encoded["labels"][index] == IGNORE_INDEX for index in range(teacher_start, teacher_end)))
        for start, end in spans:
            self.assertTrue(all(encoded["labels"][index] != IGNORE_INDEX for index in range(start, end)))

    def test_policy_action_tokens_are_trained(self):
        example = {
            "segments": [
                {"source": "user", "text": "Question", "loss": False},
                {"source": "student", "text": "<ASK>", "loss": True},
            ]
        }
        collator = HSPDataCollator(CharacterTokenizer(), max_length=4096, append_eos_token=False)
        text, _, _ = collator._render_example(example)
        encoded = collator.encode_example(example)
        action_start = text.index("<ASK>")
        self.assertTrue(all(encoded["labels"][index] != IGNORE_INDEX for index in range(action_start, len(text))))

    def test_teacher_segment_cannot_be_marked_trainable(self):
        example = {
            "segments": [
                {"source": "user", "text": "Question", "loss": False},
                {"source": "teacher", "text": "feedback", "loss": True},
            ]
        }
        collator = HSPDataCollator(CharacterTokenizer(), max_length=4096)
        with self.assertRaisesRegex(ValueError, "Only source=student"):
            collator.encode_example(example)

    def test_truncated_interaction_is_marked_incomplete(self):
        example = {
            "segments": [
                {"source": "user", "text": "Question", "loss": False},
                {"source": "student", "text": "draft <VERIFY>", "loss": True},
                {"source": "teacher", "text": "<TEACHER_REVIEW>correction</TEACHER_REVIEW>", "loss": False},
                {"source": "student", "text": "<ACCEPT> final", "loss": True},
            ]
        }
        prefix = "USER:Question\nASSISTANT:draft <VERIFY>"
        collator = HSPDataCollator(CharacterTokenizer(), max_length=len(prefix), append_eos_token=False)
        encoded = collator.encode_example(example)
        self.assertFalse(encoded["interaction_complete"])

    def test_truncated_non_interaction_solution_remains_trainable(self):
        example = {
            "segments": [
                {"source": "user", "text": "Question", "loss": False},
                {"source": "student", "text": "A long independent answer", "loss": True},
            ]
        }
        prefix_with_partial_answer = "USER:Question\nASSISTANT:A long"
        collator = HSPDataCollator(
            CharacterTokenizer(), max_length=len(prefix_with_partial_answer), append_eos_token=False
        )
        encoded = collator.encode_example(example)
        self.assertTrue(encoded["interaction_complete"])
        self.assertTrue(any(label != IGNORE_INDEX for label in encoded["labels"]))


if __name__ == "__main__":
    unittest.main()
