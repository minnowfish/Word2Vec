import unittest
from negative_sampling import get_negative_samples
from preprocessing import build_vocab_and_mappings
from config import NEGATIVE_SAMPLES

class TestGetNegativeSamples(unittest.TestCase):
    def setUp(self):
        tokens = ["the", "quick", "brown", "fox", "jumped"]
        self.vocab, self.word_to_index, _ = build_vocab_and_mappings(tokens)
        self.tokens = tokens

    def test_basic(self):
        target_idx = self.word_to_index["quick"]
        negative_samples_quick = get_negative_samples(target_idx, len(self.vocab), 4)

        self.assertNotIn(target_idx, negative_samples_quick)
        self.assertEqual(len(negative_samples_quick), 4)

    def test_neg_samples_greater_than_no_words(self):
        tokens = ["the", "quick"]
        target_idx = self.word_to_index["quick"]
        negative_samples_quick = get_negative_samples(target_idx, len(tokens), 4)

        self.assertNotIn(target_idx, negative_samples_quick)
        self.assertEqual(len(negative_samples_quick), 1)

    def test_empty_tokens(self):
        tokens = []
        negative_samples = get_negative_samples(-1, len(tokens), 4)

        self.assertEqual(len(negative_samples), 0)


if __name__ == "__main__":
    unittest.main()
