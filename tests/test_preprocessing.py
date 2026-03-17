import unittest
from preprocessing import tokenise_corpus, build_vocab_and_mappings, get_training_pairs

class TestTokeniseCorpus(unittest.TestCase):
    def test_basic(self):
        corpus = ["Hello, World!"]
        result = tokenise_corpus(corpus)
        self.assertEqual(result, ["hello", "world"])

    def test_basic_with_numbers(self):
        corpus = ["3cats, 5 dogs"]
        result = tokenise_corpus(corpus)
        self.assertEqual(result, ["3cats", "5", "dogs"])

    def test_empty_corpus(self):
        result = tokenise_corpus([])
        self.assertEqual(result, [])

    def test_punctuation_stripped(self):
        result = tokenise_corpus(["hello!?! &world?'??"])
        self.assertEqual(result, ["hello", "world"])

    def test_multiple_spaces(self):
        result = tokenise_corpus(["hello   world"])
        self.assertEqual(result, ["hello", "world"])

class TestBuildVocabAndMappings(unittest.TestCase):
    def test_basic(self):
        tokens = ["the", "quick", "brown"]
        vocab, word_to_index, index_to_word = build_vocab_and_mappings(tokens)
        self.assertEqual(vocab, ["the", "quick", "brown"])
        self.assertEqual(word_to_index, {"the": 0, "quick": 1, "brown": 2})
        self.assertEqual(index_to_word, {0: "the", 1: "quick", 2: "brown"})

    def test_duplicate_words(self):
        tokens = ["the", "the", "quick", "brown", "quick"]
        vocab, word_to_index, index_to_word = build_vocab_and_mappings(tokens)

        self.assertEqual(vocab, ["the", "quick", "brown"])
        self.assertEqual(len(word_to_index), 3)
        self.assertEqual(word_to_index, {"the": 0, "quick": 1, "brown": 2})

    def test_empty_tokens(self):
        vocab, word_to_index, index_to_word = build_vocab_and_mappings([])

        self.assertEqual(vocab, [])
        self.assertEqual(word_to_index, {})
        self.assertEqual(index_to_word, {})

    def test_mappings_are_consistent(self):
        tokens = ["hello", "world", "2"]
        vocab, word_to_index, index_to_word = build_vocab_and_mappings(tokens)

        for word, idx in word_to_index.items():
             self.assertEqual(index_to_word[idx], word)

    def test_insertion_order_preserved(self):
        tokens = ["glaceon", "umbreon", "jolteon"]
        vocab, word_to_index, index_to_word = build_vocab_and_mappings(tokens)

        self.assertEqual(vocab[0], "glaceon")
        self.assertEqual(vocab[1], "umbreon")
        self.assertEqual(vocab[2], "jolteon")

class TestGetTrainingPairs(unittest.TestCase):
    def setUp(self):
        tokens = ["the", "quick", "brown", "fox", "jumped"]
        _, self.word_to_index, _ = build_vocab_and_mappings(tokens)
        self.tokens = tokens

    def test_center_word_pairs_with_context(self):
        pairs = get_training_pairs(self.tokens, self.word_to_index)
        
        self.assertIn((1, 0), pairs)
        self.assertIn((1, 2), pairs)

    def test_left_edge(self):
        pairs = get_training_pairs(self.tokens, self.word_to_index)

        the_idx = self.word_to_index["the"]
        quick_idx = self.word_to_index["quick"]
        jumped_idx = self.word_to_index["jumped"]

        self.assertIn((the_idx, quick_idx), pairs)
        self.assertNotIn((the_idx, jumped_idx), pairs)

    def test_right_edge(self):
        pairs = get_training_pairs(self.tokens, self.word_to_index)

        jumped_idx = self.word_to_index["jumped"]
        fox_idx = self.word_to_index["fox"]
        the_idx = self.word_to_index["the"]

        self.assertIn((jumped_idx, fox_idx), pairs)
        self.assertNotIn((jumped_idx, the_idx), pairs)  

    def test_empty_tokens(self):
        pairs = get_training_pairs([], self.word_to_index)
        self.assertEqual(pairs, [])

    def test_single_token(self):
        tokens = ["hello"]
        _, word_to_index, _ = build_vocab_and_mappings(tokens)
        pairs = get_training_pairs(tokens, word_to_index)
        self.assertEqual(pairs, [])


if __name__ == "__main__":
    unittest.main()
    