import re

WINDOW_SIZE: int = 4

def tokenise_corpus(corpus: list[str]) -> list[str]:
    tokens: list[str] = []
    for line in corpus:
        line = line.lower()
        line = re.sub(r'[^\w\s]', '', line)
        words = line.split()

        for word in words:
            tokens.append(word)

    return tokens

def build_vocab_and_mappings(tokens: list[str]) -> tuple[list[str], dict[str, int], dict[int, str]]:
    vocabulary: list[str] = [] 
    word_to_index: dict[str, int] = {}
    index_to_word: dict[int, str] = {}

    for token in tokens:
        if token not in word_to_index:
            word_to_index[token] = len(word_to_index)
            index_to_word[len(index_to_word)] = token
            vocabulary.append(token)

    return (vocabulary, word_to_index, index_to_word)

def get_training_pairs(tokens: list[str], word_to_index: dict[str, int]) -> list[tuple[int, int]]:
    training_pairs: list[tuple[int, int]] = []
    for pos in range(len(tokens)):
        center_index = word_to_index[tokens[pos]]
        for i in range(1, WINDOW_SIZE // 2 + 1):
            if pos - i >= 0:
                context_index: int = word_to_index[tokens[pos - i]]
                training_pairs.append((center_index, context_index))

            if pos + i < len(tokens):
                context_index: int = word_to_index[tokens[pos + i]]
                training_pairs.append((center_index, context_index))
    return training_pairs
