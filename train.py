from preprocessing import tokenise_corpus, build_vocab_and_mappings, get_training_pairs
import numpy as np
from model import Word2Vec
from config import EPOCHS

# with open("corpus.txt", "r") as f:
#     corpus = f.readlines()

with open("texts/mobydick.txt", "r") as f:
    text = f.read()

start = text.find("*** START") 
end = text.find("*** END")
corpus = [text[start:end]]
        
tokens = tokenise_corpus(corpus)
vocabulary, word_to_index, index_to_word = build_vocab_and_mappings(tokens)
training_pairs = get_training_pairs(tokens, word_to_index)

# training
model = Word2Vec(vocabulary)

for epoch in range(EPOCHS):
    total_loss = 0
    for pair in training_pairs:
        positive_prob, negative_samples, negative_probs = model.forward_pass(pair)

        # accumulate loss
        total_loss += model.loss(positive_prob, 1)
        for prob in negative_probs:
            total_loss += model.loss(prob, 0)

        # update weights
        model.update_weights(pair, positive_prob, negative_samples, negative_probs)
    print(f"Epoch {epoch}, Loss: {total_loss / len(training_pairs)}")
