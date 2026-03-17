import numpy as np
from config import EMBEDDING_DIM, WEIGHT_INIT_SCALE, LEARNING_RATE
from negative_sampling import get_negative_samples

class Word2Vec:
    def __init__(
            self,
            vocabulary: list[str],
            ):
        self.vocabulary = vocabulary
        self.w_embedding, self.w_context = self.__initialise_weights(self.vocab_size)

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)

    def __initialise_weights(
            self, 
            vocab_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        w_embedding = np.random.randn(vocab_size, EMBEDDING_DIM) * WEIGHT_INIT_SCALE
        w_context = np.random.randn(vocab_size, EMBEDDING_DIM) * WEIGHT_INIT_SCALE
        return w_embedding, w_context

    def forward_pass(
            self,
            pair: tuple[int, int],
    ) -> tuple[float, list[int], list[float]]:
        # handle forward pass for positive sample
        target_idx, context_idx = pair
        target_vector = self.w_embedding[target_idx]
        context_vector = self.w_context[context_idx]

        # woohoo dot product and sigmoid
        dot_product = np.dot(target_vector, context_vector)
        positive_prob = self.__sigmoid(dot_product)

        # handle forward pass for negative samples
        negative_probs = []
        negative_samples = get_negative_samples(target_idx, self.vocab_size)
        for sample in negative_samples:
            sample_vector = np.array(self.w_context[sample])

            dot_product = np.dot(target_vector, sample_vector)
            negative_prob = self.__sigmoid(dot_product)
            negative_probs.append(negative_prob)

        return positive_prob, negative_samples, negative_probs

    def loss(self, prob: float, label: int):
        prob = np.clip(prob, 1e-10, 1 - 1e-10) # avoid log(0)!!
        return -(label * np.log(prob) + (1 - label) * np.log(1 - prob))

    def update_weights(
            self,
            pair: tuple[int, int],
            positive_prob: float,
            negative_samples: list[int],
            negative_probs: list[float]
            ):
        target_idx, context_idx = pair
        target_vector = self.w_embedding[target_idx].copy()

        # positive grad descent 
        error = positive_prob - 1
        context_vector = self.w_context[context_idx]
        self.w_embedding[target_idx] -= LEARNING_RATE * error * context_vector
        self.w_context[context_idx] -= LEARNING_RATE * error * target_vector

        # negative grad descent
        for i in range(len(negative_samples)):
            sample = negative_samples[i]
            error = negative_probs[i] - 0

            sample_vector = self.w_context[sample]
            self.w_embedding[target_idx] -= LEARNING_RATE * error * sample_vector
            self.w_context[sample] -= LEARNING_RATE * error * target_vector

    def __sigmoid(self, x):
        return np.divide(1, 1 + np.exp(-x))