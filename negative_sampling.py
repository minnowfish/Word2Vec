from config import NEGATIVE_SAMPLES
from random import randint

def get_negative_samples(
        target_idx: int, 
        vocab_size: int,
        no_negative_sample: int = NEGATIVE_SAMPLES,
        ) -> list[int]:
    negative_samples: list[int] = []
    for i in range(min(no_negative_sample, vocab_size - 1)):
        other_idx = target_idx

        while other_idx == target_idx:
            other_idx = randint(0, vocab_size - 1)

        negative_samples.append(other_idx)

    return negative_samples
