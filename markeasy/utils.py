import numpy as np
import pandas as pd

from itertools import product

from markov.definitions import ProbabilityMatrix, ProbabilityVector
from markov.layers import MarkovHiddenLayer
from markov.src import A, B, pi


def create_all_observable_chains(hml: MarkovHiddenLayer, chain_length: int) -> list:
    return list(product(*(hml.observables,) * chain_length))

def grade_all_observable_chains(hml: MarkovHiddenLayer, chain_length: int) -> list:
    all_chains = create_all_observable_chains(hml, chain_length)
    scores = [0] * len(all_chains)

    for idx, chain in enumerate(all_chains):
        score = hml.score(chain)
        scores[idx] = score
    return list(zip(all_chains, scores))

def observable_norm(hml: MarkovHiddenLayer, chain_length: int) -> float:
    assert chain_length > 0, "The chain length must be greater than zero."
    grades = grade_all_observable_chains(hml, chain_length)
    return sum(list(map(lambda x: x[1], grades)))


if __name__ == '__main__':
    hml = MarkovHiddenLayer(A, B, pi)
    scores = grade_all_observable_chains(hml, 2)
