import numpy as np

from markov.models import HiddenMarkovModel
from markov.layers import MarkovHiddenLayer
from markov.src import A, B, pi


if __name__ == '__main__':
    states = list('abcd')
    observables = list('XYZ')
    observations = ['X'] * 10 + ['Y'] * 7 + ['Z'] * 3

    hmm = HiddenMarkovModel.initialize(states, observables)
    hmm.train(observations, epochs=200, tol=2e-3)



