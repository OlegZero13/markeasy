import numpy as np

from ..utils import observable_norm
from ..layers import MarkovHiddenLayer
from ..definitions import ProbabilityVector, ProbabilityMatrix
from ..src import A, B, pi


hml = MarkovHiddenLayer(A, B, pi)

# tests if probabilities of all observations sum to 1.
def test__observables_norm_1():
    assert (observable_norm(hml, 1) - 1) < 1e-12

def test__observables_norm_2():
    assert (observable_norm(hml, 2) - 1) < 1e-12

def test__observables_norm_3():
    assert (observable_norm(hml, 3) - 1) < 1e-12

def test__observables_norm_4():
    assert (observable_norm(hml, 4) - 1) < 1e-12

# tests if score calculated through alpha and naively match
def test__hml_scores_4():
    obs = hml.run(4)
    score = hml.score(obs)
    score_naive = hml.score_naive(obs)
    assert abs(score - score_naive) < 1e-12

# tests if normalized alpha * beta vector is row stochastic
def test__hml_uncover():
    obs = hml.run(4)
    alphas = hml._alphas(obs)
    betas = hml._betas(obs)
    score = hml.score(obs)
    assert abs((alphas * betas / score).sum(axis=1).sum() - len(obs)) < 1e-12

def test__hml_digammas():
    obs = hml.run(4)
    digamma = hml._digammas(obs)
    assert abs(digamma.sum() + 1 - len(obs)) < 1e-12


