import numpy as np
import pandas as pd

from itertools import product
from functools import reduce

from .definitions import ProbabilityMatrix, ProbabilityVector
from .src import A, B, pi


class MarkovHiddenLayer:
    def __init__(self, T, E, pi):
        
        self.T           = T
        self.E           = E
        self.pi          = pi
        self.states      = pi.states
        self.observables = E.observables

    def __repr__(self):
        return "HML states: {} -> observables: {}.".format(
            len(self.states), len(self.observables))

    @classmethod
    def initialize(cls, states: list, observables: list):
        T = ProbabilityMatrix.initialize(states, states)
        E = ProbabilityMatrix.initialize(states, observables)
        pi = ProbabilityVector.initialize(states)
        return cls(T, E, pi)

    def run(self, length: int, exposed: bool = False) -> (list, list):
        assert length >= 0, "The chain's length needs to be a non-negative number."
        s_history = [0] * (length + 1)
        o_history = [0] * (length + 1)

        prb = self.pi.values
        obs = prb @ self.E.values
        
        s_history[0] = np.random.choice(self.states, p=prb.flatten())
        o_history[0] = np.random.choice(self.observables, p=obs.flatten())

        for t in range(1, length + 1):
            prb = prb @ self.T.values
            obs = prb @ self.E.values

            s_history[t] = np.random.choice(self.states, p=prb.flatten())
            o_history[t] = np.random.choice(self.observables, p=obs.flatten())
        
        if exposed:
            return o_history, s_history
        return o_history

    def _alphas(self, observations: list) -> np.ndarray:
        alphas = np.zeros((len(observations), len(self.states)))
        alphas[0, :] = self.pi.values * self.E[observations[0]].T
        for t in range(1, len(observations)):
            alphas[t, :] = (alphas[t - 1, :].reshape(1, -1) @ self.T.values) * self.E[observations[t]].T
        return alphas

    def _betas(self, observations: list) -> np.ndarray:
        betas = np.zeros((len(observations), len(self.states)))
        betas[-1, :] = 1
        for t in range(len(observations) - 2, -1, -1):
            betas[t, :] = (self.T.values @ (self.E[observations[t + 1]] \
                        * betas[t + 1, :].reshape(-1, 1))).reshape(1, -1)
        return betas

    def _b2(self, observations: list) -> np.ndarray:
        betas = np.zeros((len(observations), len(self.observables), len(self.states)))
        betas[-1, :, :] = 1
        for t in range(len(observations) - 2, -1, -1):
            for i, o in enumerate(self.observables):
                betas[t, i, :] = (self.T.values @ (self.E[o]) \
                           * betas[t + 1, i, :].reshape(-1, 1)).reshape(1, -1)
        return betas

    def _digammas(self, observations: list) -> np.ndarray:
        L, N = len(observations), len(self.states)
        digammas = np.zeros((L - 1, N, N))

        alphas = self._alphas(observations)
        betas = self._betas(observations)
        score = self.score(observations)
        for t in range(L - 1):
            P1 = (alphas[t, :].reshape(-1, 1) * self.T.values)
            P2 = self.E[observations[t + 1]].T * betas[t + 1].reshape(1, -1)
            digammas[t, :, :] = P1 * P2 / score
        return digammas

    def uncover(self, observations: list) -> list:
        alphas = self._alphas(observations)
        betas = self._betas(observations)
        maxargs = (alphas * betas).argmax(axis=1)
        return list(map(lambda x: self.states[x], maxargs))

    def score(self, observations: list) -> float:
        alphas = self._alphas(observations)
        return float(alphas[-1].sum())
    
    def score_naive(self, observations: list) -> float:
        def mul(x, y): return x * y

        score = 0
        all_chains = list(product(*(self.states,) * len(observations)))
        for idx, chain in enumerate(all_chains):
            expanded_chain = list(zip(chain, [self.T.states[0]] + list(chain)))
            expanded_obser = list(zip(observations, chain))

            p_observations = list(map(lambda x: self.E.df.loc[x[1], x[0]], expanded_obser))
            p_hidden_state = list(map(lambda x: self.T.df.loc[x[1], x[0]], expanded_chain))
            p_hidden_state[0] = self.pi[chain[0]]
            
            score += reduce(mul, p_observations) * reduce(mul, p_hidden_state)
        return score




if __name__ == '__main__':
    hml = MarkovHiddenLayer(A, B, pi)
    obs, sts = hml.run(5, True)
    a = hml._alphas(obs)
    b = hml._betas(obs)
    d = hml._digammas(obs)
    s = hml.score(obs)
