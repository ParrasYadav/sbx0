"""A simple Hidden Markov Model implementation without third-party dependencies."""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple


class HiddenMarkovModel:
    """Simple discrete Hidden Markov Model.

    This implementation relies only on the Python standard library. Probabilities
    are stored as floating point numbers and automatically normalised.
    """

    def __init__(self, n_states: int, n_observations: int) -> None:
        self.n_states = n_states
        self.n_observations = n_observations
        self.start_probs = self._random_prob_vector(n_states)
        self.trans_probs = [self._random_prob_vector(n_states) for _ in range(n_states)]
        self.emit_probs = [self._random_prob_vector(n_observations) for _ in range(n_states)]

    @staticmethod
    def _random_prob_vector(n: int) -> List[float]:
        values = [random.random() for _ in range(n)]
        total = sum(values)
        return [v / total for v in values]

    def _forward(self, obs_seq: Sequence[int]) -> Tuple[float, List[List[float]]]:
        T = len(obs_seq)
        alpha = [[0.0 for _ in range(self.n_states)] for _ in range(T)]
        for i in range(self.n_states):
            alpha[0][i] = self.start_probs[i] * self.emit_probs[i][obs_seq[0]]
        for t in range(1, T):
            for j in range(self.n_states):
                s = 0.0
                for i in range(self.n_states):
                    s += alpha[t - 1][i] * self.trans_probs[i][j]
                alpha[t][j] = s * self.emit_probs[j][obs_seq[t]]
        prob = sum(alpha[T - 1])
        return prob, alpha

    def _backward(self, obs_seq: Sequence[int]) -> Tuple[float, List[List[float]]]:
        T = len(obs_seq)
        beta = [[0.0 for _ in range(self.n_states)] for _ in range(T)]
        for i in range(self.n_states):
            beta[T - 1][i] = 1.0
        for t in range(T - 2, -1, -1):
            for i in range(self.n_states):
                s = 0.0
                for j in range(self.n_states):
                    s += self.trans_probs[i][j] * self.emit_probs[j][obs_seq[t + 1]] * beta[t + 1][j]
                beta[t][i] = s
        prob = 0.0
        for i in range(self.n_states):
            prob += self.start_probs[i] * self.emit_probs[i][obs_seq[0]] * beta[0][i]
        return prob, beta

    def sequence_probability(self, obs_seq: Sequence[int]) -> float:
        prob, _ = self._forward(obs_seq)
        return prob

    def viterbi(self, obs_seq: Sequence[int]) -> Tuple[float, List[int]]:
        T = len(obs_seq)
        delta = [[0.0 for _ in range(self.n_states)] for _ in range(T)]
        psi = [[0 for _ in range(self.n_states)] for _ in range(T)]
        for i in range(self.n_states):
            delta[0][i] = self.start_probs[i] * self.emit_probs[i][obs_seq[0]]
            psi[0][i] = 0
        for t in range(1, T):
            for j in range(self.n_states):
                best_val = -1.0
                best_state = 0
                for i in range(self.n_states):
                    val = delta[t - 1][i] * self.trans_probs[i][j]
                    if val > best_val:
                        best_val = val
                        best_state = i
                delta[t][j] = best_val * self.emit_probs[j][obs_seq[t]]
                psi[t][j] = best_state
        max_prob = max(delta[T - 1])
        state = delta[T - 1].index(max_prob)
        path = [state]
        for t in range(T - 1, 0, -1):
            state = psi[t][state]
            path.insert(0, state)
        return max_prob, path

    def baum_welch(self, sequences: Sequence[Sequence[int]], n_iter: int = 10) -> None:
        for _ in range(n_iter):
            start_sum = [0.0 for _ in range(self.n_states)]
            trans_sum = [[0.0 for _ in range(self.n_states)] for _ in range(self.n_states)]
            trans_count = [0.0 for _ in range(self.n_states)]
            emit_sum = [[0.0 for _ in range(self.n_observations)] for _ in range(self.n_states)]
            emit_count = [0.0 for _ in range(self.n_states)]

            for obs_seq in sequences:
                prob, alpha = self._forward(obs_seq)
                _, beta = self._backward(obs_seq)
                T = len(obs_seq)

                gamma = [[0.0 for _ in range(self.n_states)] for _ in range(T)]
                xi = [[[0.0 for _ in range(self.n_states)] for _ in range(self.n_states)] for _ in range(T - 1)]

                for t in range(T):
                    norm = 0.0
                    for i in range(self.n_states):
                        gamma[t][i] = alpha[t][i] * beta[t][i]
                        norm += gamma[t][i]
                    if norm == 0.0:
                        norm = 1.0
                    for i in range(self.n_states):
                        gamma[t][i] /= norm
                for t in range(T - 1):
                    norm = 0.0
                    for i in range(self.n_states):
                        for j in range(self.n_states):
                            xi[t][i][j] = (
                                alpha[t][i]
                                * self.trans_probs[i][j]
                                * self.emit_probs[j][obs_seq[t + 1]]
                                * beta[t + 1][j]
                            )
                            norm += xi[t][i][j]
                    if norm == 0.0:
                        norm = 1.0
                    for i in range(self.n_states):
                        for j in range(self.n_states):
                            xi[t][i][j] /= norm

                for i in range(self.n_states):
                    start_sum[i] += gamma[0][i]
                    for t in range(T - 1):
                        trans_count[i] += gamma[t][i]
                        for j in range(self.n_states):
                            trans_sum[i][j] += xi[t][i][j]
                    for t in range(T):
                        emit_count[i] += gamma[t][i]
                        emit_sum[i][obs_seq[t]] += gamma[t][i]

            # Normalise to update parameters
            total = sum(start_sum)
            if total == 0.0:
                total = 1.0
            self.start_probs = [s / total for s in start_sum]

            for i in range(self.n_states):
                denom = trans_count[i]
                if denom == 0.0:
                    denom = 1.0
                self.trans_probs[i] = [trans_sum[i][j] / denom for j in range(self.n_states)]
                denom = emit_count[i]
                if denom == 0.0:
                    denom = 1.0
                self.emit_probs[i] = [emit_sum[i][k] / denom for k in range(self.n_observations)]

    def generate(self, length: int) -> Tuple[List[int], List[int]]:
        state = random.choices(range(self.n_states), weights=self.start_probs)[0]
        states = [state]
        observations = []
        for _ in range(length):
            obs = random.choices(range(self.n_observations), weights=self.emit_probs[state])[0]
            observations.append(obs)
            state = random.choices(range(self.n_states), weights=self.trans_probs[state])[0]
            states.append(state)
        return observations, states[:-1]
