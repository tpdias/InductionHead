"""Markov chain generator (week 1)."""
import numpy as np


def make_kernel(num_states, concentration, rng):
    """Return a (num_states, num_states) transition matrix; row i is P(next | current=i).

    Small concentration -> sharp rows (easy to learn), large -> near-uniform rows.
    """
    alphas = concentration * np.ones(num_states)
    rows = []
    for _ in range(num_states):
        rows.append(rng.dirichlet(alphas))
    return np.array(rows)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    P = make_kernel(4, 0.3, rng)
    print(P)
    print(P.sum(axis=1))
    assert np.allclose(P.sum(axis=1), 1)