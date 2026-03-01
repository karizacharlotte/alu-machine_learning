#!/usr/bin/env python3
"""
Defines function that determines if the Markov Chain is absorbing
"""
import numpy as np


def absorbing(P):
    """
    Determines if the Markov Chain is absorbing
    parameters:
        P [square 2D numpy.ndarray of shape (n, n)]:
            representing the standard transition matrix
            P[i, j] is the probability of transitioning from state i to state j
            n: the number of state in the Markov Chain
    returns:
        True, if absorbing
        False, if not absorbing or on failure
    """
    # check that P is the correct type and dimensions
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return False

    # save value of n and check that P is square
    n, n_check = P.shape
    if n != n_check:
        return False

    # Find absorbing states: P[i][i] == 1 means state i is absorbing
    diag = np.diag(P)
    absorbing_states = set(np.where(diag == 1)[0])

    # If no absorbing states, not absorbing
    if not absorbing_states:
        return False

    # Check every non-absorbing state can eventually reach an absorbing state
    reachable = set(absorbing_states)

    for _ in range(n):
        new_reachable = set()
        for state in range(n):
            if state not in reachable:
                # Can this state reach any state in reachable?
                if any(P[state, r] > 0 for r in reachable):
                    new_reachable.add(state)
        if not new_reachable:
            break
        reachable.update(new_reachable)

    return len(reachable) == n