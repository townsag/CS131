from typing import Iterator, Tuple

import numpy as np
from itertools import product
import time
from numpy import float64, int32


#three questions:
#   - probability of observation sequence given model: p(O| lam)
#   - find optimal state sequence given model and observation sequence
#   - given observation sequence, # different states, # types of observations
#     find the model that maximizes the probability of the given observation sequence



def main():
    all_start = time.time()
    # matrix_A: list[list[int]] = list()
    A: np.ndarray[float64] = np.array([[0.7, 0.3],
                                       [0.4, 0.6]],
                                      dtype="float64")
    # matrix_B: list[list[int]] = list()
    B: np.ndarray[float64]= np.array([[0.1, 0.4, 0.5],
                                      [0.7, 0.2, 0.1]],
                                     dtype="float64")
    #matrix_pi: list[int] = list()
    pi: np.ndarray[float64] = np.array([[0.6, 0.4]], dtype="float64")

    O_1: np.ndarray[int32] = np.array([[0, 0, 0, 1]])

    # start1 = time.time()
    # print(score_observation(A,B,pi,O_1))
    # start2 = time.time()
    # print(f'took {start2 - start1} seconds')
    # print(score_observation_alpha_pass(A, B, pi, O_1))
    # end2 = time.time()
    # print(f'took {end2 - start2} seconds')
    # all_end = time.time()
    # print(f'the entire process took: {all_end - all_start}')
    find_hidden_states(A, B, pi, O_1)



def score_observation(a: np.ndarray, b: np.ndarray, pi: np.ndarray, observation_sequence: np.ndarray):
    # find probability of observation given model
    # sum the probability of the observation given the state sequence and the model for each state sequence
    # solution without forward pass algo
    sum: np.float64 = np.float64(0)
    num_states: int = a.shape[0]
    sequence_length = observation_sequence.shape[1]
    state_sequences: """iterator[tuple[int]]""" = product(range(0, num_states), repeat=sequence_length)
    for state_sequence in state_sequences:
        # sum += P(observation_sequence, state_sequence | given model)
        print(state_sequence)
        probability_product: np.float64 = np.float64(1)

        pi_x_0 = pi[0][state_sequence[0]]
        prob_initial_observation = b[state_sequence[0]][observation_sequence[0][0]]
        prob_first_transition = a[state_sequence[0]][state_sequence[1]]
        probability_product *= pi_x_0 * prob_initial_observation * prob_first_transition

        for i in range(1, sequence_length - 1):
            prob_ith_observation = b[state_sequence[i]][observation_sequence[0][i]]
            prob_ith_transition = a[state_sequence[i]][state_sequence[i+1]]
            probability_product *= prob_ith_observation * prob_ith_transition

        prob_last_observation = b[state_sequence[sequence_length - 1]][observation_sequence[0][sequence_length - 1]]
        probability_product *= prob_last_observation

        sum += probability_product

    return sum


def score_observation_alpha_pass(a: np.ndarray, b: np.ndarray, pi: np.ndarray, observation_sequence: np.ndarray):
    sequence_length: int = observation_sequence.shape[1]
    num_states: int = a.shape[0]
    alpha: np.ndarray[float64] = np.zeros((sequence_length,num_states), dtype=float64)

    for i in range(0, num_states):
        alpha[0, i] = pi[0][i] * b[i][observation_sequence[0][0]]

    for t in range(1, sequence_length):
        for i in range(0, num_states):
            my_sum: np.float64 = np.float64(0)
            for j in range(0, num_states):
                my_sum += alpha[t - 1][j] * a[j][i]
            alpha[t][i] = my_sum * b[i][observation_sequence[0][t]]
            # alpha[t][i] = np.sum([alpha[t - 1][j] * a[j][i] for j in range(0, num_states)]) * b[i][t]

    return np.sum([alpha[sequence_length - 1][i] for i in range(0, num_states)]), alpha


def find_hidden_states(a: np.ndarray, b: np.ndarray, pi: np.ndarray, observation_sequence: np.ndarray):
    sequence_length: int = observation_sequence.shape[1]
    num_states: int = a.shape[0]
    beta: np.ndarray[float64] = np.zeros((sequence_length, num_states))

    for i in range(0, num_states):
        beta[sequence_length - 1][i] = 1

    for t in range(sequence_length - 2, 0 - 1, -1):
        for i in range(0, num_states):
            my_sum: np.float64 = np.float64(0)
            for j in range(0, num_states):
                my_sum += a[i][j] * b[j][observation_sequence[0][t + 1]] * beta[t + 1][j]
            beta[t][i] = my_sum

    # calculate P(observation sequence| model)
    score, alpha = score_observation_alpha_pass(a, b, pi, observation_sequence)
    gamma: np.array[float64] = np.zeros((sequence_length, num_states))
    for t in range(0, sequence_length):
        for i in range(0, num_states):
            gamma[t][i] = alpha[t][i] * beta[t][i] / score

    print(gamma)
    hidden_states: np.ndarray[int32] = np.zeros((1, sequence_length))
    for i in range(0, sequence_length):
        hidden_states[0][i] = np.argmax(gamma[i])
    print(hidden_states)


def get_alpha_and_c(a: np.ndarray, b: np.ndarray, pi: np.ndarray, observation_sequence: np.ndarray):
    sequence_length: int = observation_sequence.shape[1]
    num_states: int = a.shape[0]
    alpha: np.ndarray[float64] = np.zeros((sequence_length,num_states), dtype=float64)
    scaling_constants: np.ndarray[float64] = np.zeros((1,sequence_length), dtype='float64')

    for i in range(0, num_states):
        alpha[0, i] = pi[0][i] * b[i][observation_sequence[0][0]]
        scaling_constants[0, 0] += alpha[0, i]
    scaling_constants[0, 0] = 1/scaling_constants[0, 0]

    for i in range(0, num_states):
        alpha[0, i] *= scaling_constants[0,0]

    for t in range(1, sequence_length):
        for i in range(0, num_states):
            my_sum: np.float64 = np.float64(0)
            for j in range(0, num_states):
                my_sum += alpha[t - 1][j] * a[j][i]

            alpha[t][i] = my_sum * b[i][observation_sequence[0][t]]
            scaling_constants[0, t] += alpha[t][i]

        scaling_constants[0, t] = 1 / scaling_constants[0, t]

        for i in range(0, num_states):
            alpha[t][i] *= scaling_constants[0, t]

    return alpha, scaling_constants


def get_beta(a: np.ndarray, b: np.ndarray, observation_sequence: np.ndarray, c: np.ndarray):
    sequence_length: int = observation_sequence.shape[1]
    num_states: int = a.shape[0]
    beta: np.ndarray[float64] = np.zeros((sequence_length, num_states))

    for i in range(0, num_states):
        beta[sequence_length - 1][i] = c[0, sequence_length - 1]

    for t in range(sequence_length - 2, 0 - 1, -1):
        for i in range(0, num_states):
            my_sum: np.float64 = np.float64(0)
            for j in range(0, num_states):
                my_sum += a[i][j] * b[j][observation_sequence[0][t + 1]] * beta[t + 1][j]
            beta[t][i] = my_sum * c[0, t]

    return beta


def get_beta_gamma_degamma(a: np.ndarray, b: np.ndarray, alpha: np.ndarray, beta: np.ndarray, observation_sequence: np.ndarray):
    sequence_length: int = observation_sequence.shape[1]
    num_states: int = a.shape[0]
    # alpha, c = get_alpha_and_c(a, b, pi, observation_sequence)
    # beta: np.ndarray = get_beta(a, b, observation_sequence, c)

    gamma: np.ndarray[float64] = np.zeros((sequence_length, num_states))
    degamma: np.ndarray[float64] = np.zeros((sequence_length, num_states, num_states))
    for t in range(0, sequence_length - 1):
        denominator: np.float64 = np.float64(0)
        for i in range(0, num_states):
            for j in range(0, num_states):
                denominator += alpha[t][j] * a[i][j] * b[j][observation_sequence[t + 1]] * beta[t + 1][j]
        for i in range(0, num_states):
            gamma[t][i]: np.float64 = np.float64(0)
            for j in range(0, num_states):
                degamma[t][i][j] = (alpha[t][i] * a[i][j] * b[j][observation_sequence[t+1]] * beta[t+1][j]) / denominator
                gamma[t][i] = gamma[t][i] + degamma[t][i][j]

    # special case for gamma(T - 1)(i)
    denominator: np.float64 = np.float64(0)
    for i in range(0, num_states):
        denominator += alpha[sequence_length - 1][i]

    for i in range(0, num_states):
        gamma[sequence_length - 1][i] = alpha[sequence_length - 1][i] / denominator

    return gamma, degamma


def train_model(observation_sequence: np.ndarray[int32], num_states: int, num_observations: int,
                model_tup: Tuple[np.ndarray, np.ndarray, np.ndarray] = None):
    if model_tup is None:
        # create randomized inputs for pi, a, and b using the N and M input by the user
        pi: np.ndarray[int32] = np.random.dirichlet(np.ones(num_states)*1000., size=1)
        assert pi.shape[1] == num_states and pi.shape[0] == 1, f'the shape of pi is {pi.shape} and the num states is: {num_states}'
        a: np.ndarray[float64] = np.random.dirichlet(np.ones(num_states)*1000., size=num_states)
        assert a.shape[0] == num_states and a.shape[1] == num_states, \
            f'the shape of a is {a.shape} and the num states is: {num_states}'
        b: np.ndarray[float64] = np.random.dirichlet(np.ones(num_observations)*1000., size=num_states)
        assert b.shape[0] == num_states and b.shape[1] == num_observations, \
            f'the shape of b is {b.shape} and the num states is: {num_states} and the num_observations is: {num_observations}'
    else:
        pi: np.ndarray[int32] = model_tup[0]
        assert pi.shape[1] == num_states and pi.shape[0] == 1, f'the shape of pi is {pi.shape} and the num states is: {num_states}'
        a: np.ndarray[float64] = model_tup[1]
        assert a.shape[0] == num_states and a.shape[1] == num_states, \
            f'the shape of a is {a.shape} and the num states is: {num_states}'
        b: np.ndarray[float64] = model_tup[2]
        assert b.shape[0] == num_states and b.shape[1] == num_observations, \
            f'the shape of b is {b.shape} and the num states is: {num_states} and the num_observations is: {num_observations}'



if __name__ == "__main__":
    main()

"""
def get_beta_gamma_degamma(a: np.ndarray, b: np.ndarray, pi: np.ndarray, observation_sequence: np.ndarray):
    sequence_length: int = observation_sequence.shape[1]
    num_states: int = a.shape[0]
    alpha, c = get_alpha_and_c(a, b, pi, observation_sequence)
    beta: np.ndarray = get_beta(a, b, observation_sequence, c)

    gamma: np.ndarray[float64] = np.zeros((sequence_length, num_states))
    degamma: np.ndarray[float64] = np.zeros((sequence_length, num_states, num_states))
    for t in range(0, sequence_length - 1):
        denominator: np.float64 = np.float64(0)
        for i in range(0, num_states):
            for j in range(0, num_states):
                denominator += alpha[t][j] * a[i][j] * b[j][observation_sequence[t + 1]] * beta[t + 1][j]
        for i in range(0, num_states):
            gamma[t][i]: np.float64 = np.float64(0)
            for j in range(0, num_states):
                degamma[t][i][j] = (alpha[t][i] * a[i][j] * b[j][observation_sequence[t+1]] * beta[t+1][j]) / denominator
                gamma[t][i] = gamma[t][i] + degamma[t][i][j]

    # special case for gamma(T - 1)(i)
    denominator: np.float64 = np.float64(0)
    for i in range(0, num_states):
        denominator += alpha[sequence_length - 1][i]

    for i in range(0, num_states):
        gamma[sequence_length - 1][i] = alpha[sequence_length - 1][i] / denominator

    return gamma, degamma
"""

# def get_beta_gamma_degamma(a: np.ndarray, b: np.ndarray, pi: np.ndarray, observation_sequence: np.ndarray):
#     sequence_length: int = observation_sequence.shape[1]
#     num_states: int = a.shape[0]
#     alpha, c = get_alpha_and_c(a, b, pi, observation_sequence)
#     # beta: np.ndarray[float64] = np.zeros((sequence_length, num_states))
#     #
#     # for i in range(0, num_states):
#     #     beta[sequence_length - 1][i] = c[0, sequence_length - 1]
#     #
#     # for t in range(sequence_length - 2, 0 - 1, -1):
#     #     for i in range(0, num_states):
#     #         my_sum: np.float64 = np.float64(0)
#     #         for j in range(0, num_states):
#     #             my_sum += a[i][j] * b[j][observation_sequence[0][t + 1]] * beta[t + 1][j]
#     #         beta[t][i] = my_sum * c[0, t]
#     beta: np.ndarray = get_beta(a, b, observation_sequence, c)
#
# gamma: np.ndarray[float64] = np.zeros((sequence_length, num_states)) degamma: np.ndarray[float64] = np.zeros((
# sequence_length,num_states,num_states)) for t in range(0, sequence_length - 1): denominator: np.float64 =
# np.float64(0) for i in range(0, num_states): for j in range(0, num_states): denominator += alpha[t][j] * a[i][j] *
# b[j][observation_sequence[t + 1]] * beta[t + 1][j] for i in range(0, num_states): gamma[t][i]: np.float64 =
# np.float64(0) for j in range(0, num_states): degamma[t][i][j] = (alpha[t][i] * a[i][j] * b[j][observation_sequence[
# t+1]] * beta[t+1][j]) / denominator gamma[t][i] = gamma[t][i] + degamma[t][i][j]
#
#     # special case for gamma(T - 1)(i)
#     denominator: np.float64 = np.float64(0)
#     for i in range(0, num_states):
#         denominator += alpha[sequence_length - 1][i]
#
#     for i in range(0, num_states):
#         gamma[sequence_length - 1][i] = alpha[sequence_length - 1][i] / denominator
#
#     return beta, gamma, degamma