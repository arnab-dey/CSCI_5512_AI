#######################################################################
# IMPORTS
#######################################################################
import numpy as np
#######################################################################
# Function definition
#######################################################################
#######################################################################
# Bias coin outcome simulator
# 1 represents head, 0 represents tail
#######################################################################
def tossBiasedCoin(prob_head):
    random_number = np.random.random()
    if (random_number <= prob_head):
        return 1.
    else:
        return 0.
#######################################################################
# Function to simulate fair coin from bias coin
# 1 represents head, 0 represents tail
#######################################################################
def simulateFairCoin(prob_head):
    #######################################################################
    # Get two consecutive bias coin outcome
    #######################################################################
    fair_coin_outcome = None
    while (None == fair_coin_outcome):
        bias_outcome_1 = tossBiasedCoin(prob_head=prob_head)
        bias_outcome_2 = tossBiasedCoin(prob_head=prob_head)
        if ((1. == bias_outcome_1) and (0. == bias_outcome_2)):
            fair_coin_outcome = 1.
        if ((0. == bias_outcome_1) and (1. == bias_outcome_2)):
            fair_coin_outcome = 0.
    return fair_coin_outcome
#######################################################################
# Function to simulate bias coin from fair coin
# 1 represents head, 0 represents tail
#######################################################################
def simulateBiasedCoin(actual_bias):
    #######################################################################
    # Get two consecutive bias coin outcome
    #######################################################################
    bias_coin_outcome = None
    while (None == bias_coin_outcome):
        fair_outcome_1 = simulateFairCoin(actual_bias)
        fair_outcome_2 = simulateFairCoin(actual_bias)
        if ((1. == fair_outcome_1) and (1. == fair_outcome_2)):
            bias_coin_outcome = 1.
        elif ((0. == fair_outcome_1) and (0. == fair_outcome_2)):
            bias_coin_outcome = None
        else:
            bias_coin_outcome = 0.
    return bias_coin_outcome
#######################################################################
# CODE STARTS HERE
#######################################################################
bias_prob_arr = [0.01, 0.1, 0.4]
n_samples = 10
fair_coin_samples = np.zeros((len(bias_prob_arr), n_samples))
bias_coin_samples = np.zeros((len(bias_prob_arr), n_samples))
np.random.seed(19)
print('########## 1. = Head, 0. = Tail ##########')
for prob_idx in range(len(bias_prob_arr)):
    for sample_idx in range(n_samples):
        fair_coin_samples[prob_idx, sample_idx] = simulateFairCoin(bias_prob_arr[prob_idx])
        bias_coin_samples[prob_idx, sample_idx] = simulateBiasedCoin(bias_prob_arr[prob_idx])
    #######################################################################
    # console log
    #######################################################################
    print('##########')
    print('Actual coin probability of head = ', bias_prob_arr[prob_idx])
    print('Simulated ', n_samples, ' fair coin samples = ', fair_coin_samples[prob_idx, :])
    print('Simulated ', n_samples, ' biased coin samples with 1/3 head probability = ',
          bias_coin_samples[prob_idx, :])
