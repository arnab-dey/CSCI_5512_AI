#######################################################################
# IMPORTS
#######################################################################
import numpy as np
#######################################################################
# Policy matrix
#######################################################################
n_action = 4
n_state = 12
n_obs = 4
n_sample = 1000
P_a_s = np.array([[0.9, 0.05, 0.9, 0.05, 0., 0., 0., 0.05, 0.25, 0.05, 0.05, 0.05],
        [0., 0.05, 0., 0.05, 0.9, 0.9, 0.9, 0.05, 0.25, 0.05, 0.05, 0.05],
        [0.05, 0., 0.05, 0., 0.05, 0.05, 0.05, 0., 0.25, 0.9, 0., 0.9],
        [0.05, 0.9, 0.05, 0.9, 0.05, 0.05, 0.05, 0.9, 0.25, 0., 0.9, 0.]])
#######################################################################
# Transition matrix
#######################################################################
T_sbar_a_s = np.array([
    #up
    [[0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]],
    #down
    [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]],
    # left
    [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
    [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]],
    # right
    [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
])
#######################################################################
# Markov dynamics calculation
#######################################################################
P_sbar_s_a = np.zeros(T_sbar_a_s.T.shape)
for action in range(n_action):
    for state in range(n_state):
        P_sbar_s_a[:, state, action] = T_sbar_a_s[action, state, :]*P_a_s[action, state]
P_sbar_s = np.sum(P_sbar_s_a, axis=2)
print('############################################################################')
print('The Markov dynamics matrix is given by:')
print('############################################################################')
print(P_sbar_s)
print('############################################################################')
#######################################################################
# Observation function calculation
#######################################################################
P_y_s = np.array([
    [0.9, 0.1/3., 0.1/3., 0.9, 0.1/3., 0.9, 0.1/3., 0.1/3., 0.1/3., 0.1/3., 0.1/3., 0.1/3.],
    [0.1/3., 0.9, 0.1/3., 0.1/3., 0.1/3., 0.1/3., 0.9, 0.1/3., 0.1/3., 0.1/3., 0.9, 0.1/3.],
    [0.1/3., 0.1/3., 0.9, 0.1/3., 0.1/3., 0.1/3., 0.1/3., 0.1/3., 0.9, 0.1/3., 0.1/3., 0.9],
    [0.1/3., 0.1/3., 0.1/3., 0.1/3., 0.9, 0.1/3., 0.1/3., 0.9, 0.1/3., 0.9, 0.1/3., 0.1/3.]
])
P_ya_s = np.zeros((n_action*n_obs, n_state))
for obs in range(n_obs):
    for action in range(n_action):
        P_ya_s[obs*n_action+action, :] = P_a_s[action, :]*P_y_s[obs, :]
print('############################################################################')
print('The observation function matrix is given by:')
print('############################################################################')
print(P_ya_s)
print('############################################################################')
#######################################################################
# Sampling next states and observations
#######################################################################
def get_samples(P_sbar_s, P_ya_s, n_sample):
    n_observation = P_ya_s.shape[0]
    n_state = P_ya_s.shape[1]
    # Choosing starting state with equal probability
    start_state = np.random.choice(n_state, size=1)
    # Now we need to sample for n_sample time steps
    curr_state = start_state
    obs_sequence = np.zeros((n_sample,))
    for sample in range(n_sample):
        # Get observation
        obs_sequence[sample] = np.random.choice(n_observation, size=1, p=P_ya_s[:, curr_state].reshape((n_observation,)))
        # Go to next state
        curr_state = np.random.choice(n_state, size=1, p=P_sbar_s[:, curr_state].reshape((n_state,)))
    return obs_sequence

#######################################################################
# Get samples first
#######################################################################
obs = get_samples(P_sbar_s, P_ya_s, n_sample)
#######################################################################
# Forward variable calculation
#######################################################################
def get_foraward_var(a, b, pi_init, obs):
    T = obs.shape[0]
    n_state = a.shape[0]
    alpha = np.zeros((T, n_state))
    # c = np.ones((n_obs,))
    # Initialization
    alpha[0, :] = pi_init*b[int(obs[0]), :]
    alpha[0, :] = alpha[0, :]/np.sum(alpha[0, :])
    # Recursion
    for t in range(T-1):
        for j in range(n_state):
            temp = 0.
            temp = alpha[t, :] @ a[:, j]
            # for i in range(n_state):
            #     temp += alpha[t, i]*a[i, j]
            alpha[t+1, j] = temp * b[int(obs[t+1]), j]
        # Normalization
        # c[t] = 1./np.sum(alpha[t, :])
        # alpha[t, :] = alpha[t, :]*c[t]
        alpha[t+1, :] = alpha[t+1, :]/np.sum(alpha[t+1, :])

    # modify end probability because of normalization
    # temp = 1.
    # for t in range(n_obs):
    #     temp *= c[t]

    return alpha


def get_backward_var(a, b, beta_init, obs):
    T = obs.shape[0]
    n_state = a.shape[0]
    beta = np.zeros((T, n_state))
    beta[-1, :] = beta_init

    for t in range(T-2, -1, -1):
        for i in range(n_state):
            for j in range(n_state):
                beta[t, i] += a[i, j] * b[int(obs[t+1]), j] * beta[t+1, j]
            # Normalization
            beta[t, i] = beta[t, i]/np.sum(beta[t, :])
    return beta

def get_si_var(alpha, beta, a, b, obs):
    T = obs.shape[0]
    n_state = a.shape[0]
    si = np.zeros((T, n_state, n_state))/n_state
    for t in range(T-1):
        for i in range(n_state):
            for j in range(n_state):
                numerator = alpha[t, i] * a[i, j] * b[int(obs[t+1]), j] * beta[t+1, j]
                temp = 0.
                for k in range(n_state):
                    for l in range(n_state):
                        temp += alpha[t, k]*a[k, l] * b[int(obs[t+1]), l] * beta[t+1, l]
                denominator = temp
                si[t, i, j] = numerator/denominator
    return si

def get_gamma_var(si):
    T = si.shape[0]
    n_state = si.shape[1]
    gamma = np.zeros((T, n_state))
    for t in range(T):
        for i in range(n_state):
            gamma[t, i] = np.sum(si[t, i, :])
    return gamma

#######################################################################
# Initialize and run BW
#######################################################################
a = np.ones((n_state, n_state))/n_state
b = np.ones((int(n_obs*n_action), n_state))/n_sample
pi_init = 1/n_state
n_iteration = 200
# BAUM WELCH ALGORITHM
for iteration in range(n_iteration):
    print('############ iteration = ', iteration, ' ################')
    # Forward
    alpha = get_foraward_var(a, b, pi_init, obs)
    # Backward
    beta = get_backward_var(a, b, beta_init=1., obs=obs)
    # Si values
    si = get_si_var(alpha, beta, a, b, obs)
    # gamma values
    gamma = get_gamma_var(si)

    # Update a transition matrix
    for i in range(n_state):
        for j in range(n_state):
            a[i, j] = np.sum(si[0:-1, i, j])/np.sum(gamma[0:-1, i])
    # Update emission matrix
    for j in range(n_state):
        for m in range(int(n_obs*n_action)):
            # curr_sample = obs[t]
            indices = obs == m
            # indices = [idx for idx, val in enumerate(test_sequence) if val == sequence[i]]
            b[m, j] = np.sum(gamma[indices, j])/np.sum(gamma[:, j])

    temp = get_foraward_var(a, b, pi_init, obs)
    if (np.abs(np.linalg.norm(a)-np.linalg.norm(temp)) < 0.001):
        break
print('Learned transition probability matrix:')
print(a)
print('Learned emmission probability matrix:')
print(b)