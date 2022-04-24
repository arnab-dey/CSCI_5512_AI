#######################################################################
# IMPORTS
#######################################################################
import numpy as np
#######################################################################
# Constants
#######################################################################
forward = 1 # Just init value. forward=1 means up movement along rows, -1 means down movement
side = 1 # Just init value. side=1 means right movement along columns, -1 means left movement
INVALID_UTILITY = -100. # To denote the greyed out blocks
#######################################################################
# Utility matrix
#######################################################################
U = np.array([[0.551, 0.504, 0.454, 0.187], # (0,0)(bottom-left) to (0,3) of the board in the question
            [0.618, 0.551, 0.487, -1], # (1,0) to (1,3)
            [0.690, INVALID_UTILITY, INVALID_UTILITY, 1], # (2,0) to (2,3)
            [0.747, 0.816, 0.874, 0.931]]) # (3,0) to (3,3)
n_state_row = U.shape[0]
n_state_col = U.shape[1]
end_state = np.array([[1, 3],
                      [2, 3]])
action_state = np.array([[1, 0],
                        [-1, 0],
                        [0, 1],
                        [0, -1]])
#######################################################################
# Function definitions
#######################################################################
#######################################################################
# This function returns the state utility value for an action
#######################################################################
def get_exp_utility_from_action(U, curr_state, action):
    ROW = 0
    COL = 1
    # get forward/backward or left/right movement indicator
    forward = int(action[0])
    side = int(action[1])
    # Form current state
    stay_put_probability = 0.
    stay_put = np.array([curr_state[ROW], curr_state[COL], stay_put_probability])
    # get next state
    next_state_row = curr_state[ROW]+forward
    next_state_col = curr_state[COL]+side
    next_desired = np.array([next_state_row, next_state_col, 0.7])
    # Get spurious transition
    if (0 == side):
        # spurious would be along the col
        next_spurious = np.array([[curr_state[ROW], curr_state[COL]+1, 0.15],
                                  [curr_state[ROW], curr_state[COL]-1, 0.15]])
    else:
        # spurious would be along the rows
        next_spurious = np.array([[curr_state[ROW]+1, curr_state[COL], 0.15],
                                  [curr_state[ROW]-1, curr_state[COL], 0.15]])

    # Check for wall boundary or blocked cell for desired motion
    if ((next_desired[ROW] <0) or (next_desired[COL] < 0) or
            (next_desired[ROW] >= U.shape[0]) or (next_desired[COL] >= U.shape[1]) or
            (INVALID_UTILITY == U[int(next_desired[ROW]), int(next_desired[COL])])):
        next_desired[ROW] = curr_state[ROW]
        next_desired[COL] = curr_state[COL]
        stay_put_probability += next_desired[2]
        next_desired[2] = 0.
    # Check for wall boundary or blocked cell for spurious motion
    for spurious in range(next_spurious.shape[0]):
        if ((next_spurious[spurious, ROW] < 0) or (next_spurious[spurious, COL] < 0) or
                (next_spurious[spurious, ROW] >= U.shape[0]) or (next_spurious[spurious, COL] >= U.shape[1]) or
            (INVALID_UTILITY == U[int(next_spurious[spurious, ROW]), int(next_spurious[spurious, COL])])):
            next_spurious[spurious, ROW] = curr_state[ROW]
            next_spurious[spurious, COL] = curr_state[COL]
            stay_put_probability += next_spurious[spurious, 2]
            next_spurious[spurious, 2] = 0.
    # Update the stay put probability
    stay_put[2] = stay_put_probability
    # Get utility value
    curr_utility = U[int(next_desired[ROW]), int(next_desired[COL])]*next_desired[2]+\
                   U[int(stay_put[ROW]), int(stay_put[COL])]*stay_put[2]+\
                   U[int(next_spurious[0, ROW]), int(next_spurious[0, COL])]*next_spurious[0, 2] +\
                   U[int(next_spurious[1, ROW]), int(next_spurious[1, COL])]*next_spurious[1, 2]
    return curr_utility
#######################################################################
# This function initializes the policy with some random policy
#######################################################################
def initialize_policy(n_state_row, n_state_col):
    policy = np.zeros((n_state_row, n_state_col))
    for row in range(n_state_row):
        for col in range(n_state_col):
            policy[row, col] = np.random.choice(4, size=1)
    return policy
#######################################################################
# This function decodes action number to corresponding action
# up = 0, down = 1, left = 2, right = 3
#######################################################################
def decode_action(action_number):
    action = np.zeros((2,))
    if (0 == int(action_number)):
        action[0] = 1
    elif (1 == int(action_number)):
        action[0] = -1
    elif (2 == int(action_number)):
        action[1] = -1
    elif (3 == int(action_number)):
        action[1] = 1
    return action
#######################################################################
# This function encodes the up, down, left, right movements
# to corresponding codes
# up = 0, down = 1, left = 2, right = 3
#######################################################################
def encode_action(action):
    action_number = 100 # invalid
    if ((1 == action[0]) and (0 == action[1])):
        action_number = 0
    elif ((-1 == action[0]) and (0 == action[1])):
        action_number = 1
    elif ((0 == action[0]) and (-1 == action[1])):
        action_number = 2
    elif ((0 == action[0]) and (1 == action[1])):
        action_number = 3
    return action_number
#######################################################################
# CODE STARTS HERE
#######################################################################
policy = initialize_policy(U.shape[0], U.shape[1])
for row in range(n_state_row):
    for col in range(n_state_col):
        # get current state utility according to the policy
        utility_per_policy = get_exp_utility_from_action(U, [row, col], decode_action(policy[row, col]))
        preferred_action = action_state[0, :]
        utility = 0.
        # get action which maximizes the utility: One step look-ahead
        for action in range(action_state.shape[0]):
            curr_action_utility = get_exp_utility_from_action(U, [row, col], action_state[action, :])
            if (utility < curr_action_utility):
                utility = curr_action_utility
                preferred_action = action_state[action, :]
        # Check if max over all action is greater than the utility based on policy action
        if (utility > utility_per_policy):
            policy[row, col] = encode_action(preferred_action)
#######################################################################
# Print output
#######################################################################
print('action code: 0=up, 1=down, 2=left, 3=right')
print('Optimal policy is')
print(np.flip(policy, axis=0))
print('Please ignore greyed out boxes and +1/-1 boxes as we assume agent cannot start from those states')



