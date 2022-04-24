####################################################################
# IMPORTS
####################################################################
from utils import argmax, vector_add, print_table  # noqa
from grid import orientations, turn_right, turn_left
from collections import defaultdict
import numpy as np

import random
import matplotlib.pyplot as plt
####################################################################
# VARIABLE DECLARATION
####################################################################
prob_forward = 0.7
prob_left = 0.15
prob_right = 0.15
prob_backward = 0.
R_s_array = [-0.1, -0.08, -0.04, -0.02, -0.001]
gamma = 0.99
n_trial = 5000
is_plot_reqd = False
####################################################################
# CLASS DEFINITIONS
####################################################################
class MDP:

    """A Markov Decision Process, defined by an initial state, transition model,
    and reward function. We also keep track of a gamma value, for use by
    algorithms. The transition model is represented somewhat differently from
    the text.  Instead of P(s' | s, a) being a probability number for each
    state/state/action triplet, we instead have T(s, a) return a
    list of (p, s') pairs.  We also keep track of the possible states,
    terminal states, and actions for each state. [page 646]"""

    def __init__(self, init, actlist, terminals, gamma=.9):
        self.init = init
        self.actlist = actlist
        self.terminals = terminals
        if not (0 <= gamma < 1):
            raise ValueError("An MDP must have 0 <= gamma < 1")
        self.gamma = gamma
        self.states = set()
        self.reward = {}

    def R(self, state):
        "Return a numeric reward for this state."
        return self.reward[state]

    def T(self, state, action):
        """Transition model.  From a state and an action, return a list
        of (probability, result-state) pairs."""
        raise NotImplementedError

    def actions(self, state):
        """Set of actions that can be performed in this state.  By default, a
        fixed list of actions, except for terminal states. Override this
        method if you need to specialize by state."""
        if state in self.terminals:
            return [None]
        else:
            return self.actlist

class GridMDP(MDP):

    """A two-dimensional grid MDP, as in [Figure 17.1].  All you have to do is
    specify the grid as a list of lists of rewards; use None for an obstacle
    (unreachable state).  Also, you should specify the terminal states.
    An action is an (x, y) unit vector; e.g. (1, 0) means move east."""

    def __init__(self, grid, terminals, init=(0, 0), gamma=.9):
        grid.reverse()  # because we want row 0 on bottom, not on top
        MDP.__init__(self, init, actlist=orientations,
                     terminals=terminals, gamma=gamma)
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        for x in range(self.cols):
            for y in range(self.rows):
                self.reward[x, y] = grid[y][x]
                if grid[y][x] is not None:
                    self.states.add((x, y))

    def T(self, state, action):
        if action is None:
            return [(0.0, state)]
        else:
            return [(prob_forward, self.go(state, action)),
                    (prob_right, self.go(state, turn_right(action))),
                    (prob_left, self.go(state, turn_left(action)))]

    def go(self, state, direction):
        """Return the state that results from going in this direction."""
        state1 = vector_add(state, direction)
        return state1 if state1 in self.states else state

    def to_grid(self, mapping):
        """Convert a mapping from (x, y) to v into a [[..., v, ...]] grid."""
        return list(reversed([[mapping.get((x, y), None)
                               for x in range(self.cols)]
                              for y in range(self.rows)]))

    def to_arrows(self, policy):
        chars = {
            (1, 0): '>', (0, 1): '^', (-1, 0): '<', (0, -1): 'v', None: '.'}
        return self.to_grid({s: chars[a] for (s, a) in policy.items()})

# ______________________________________________________________________________

class QLearningAgent:
    """ An exploratory Q-learning agent. It avoids having to learn the transition
        model because the Q-value of a state can be related directly to those of
        its neighbors. [Figure 21.8]
    """
    def __init__(self, mdp, Ne, Rplus, alpha=None):

        self.gamma = mdp.gamma
        self.terminals = mdp.terminals
        self.all_act = mdp.actlist
        self.Ne = Ne  # iteration limit in exploration function
        self.Rplus = Rplus  # large value to assign before iteration limit
        self.Q = defaultdict(float)
        self.Nsa = defaultdict(float)
        self.s = None
        self.a = None
        self.r = None
        self.pi = {s: random.choice(mdp.actions(s)) for s in mdp.states}
        self.mdp = mdp

        if alpha:
            self.alpha = alpha
        else:
            self.alpha = lambda n: 1./(1+n)  # udacity video

    def f(self, u, n):
        """ Exploration function. Returns fixed Rplus untill
        agent has visited state, action a Ne number of times.
        Same as ADP agent in book."""
        if n < self.Ne:
            return self.Rplus
        else:
            return u

    def actions_in_state(self, state):
        """ Returns actions possible in given state.
            Useful for max and argmax. """
        if state in self.terminals:
            return [None]
        else:
            return self.all_act

    def __call__(self, percept):
        s1, r1 = self.update_state(percept)
        Q, Nsa, s, a, r = self.Q, self.Nsa, self.s, self.a, self.r
        alpha, gamma, terminals = self.alpha, self.gamma, self.terminals,
        actions_in_state = self.actions_in_state

        if s1 in terminals:
            Q[s1, None] = r1
        if s is not None:
            Nsa[s, a] += 1
            Q[s, a] += alpha(Nsa[s, a]) * (r + gamma * max(Q[s1, a1]
                                           for a1 in actions_in_state(s1)) - Q[s, a])
        if s in terminals:
            self.s = self.a = self.r = None
        else:
            self.s, self.r = s1, r1
            self.a = argmax(actions_in_state(s1), key=lambda a1: self.f(Q[s1, a1], Nsa[s1, a1]))
        return self.a

    def update_state(self, percept):
        ''' To be overridden in most cases. The default case
        assumes the percept to be of type (state, reward)'''
        return percept

def run_single_trial(agent_program, mdp):
    ''' Execute trial for given agent_program
    and mdp. mdp should be an instance of subclass
    of mdp.MDP '''

    def take_single_action(mdp, s, a):
        '''
        Selects outcome of taking action a
        in state s. Weighted Sampling.
        '''
        x = random.uniform(0, 1)
        cumulative_probability = 0.0
        for probability_state in mdp.T(s, a):
            probability, state = probability_state
            cumulative_probability += probability
            if x < cumulative_probability:
                break
        return state

    current_state = mdp.init
    while True:
        current_reward = mdp.R(current_state)
        percept = (current_state, current_reward)
        next_action = agent_program(percept)
        agent_program.pi[current_state] = next_action
        if next_action is None:
            break
        current_state = take_single_action(mdp, current_state, next_action)
####################################################################
# CODE STARTS HERE
####################################################################
for R_s in R_s_array:
    mdp = GridMDP([[R_s, R_s, R_s, R_s],
                   [R_s, None, None, 1],
                   [R_s, R_s, R_s, -1],
                   [R_s, R_s, R_s, R_s]],
                  terminals=[(3, 2), (3, 1)], gamma=gamma)
    agent = QLearningAgent(mdp=mdp, Ne=10, Rplus=2)
    U = np.zeros((n_trial,))
    st = (0, 0)
    for trial in range(n_trial):
        run_single_trial(agent, mdp)
        agent.s = None
        # Calculate utility value for a state
        U[trial] = max(agent.Q[st, a1] for a1 in agent.actions_in_state(st))
    print('Policy obtained for R = ', R_s)
    print_table(mdp.to_arrows(agent.pi))
#######################################################################
# Plot of error rate
#######################################################################
if (True == is_plot_reqd):
    ###########################################################################
    # Configure axis and grid
    ###########################################################################
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)

    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.5')
    ax.grid(which='minor', linestyle="-.", linewidth='0.5')

    x_axis = np.linspace(1, n_trial, num=int(n_trial))
    ax.plot(x_axis, U, label='utility error')

    ax.set_xlabel(r'number of trials', fontsize=8)
    ax.set_ylabel(r'Utility estimates', fontsize=8)

    plt.legend()
    plt.show()
