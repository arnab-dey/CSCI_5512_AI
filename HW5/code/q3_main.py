####################################################################
# IMPORTS
####################################################################
import gym
from gym import spaces
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.cmd_util import make_vec_env

class GridWorld(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {'render.modes': ['console']}
    # Define constants for clearer code
    FORWARD = 0
    LEFT = 1
    RIGHT = 2
    AGENT_DIR_N = 0  # 0 = north, 1 = east, 2 = south, 3 = west
    AGENT_DIR_E = 1  # 0 = north, 1 = east, 2 = south, 3 = west
    AGENT_DIR_S = 2  # 0 = north, 1 = east, 2 = south, 3 = west
    AGENT_DIR_W = 3  # 0 = north, 1 = east, 2 = south, 3 = west
    AGENT_DIR_MAX = 4
    R = -0.04
    # Wall positions in (row, col) format in the grid. While rendering they
    # will be marked as 'W'
    WALL_1_POS = np.array([2, 1])
    WALL_2_POS = np.array([2, 2])
    # Negative and positive Goal positions in (row, col) format. While rendering they
    # will be marked as '+1'/'-1'
    NEG_GOAL_POS = np.array([1, 3])
    POS_GOAL_POS = np.array([2, 3])

    def __init__(self, grid_size=(4, 4)):
        super(GridWorld, self).__init__()

        # Size of the 2D-grid
        self.grid_size = np.array([grid_size])
        # Initialize the agent at the (0, 0) of the grid
        self.agent_pos = np.zeros((2,))
        self.agent_dir = 0

        # Define action and observation space
        # They must be gym.spaces objects
        # In our case, we have three: forward, left and right
        n_actions = 3
        self.action_space = spaces.Discrete(n_actions)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        # self.observation_space = spaces.Box(low=0, high=3,
        #                                     shape=(2,), dtype='uint8')
        self.observation_space = spaces.Discrete(16)  # Using encoded observation

    ####################################################################
    # This function calcuates the row/col increments based on the
    # agent direction. While taking action agent first changes its
    # direction and then tries to proceed towards that direction
    ####################################################################
    def get_movement(self):
        row_increment = 0
        col_increment = 0
        if (self.AGENT_DIR_N == self.agent_dir):
            row_increment = 1
        elif (self.AGENT_DIR_E == self.agent_dir):
            col_increment = 1
        elif (self.AGENT_DIR_S == self.agent_dir):
            row_increment = -1
        else:
            col_increment = -1
        return np.array([row_increment, col_increment])

    ####################################################################
    # This function returns the encoded observation
    # Agent position is described by (row, col) in the 2D grid.
    # Observation = 4 x row + col
    # Therefore, range of observation is 0-15
    ####################################################################
    def get_obs(self):
        return int(4*self.agent_pos[0]+self.agent_pos[1])

    ####################################################################
    # This function resets the agents to the initial position and
    # initial direction
    ####################################################################
    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # Initialize the agent at (0, 0) of the grid
        self.agent_pos = np.zeros((2,))
        self.agent_dir = 0
        return self.get_obs()

    ####################################################################
    # This function checks if the agents is going out of grid boundary
    # or going to hit a wall
    ####################################################################
    def is_wall_or_boundary(self):
        # Check for boundary
        is_blocked = False
        if ((self.agent_pos <0).any() or (self.agent_pos > 3).any()):
            is_blocked = True
        elif ((np.array_equal(self.agent_pos, self.WALL_1_POS)) or (np.array_equal(self.agent_pos, self.WALL_2_POS))):
            is_blocked = True
        return is_blocked

    ####################################################################
    # This function reverts back the direction of the agent to the
    # previous direction. In case of wall and boundary, we use this
    # function so that the agent stays put at the same place
    # with same direction
    ####################################################################
    def revert_back_to_prev_direction(self, action):
        if action == self.LEFT:
            self.agent_dir = (self.agent_dir + 1) % self.AGENT_DIR_MAX
        elif action == self.RIGHT:
            if (self.agent_dir - 1 < 0):
                self.agent_dir = self.AGENT_DIR_MAX - 1
            else:
                self.agent_dir = (self.agent_dir - 1) % self.AGENT_DIR_MAX
        elif action == self.FORWARD:
            self.agent_dir = self.agent_dir  # No change in direction TODO: Remove later
        else:
          raise ValueError("Received invalid action={} which is not part of the action space".format(action))

    ####################################################################
    # This function is used to take steps based on action
    ####################################################################
    def step(self, action):
        if action == self.LEFT:
            if (self.agent_dir-1 < 0):
                self.agent_dir = self.AGENT_DIR_MAX-1
            else:
                self.agent_dir = (self.agent_dir-1) % self.AGENT_DIR_MAX
        elif action == self.RIGHT:
            self.agent_dir = (self.agent_dir + 1) % self.AGENT_DIR_MAX
        elif action == self.FORWARD:
            self.agent_dir = self.agent_dir  # No change in direction TODO: Remove later
        else:
          raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        # Update agent position
        incr = self.get_movement()
        self.agent_pos[0] += incr[0]
        self.agent_pos[1] += incr[1]

        # Account for the boundaries/wall of the grid
        if (self.is_wall_or_boundary()):
            # Do not change position
            self.agent_pos[0] -= incr[0]
            self.agent_pos[1] -= incr[1]
            self.agent_pos = np.clip(self.agent_pos, 0, 3)
            # Do not change direction
            self.revert_back_to_prev_direction(action)

        done = False
        reward = self.R
        # Have we reached goal?
        if (np.array_equal(self.agent_pos, self.NEG_GOAL_POS)):
            done = True
            reward = -1
        elif (np.array_equal(self.agent_pos, self.POS_GOAL_POS)):
            done = True
            reward = 1

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return self.get_obs(), reward, done, info

    ####################################################################
    # This function renders the grid and agent in console
    # Agent is marked by '^'/'v'/'<'/'>' based on its direction
    # Walls are marked by 'W'
    # Goals are marked by '+1'/'-1'
    # All other positions are marked by '.'
    ####################################################################
    def render(self, mode='console'):
        if mode != 'console':
          raise NotImplementedError()
        for i in range(3, -1, -1):
            for j in range(4):
                curr_pos = np.array([i, j])
                print(" ", end="")
                if (np.array_equal(self.agent_pos, curr_pos)):
                    print(self.AGENT_DIR_TO_STR[self.agent_dir], end="")
                elif (np.array_equal(self.WALL_1_POS, curr_pos) or np.array_equal(self.WALL_2_POS, curr_pos)):
                    print("W", end="")
                elif (np.array_equal(self.NEG_GOAL_POS, curr_pos)):
                    print("-1", end="")
                elif (np.array_equal(self.POS_GOAL_POS, curr_pos)):
                    print("+1", end="")
                else:
                    print(".", end="")
                print(" ", end="")
            print("\n")

    ####################################################################
    # This is used to convert agent direction to string for rendering
    ####################################################################
    AGENT_DIR_TO_STR = {
        0: '^',
        1: '>',
        2: 'v',
        3: '<'
    }
    ####################################################################
    # This is used to convert agent action to string for rendering
    ####################################################################
    AGENT_TO_STR = {
        0: 'FORWARD',
        1: 'LEFT',
        2: 'RIGHT',
        3: 'INVALID'
    }

    def close(self):
        pass


env = GridWorld()
# If the environment don't follow the interface, an error will be thrown
check_env(env, warn=True)

# Instantiate the env
env = GridWorld()
# wrap it. Not required
# env = make_vec_env(lambda: env, n_envs=1)

# Train the agent
model = PPO('MlpPolicy', env, verbose=1, gamma=0.9).learn(total_timesteps=500, n_eval_episodes=100)

# Test the trained agent
obs = env.reset()
n_steps = 40
print('Initial grid world')
env.render(mode='console')
for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    print("Step {}".format(step + 1))
    print("Action: ", env.AGENT_TO_STR[action])
    obs, reward, done, info = env.step(action)
    print('obs=', obs, 'reward=', reward, 'done=', done)
    env.render(mode='console')
    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        print("Goal reached!", "reward=", reward)
        break