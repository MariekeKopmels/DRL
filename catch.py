import itertools
import random
from collections import deque

import numpy as np
import torch
from skimage.transform import resize
from torch import nn

# parameters
GAMMA=0.99
BATCH_SIZE=32
BUFFER_SIZE=int(1e5)
MIN_REPLAY_SIZE=500
EPSILON_START=1.0
EPSILON_END=0.01
EPSILON_DECAY=int(1e4)
NUM_ENVS = 4
TARGET_UPDATE_FREQ=10000 // NUM_ENVS
LR = 5e-5

class CatchEnv():
    '''Class implemented by course'''

    def __init__(self):
        self.size = 21
        self.image = np.zeros((self.size, self.size))
        self.state = []
        self.fps = 4
        self.output_shape = (84, 84)

    def reset_random(self):
        self.image.fill(0)
        self.pos = np.random.randint(2, self.size-2)
        self.vx = np.random.randint(5) - 2
        self.vy = 1
        self.ballx, self.bally = np.random.randint(self.size), 4
        self.image[self.bally, self.ballx] = 1
        self.image[-5, self.pos - 2:self.pos + 3] = np.ones(5)

        return self.step(2)[0]

    def step(self, action):
        def left():
            if self.pos > 3:
                self.pos -= 2
        def right():
            if self.pos < 17:
                self.pos += 2
        def noop():
            pass
        {0: left, 1: right, 2: noop}[action]()
        
        self.image[self.bally, self.ballx] = 0
        self.ballx += self.vx
        self.bally += self.vy
        if self.ballx > self.size - 1:
            self.ballx -= 2 * (self.ballx - (self.size-1))
            self.vx *= -1
        elif self.ballx < 0:
            self.ballx += 2 * (0 - self.ballx)
            self.vx *= -1
        self.image[self.bally, self.ballx] = 1

        self.image[-5].fill(0)
        self.image[-5, self.pos-2:self.pos+3] = np.ones(5)
    
        terminal = self.bally == self.size - 1 - 4
        reward = int(self.pos - 2 <= self.ballx <= self.pos + 2) if terminal else 0

        [self.state.append(resize(self.image, (84, 84))) for _ in range(self.fps - len(self.state) + 1)]
        self.state = self.state[-self.fps:]

        return np.transpose(self.state, [1, 2, 0]), reward, terminal

    def get_num_actions(self):
        return 3

    def reset(self):
        return self.reset_random()

    def state_shape(self):
        return (self.fps,) + self.output_shape

class Network(nn.Module):
    '''A simple neural network architecture.'''

    def __init__(self, env):
        super().__init__()

        input = np.prod(env.state_shape())
        self.net = nn.Sequential(
            nn.Linear(input, 64),
            nn.Tanh(),
            nn.Linear(64, env.get_num_actions())
        )

    def forward(self, x):
        '''Reshape input and implemenet forward pass.'''

        x = np.transpose(x)
        x = x.reshape((-1, x.shape[-1]))
        x = np.transpose(x)
        return self.net(x)

    def act(self, state):
        '''Choose action based on Q-values.'''

        # obtain Q-values
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        q_values = self(state_tensor.unsqueeze(0))

        # generate action from Q-values
        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()

        return action

def run_initial_observations(env, experience_replay):
    '''First observe a few states to fill the replay buffer.'''
    # get initial state
    state = env.reset()

    for _ in range(MIN_REPLAY_SIZE):
        action = random.randint(0,2)

        new_state, reward, terminal = env.step(action)

        transition = (state, action, reward, terminal, new_state)

        experience_replay.append(transition)
        state = new_state

        if terminal:
            state = env.reset()

def perform_testing(env, online_net):
    '''Performs testing as prescribed in the assignment.'''
    average_reward = 0

    # perform 10 testing runs
    for _ in range(10):
        # Get initial state
        state = env.reset()
        terminal = False

        while not terminal:
            # perform testing without exploration
            action = online_net.act(state)

            # execute action and replace state
            next_state, reward, terminal = env.step(action)
            state = np.squeeze(next_state)

        # add final reward to the average
        average_reward += reward

    state = env.reset()

    return average_reward/10

def perform_learning(experience_replay, online_net, target_net, optimizer):

    # randomly sample a few transitions
    transitions = random.sample(experience_replay, BATCH_SIZE)

    # store individual pieces of transitions as numpy array
    states = np.asarray([t[0] for t in transitions])
    actions = np.asarray([t[1] for t in transitions])
    rewards = np.asarray([t[2] for t in transitions])
    terminals = np.asarray([t[3] for t in transitions])
    new_states = np.asarray([t[4] for t in transitions])

    # turn numpy arrays to tensors
    states_tensor = torch.as_tensor(states, dtype=torch.float32)
    actions_tensor = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
    rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1)
    terminals_tensor = torch.as_tensor(terminals, dtype=torch.float32).unsqueeze(-1)
    new_states_tensor = torch.as_tensor(new_states, dtype=torch.float32)

    # calculate maximum target q-value
    target_q_values = target_net(new_states_tensor)
    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

    # update rule
    targets = rewards_tensor + GAMMA * (1 - terminals_tensor) * max_target_q_values

    # find action q-values
    q_values = online_net(states_tensor)
    action_q_values = torch.gather(input=q_values, dim=1, index=actions_tensor)

    # train model
    loss = nn.functional.smooth_l1_loss(action_q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def perform_training(env, experience_replay, online_net, target_net, optimizer):
    '''Train the model.'''
    testing_results = []

    # get initial state
    state = env.reset()

    # 11 steps per episode
    for step in range(1, 100000):

        # execute epsilon greedy policy
        epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

        rnd_sample = random.random()
        if rnd_sample <= epsilon:
            action = random.randint(0,2)
        else: 
            action = online_net.act(state)

        # step and store observations
        new_state, reward, terminal = env.step(action)
        transition = (state, action, reward, terminal, new_state)

        experience_replay.append(transition)
        state = new_state

        # reset field when terminal
        if terminal:
            state = env.reset()

        perform_learning(experience_replay, online_net, target_net, optimizer)

        if step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(online_net.state_dict())

        # Perform testing at every 10 episodes = 110 steps
        if step % 110 == 0 and step != 0:
            average_reward = perform_testing(env, online_net)

            # append the average reward over 10 testing runs
            testing_results.append(average_reward)

    return testing_results
    
def run_environment():

    # initialize environment, buffers, networks and optimizer
    env = CatchEnv()

    experience_replay = deque(maxlen=BUFFER_SIZE)

    online_net = Network(env)
    target_net = Network(env)
    target_net.load_state_dict(online_net.state_dict())

    optimizer = torch.optim.Adam(online_net.parameters(), lr = LR)

    # fill experience replay
    run_initial_observations(env, experience_replay)

    # perform training (and testing)
    testing_results = perform_training(env, experience_replay, online_net, target_net, optimizer)

    return testing_results

if __name__ == '__main__':
    testing_results = run_environment()
    log_number = 0
    np.save(f"group_56_catch_rewards_{log_number}.npy", testing_results)