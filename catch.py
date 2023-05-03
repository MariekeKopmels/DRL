from skimage.transform import resize
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#TODO Adjust values
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.01
EXPLORATION_RATE = 0.1
BATCH_SIZE = 32
NUMBER_OF_EPOCHS = 1000
INPUT_NODES = 1 # image as input
HIDDEN_NODES = 10
OUTPUT_NODES = 3 # three actions: left, right, neither
MAX_CAPACITY = 256

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

class NeuralNetwork(nn.Module):
    '''An simple neural network architecture with one hidden layer. 
    The number of nodes can be specified per layer. The network is fully
    connected and uses the ReLU activation function.'''

    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        super(NeuralNetwork, self).__init__()
        self.input_layer = nn.Linear(input_nodes, hidden_nodes)
        self.hidden_layer = nn.Linear(hidden_nodes, hidden_nodes)
        self.output_layer = nn.Linear(hidden_nodes, output_nodes)

    def forward(self, input_state):
        '''Perform a forward pass. '''

        temp_state = torch.relu(self.input_layer(input_state))
        temp_state = torch.relu(self.hidden_layer(temp_state))
        output = self.output_layer(temp_state)
        return output

class ExperienceReplay():
    '''An experience replay buffer that stores a specified number of
    experiences and is able to add to these experiences or sample a
    specified number of experiences from the buffer. '''

    def __init__(self, max_capacity):
        self.idx = 0
        self.buffer = []
        self.capacity = max_capacity

    def store(self, state, action, reward, next_state, is_finished):
        '''Stores an experience in the buffer. '''

        experience = (state, action, reward, next_state, is_finished)
        if self.capacity >= len(self.buffer):
            self.buffer.append(experience)
        else:
            self.buffer[self.idx] = experience
        
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size):
        '''Samples a specified number of experiences from the buffer. '''

        states = []
        actions = []
        rewards = []
        next_states = []
        is_finisheds = []

        # sample batch and extract their information
        batch_idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        for idx in batch_idx:
            state, action, reward, next_state, is_finished = self.buffer[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            is_finisheds.append(is_finished)

        # create tensors 
        tensors = (torch.tensor(states).float(), torch.tensor(actions).long(), torch.tensor(rewards).long(), torch.tensor(next_states).float(), torch.tensor(is_finisheds).bool())

        return tensors

class DDQN():
    # TODO make DDQN class that performs the training and playing
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        # initialize two identical networks as local and target
        self.local_network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes)
        self.target_network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes)


# TODO
# For each episode, initialize the state and play the game by selecting actions according to the epsilon-greedy policy. 
# Store the experience tuple in the replay buffer and sample a batch of experiences from the buffer. 
# Compute the target Q-values using the target network and update the Q-network using the loss function.
# Update the target network: Every n episodes, update the target network by copying the weights from the Q-network.

def choose_action():
    # TODO: Implement
    return random.randint(0,2)

def run_environment():
    env = CatchEnv()
    number_of_episodes = 1

    for ep in range(number_of_episodes):
        env.reset()
        
        state, reward, terminal = env.step(1) 

        while not terminal:
            state, reward, terminal = env.step(choose_action())
            print("Reward obtained by the agent: {}".format(reward))
            state = np.squeeze(state)

        print("End of the episode")

if __name__ == "__main__":
    run_environment()
    # agent = DDQN(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES)
    # buffer = ExperienceReplay(MAX_CAPACITY)
    
