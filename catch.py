from skimage.transform import resize
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torchvision import models
from torchsummary import summary
import math
from collections import namedtuple, deque

#parameters
LEARNING_RATE = 0.01 #alpha
DISCOUNT_FACTOR = 0.01 #gamma
INITIAL_EXPLORATION_RATE = 1
DECAY_RATE = 0.001
FINAL_EXPLORATION_RATE = 0.001
BATCH_SIZE = 1
NUMBER_OF_EPOCHS = 100 
NUMBER_OF_OBSERVATION_EPOCHS = 32
UPDATE_TARGET_FREQUENCY = 4
OUTPUT_NODES = 3 # three actions: left, right, neither
MAX_CAPACITY = 500

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

    def __init__(self, input_nodes, output_nodes):
        super(NeuralNetwork, self).__init__()
                
        # conv > relu > pool
        self.conv1 = nn.Conv2d(in_channels=input_nodes, out_channels=16, kernel_size=3) 
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        # conv > relu > pool
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3) 
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        # fc > relu
        self.fc1 = nn.Linear(in_features=2888, out_features=32)
        self.relu3 = nn.ReLU()
        
        # fc
        self.fc2 = nn.Linear(in_features=32, out_features=output_nodes)
        

    def forward(self, input_state):
        '''Perform a forward pass. '''
        
        # print("--------Going to perform forward pass!--------")
        # print("Input_state.shape: ", input_state.shape)
        
        temp_state = self.conv1(input_state)
        temp_state = self.relu1(temp_state)
        temp_state = self.maxpool1(temp_state)
        
        # print("Temp_state.shape: ", temp_state.shape)
        
        temp_state = self.conv2(temp_state)
        temp_state = self.relu2(temp_state)
        temp_state = self.maxpool2(temp_state)
        
        # print("Temp_state.shape: ", temp_state.shape)
        
        temp_state = torch.flatten(temp_state)
        
        # print("Temp_state.shape: ", temp_state.shape)
        
        temp_state = self.fc1(temp_state)
        temp_state = self.relu3(temp_state)
        
        # print("Temp_state.shape: ", temp_state.shape)
        
        output = self.fc2(temp_state)
        
        # print("Output.shape: ", output.shape)
        
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
        tensors = (torch.tensor(np.array(states)).float(), torch.tensor(np.array(actions)).long(), torch.tensor(np.array(rewards)).long(), torch.tensor(np.array(next_states)).float(), torch.tensor(np.array(is_finisheds)).bool())
        return tensors

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def store(self, state, action, reward, next_state):
        """Save a transition"""
        state = torch.tensor(np.array(state)).float()
        action = torch.tensor(np.array(action)).long()
        reward = torch.tensor(np.array(reward)).long()
        next_state = torch.tensor(np.array(next_state)).float()

        self.memory.append(Transition(state, action, reward, next_state))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DDQN():
    def __init__(self, input_nodes, output_nodes):
        # initialize two identical networks as local and target as well as the optimizer
        self.local_network = NeuralNetwork(input_nodes, output_nodes)
        self.target_network = NeuralNetwork(input_nodes, output_nodes)
        self.optimizer = torch.optim.RMSprop(self.local_network.parameters(), lr=LEARNING_RATE)
                
    def update_target_network(self):
        self.target_network.load_state_dict(self.local_network.state_dict())
        
    def optimize_model(self, batch):
        # batch.next_state = torch.tensor(np.array(batch.next_state)).float()
        # print(batch.action)
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state, dim=2).permute(2, 0, 1)
        # state_batch = batch.state
        print(batch.state[0].shape)
        # action_batch = torch.cat(batch.action)
        action_batch = batch.action
        # reward_batch = torch.cat(batch.reward)
        reward_batch = batch.reward

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.local_network(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * DISCOUNT_FACTOR) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.local_network.parameters(), 100)
        self.optimizer.step()

    def train(self, minibatch):        
        # Calculate update rule
        
        argmax_Q1_st_1_a = np.zeros(BATCH_SIZE, dtype=int)
        local_Q = np.zeros(BATCH_SIZE)
        expected_Q = np.zeros(BATCH_SIZE)
        
        for idx in range(BATCH_SIZE):
            
            Q1_st_1_a = self.local_network.forward(minibatch[3][idx].permute(2, 0, 1)).float().detach().numpy()
            argmax_Q1_st_1_a[idx] = np.argmax(Q1_st_1_a)
            
            a_star = argmax_Q1_st_1_a[idx]
            temp = self.target_network.forward(minibatch[3][idx].permute(2, 0, 1)).float().detach().numpy()
            Q2_st_1_a_star = temp[a_star]
            
            a_t = minibatch[1][idx]
            temp = self.local_network.forward(minibatch[0][idx].permute(2, 0, 1)).float().detach().numpy()
            Q1_st_at = temp[a_t]
            
            r_t = minibatch[2][idx].numpy()
            
            # Compute update rule
            local_Q[idx] = Q1_st_at + LEARNING_RATE * (r_t + DISCOUNT_FACTOR * Q2_st_1_a_star - Q1_st_at)
                        
            # Compute 
            expected_Q[idx] = np.max(self.local_network.forward(minibatch[0][idx].permute(2, 0, 1)).float().detach().numpy())
        
        # Train the local model
        mse_loss = nn.MSELoss()
        loss = mse_loss(torch.tensor(expected_Q, requires_grad=True), torch.tensor(local_Q, requires_grad=True))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def choose_action(local_network, env_state, exploration_rate):
    ''' Either explores or use previouse experiences to decide what is the best action to take'''
    if np.random.rand() < exploration_rate:
        return random.randint(0,2)
    else: 
        reformatted_env_state = np.transpose(env_state, [2, 0, 1])
        actions = local_network.forward(torch.from_numpy(reformatted_env_state).float()).detach().numpy()
        action = np.argmax(actions)

        return action

def perform_testing(env, model):
    average_reward = 0

    # perform 10 testing runs
    for _ in range(10):
        # Get initial state and do not move
        env.reset()
        state, reward, terminal = env.step(2)

        while not terminal:
            # set exploration rate to zero to disable exploration during testing
            action = choose_action(model.local_network, state, 0)

            # execute action
            next_state, reward, terminal = env.step(action)

            state = np.squeeze(next_state)

        # add final reward to the average
        average_reward += reward

    return average_reward/10

def run_environment():
    env = CatchEnv()
    testing_results = []

    print("run")

    
    model = DDQN(4, OUTPUT_NODES)
    # buffer = ExperienceReplay(MAX_CAPACITY)
    buffer = ReplayMemory(MAX_CAPACITY)
        
    
    for ep in range(1, NUMBER_OF_EPOCHS + 1):
        print("In episode ", ep)
        
        env.reset()
        
        # Get initial state and do not move
        state, reward, terminal = env.step(2)

        while not terminal:
            # first, always explore
            if ep < NUMBER_OF_OBSERVATION_EPOCHS:
                exploration_rate = 1
            else:
                exploration_rate = INITIAL_EXPLORATION_RATE
                if exploration_rate > FINAL_EXPLORATION_RATE:
                    # exponential decay
                    # exploration_rate = INITIAL_EXPLORATION_RATE - (((INITIAL_EXPLORATION_RATE - FINAL_EXPLORATION_RATE) / NUMBER_OF_EPOCHS ) * ep)
                    exploration_rate = FINAL_EXPLORATION_RATE + (INITIAL_EXPLORATION_RATE - FINAL_EXPLORATION_RATE) * math.exp(-DECAY_RATE * ep)

            # print("Exploration rate: ", exploration_rate)

            # Choose and execute action
            action = choose_action(model.local_network, state, exploration_rate)
            next_state, reward, terminal = env.step(action)
            
            # Store the trajectory in the buffer
            if terminal:
                # next_state = None
                # buffer.store(state, action, reward, next_state, 1)
                buffer.store(state, action, reward, next_state)
            else:
                # next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                # buffer.store(state, action, reward, next_state, 0)
                buffer.store(state, action, reward, next_state)
                            
            # finished observing, start training
            if ep > NUMBER_OF_OBSERVATION_EPOCHS:
                minibatch = buffer.sample(BATCH_SIZE)
                # Train target model
                # model.train(minibatch)

                # transitions = memory.sample(BATCH_SIZE)
                # # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
                # # detailed explanation). This converts batch-array of Transitions
                # # to Transition of batch-arrays.
                batch = Transition(*zip(*minibatch))

                model.optimize_model(batch)
            
                if ep % UPDATE_TARGET_FREQUENCY == 0:
                    # Update target model
                    model.update_target_network()
            
            state = np.squeeze(next_state)

        # Perform testing at every 10 episodes
        if ep % 10 == 0 and ep != 0:
            average_reward = perform_testing(env, model)

            # append the average reward over 10 testing runs
            testing_results.append(average_reward)

    return testing_results

if __name__ == "__main__":
    timestamp = time.time()
    testing_results = run_environment()
    log_number = 0
    np.save(f"group_56_catch_rewards_{log_number}.npy", testing_results)
    print("Done in {:.3f} seconds".format(time.time()-timestamp))