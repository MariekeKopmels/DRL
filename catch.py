from skimage.transform import resize
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torchvision import models
from torchsummary import summary


#TODO Adjust values
# LEARNING_RATE = 0.01
# DISCOUNT_FACTOR = 0.01
INITIAL_EXPLORATION_RATE = 0.5
FINAL_EXPLORATION_RATE = 0.0001
BATCH_SIZE = 32
NUMBER_OF_EPOCHS = 3000
NUMBER_OF_OBSERVATION_EPOCHS = 32
UPDATE_TARGET_FREQUENCY = 4
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
        
        # conv > relu > pool
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3) 
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
        self.fc2 = nn.Linear(in_features=32, out_features=OUTPUT_NODES)
        

    def forward(self, input_state):
        '''Perform a forward pass. '''
        
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
        tensors = (torch.tensor(states).float(), torch.tensor(actions).long(), torch.tensor(rewards).long(), torch.tensor(next_states).float(), torch.tensor(is_finisheds).bool())

        return tensors

class DDQN():
    # TODO make DDQN class that performs the training and playing
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        # initialize two identical networks as local and target
        self.local_network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes)
        self.target_network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes)
        
    def update_local_network(self):
        self.local_network = self.target_network



# TODO
# For each episode, initialize the state and play the game by selecting actions according to the epsilon-greedy policy. 
# Store the experience tuple in the replay buffer and sample a batch of experiences from the buffer. 
# Compute the target Q-values using the target network and update the Q-network using the loss function.
# Update the target network: Every n episodes, update the target network by copying the weights from the Q-network.

def choose_action(main_model, target_model, env_state, exploration_rate):
    ''' Either explores or use previouse experiences to decide what is the best action to take'''
    if np.random.rand() < exploration_rate:
        print("Explore")
        return random.randint(0,2)
    else: 
        print("Exploit")
        reformatted_env_state = np.transpose(env_state, [2, 0, 1])
        
        # actions = main_model.forward(torch.randn(4, 84, 84).float()).detach().numpy()[0,0,:]
        actions = main_model.forward(torch.from_numpy(reformatted_env_state).float()).detach().numpy()
        print("Action probabilities: ", actions)

        action = np.argmax(actions)
        print("Chosen action: ", action)

        # print(summary(main_model, (4, 84, 84)))

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
            action = choose_action(model.local_network, model.target_network, state, 0)

            # execute action
            next_state, reward, terminal = env.step(action)
            
        # add final reward to the average
        average_reward += reward

    return average_reward/10

def run_environment():
    env = CatchEnv()
    number_of_episodes = 15
    testing_results = []

    # TODO: checken of input/output size idd klopt zo
    input_size = env.output_shape[0] * env.output_shape[1] #7056 (84*84) stond in het paper
    
    model = DDQN(input_size, HIDDEN_NODES, OUTPUT_NODES)
    buffer = ExperienceReplay(MAX_CAPACITY)
        
    
    for ep in range(1, number_of_episodes + 1):
        env.reset()
        
        # Get initial state and do not move
        state, reward, terminal = env.step(2)

        while not terminal:
            # first, always explore
            if ep > NUMBER_OF_OBSERVATION_EPOCHS:
                exploration_rate = 1
            else:
                exploration_rate = INITIAL_EXPLORATION_RATE
                if exploration_rate > FINAL_EXPLORATION_RATE:
                    exploration_rate = (INITIAL_EXPLORATION_RATE - FINAL_EXPLORATION_RATE)/ep

            # Choose and execute action
            action = choose_action(model.local_network, model.target_network, state, exploration_rate)
            next_state, reward, terminal = env.step(action)
            
            # Store the trajectory in the buffer
            buffer.store(state, action, reward, next_state, 1)

            print("Reward obtained by the agent: {}".format(reward))
            
            # finished observing, start training
            if ep > NUMBER_OF_OBSERVATION_EPOCHS:
                minibatch = buffer.sample(BATCH_SIZE)
                # TODO: Train main model
            
                if ep % UPDATE_TARGET_FREQUENCY == 0:
                    # Update target model
                    # TODO: Even checken of ik target en main niet door elkaar haal
                    model.update_local_network()
                        
            state = np.squeeze(next_state)

        print(f"Exploration: {exploration_rate}")

        print("End of the episode")

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
    
