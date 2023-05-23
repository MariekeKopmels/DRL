# Changes with respect to v0:
# Discount_factor van 0.99 naaar 0.9

from skimage.transform import resize
import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import deque

MAX_CAPACITY = 1000
MIN_REPLAY_SIZE = 500

BATCH_SIZE = 16

LEARNING_RATE = 5e-5 #alpha
# DISCOUNT_FACTOR = 0.99 #gamma
DISCOUNT_FACTOR = 0.9 #gamma

NUM_ACTIONS = 3

INITIAL_EXPLORATION_RATE = 1
# FINAL_EXPLORATION_RATE = 0.01
# DECAY_RATE = int(1e4)

NUMBER_OF_STEPS = 3000 * 11
NUMBER_OF_TESTING_EPOCHS = 10
NUMBER_OF_OBSERVATION_STEPS = 32 * 11 #TODO: Waarop is deze 32 gebaseerd? *11 toch?

NUM_ENVS = 4
TARGET_UPDATE_FREQ=100 #// NUM_ENVS

class CatchEnv():
    ''' Creates an environment where the catch game is simulated.'''
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

class CNN(nn.Module):
    '''Class with a convolutional neural network, with two [CONV > RELU > POOL] layers after which the output is 
    flattend and put through a fully connected layer, a rectified linear unit and another full connected layer. 
    Outputs the q values of the three possible actions that can be taken in a (batch of) state(s).'''
    def __init__(self, env, output_nodes):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            
            nn.Flatten(),
            
            nn.Linear(in_features=2888, out_features=32),
            nn.ReLU(),
            
            nn.Linear(in_features=32, out_features=output_nodes)
        )

    def forward(self, x):
        '''Reshape input and perform forward pass.'''
        x = np.transpose(x, (0, 3, 1, 2))
        return self.net(x)

    
class ExperienceReplay():
    '''A buffer that stores previous trajectories that are later on sampled in minibatches and used 
    to train the local network.'''
    def __init__(self):
        self.buffer = deque(maxlen=MAX_CAPACITY)
        self.capacity = MAX_CAPACITY
        
    def store(self, state, action, reward, next_state, is_finished):
        experience = (state, action, reward, next_state, is_finished)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else: 
            idx = random.randint(0, MAX_CAPACITY-1)
            self.buffer[idx] = experience
        
    def sample(self, batch_size):
        states = []
        actions = []
        rewards = []
        next_states = []
        is_finisheds = []
        
        batch_idx = random.sample(range(0, len(self.buffer)), batch_size) 
        for idx in batch_idx: 
            state, action, reward, next_state, is_finished = self.buffer[idx]
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            is_finisheds.append(is_finished)
        
        tensors = (torch.tensor(np.array(states)).float(), 
                   torch.tensor(np.array(actions)).long(), 
                   torch.tensor(np.array(rewards)).long(), 
                   torch.tensor(np.array(next_states)).float(), 
                   torch.tensor(np.array(is_finisheds)).bool())

        # return the minibatch
        return random.sample(self.buffer, BATCH_SIZE)
        
def action_selection(network, state, exploration_rate):
    
    if np.random.rand() < exploration_rate:
        return random.randint(0,2)
    else:
        
        tensor_state = torch.as_tensor(state, dtype=torch.float32)
        temp = network(tensor_state.unsqueeze(0))
        # print("Temp: ", temp)
        return temp.argmax().item()
    
def get_exploration_rate(step, old_exploration_rate): 
    '''Calculates the decaying exploration rate, at a certain step.'''
    if step < NUMBER_OF_OBSERVATION_STEPS:
        return 1
    else:
        # exploration_rate = np.interp(step, [0, DECAY_RATE], [INITIAL_EXPLORATION_RATE, FINAL_EXPLORATION_RATE])
        if step%1000 == 0:
            print("For step ", step, ", we have an exploration rate of ", old_exploration_rate*0.9999)
        return old_exploration_rate*0.9999
    
def run_initial_observations(env, experience_replay):
    '''Runs a number of epochs to fill the experience replay buffer before starting to train the network.'''
    state = env.reset()
    
    # At every step, execute a random move, and store the experiences in the experience replay buffer.
    for _ in range(MIN_REPLAY_SIZE):
        action = random.randint(0,2)
        next_state, reward, is_finished = env.step(action)
        experience_replay.store(state, action, reward, next_state, is_finished)  
        
        if is_finished:
            state = env.reset()
    
    
def perform_testing(env, network, epoch):
    '''Test the performance of the local network, by running NUMBER_OF_TESTING_EPOCHS epochs and calculating the winrate.'''
    
    total_reward = 0
    
    for run in range(NUMBER_OF_TESTING_EPOCHS):
        state = env.reset()
        is_finished = False
        
        while not is_finished:
            # Let the local network determine the best action to take
            action = action_selection(network, state, 0)
            # print("Testing - Action: ", action)
            next_state, reward, is_finished = env.step(action)
            state = np.squeeze(next_state)
            
        # Store wether or not the network's actions led to a win or lose.
        total_reward = total_reward + reward
        
    state = env.reset()
    
    # Return the average winrate of the performed testing epochs.
    print(f"{epoch} epochs passed, winrate: {total_reward/NUMBER_OF_TESTING_EPOCHS}")
    return total_reward/NUMBER_OF_TESTING_EPOCHS

def perform_learning(experience_replay, local, target, optimizer):
    '''With a minibatch of BATCH_SIZE examples, the local network is trained.
    Using the update rule
    
    target = reward + alpha * (1 - is_finished) * max(target q values) 
    
    where 
    
    is_finished is either 0 the trajectory is not the last step of the epoch
    and 1 if it is the last step of the epoch. 
    
    and max(target q values) is the maximum q value of the next states, as approximated
    by the target network.
    
    Uses a smooth l1 loss to perform backpropagation and optimize the network.
    '''
    # Sample a minibatch from the experience replay buffer.
    minibatch = experience_replay.sample(BATCH_SIZE)
    
    # Store the data from the minibatch
    states = np.asarray([t[0] for t in minibatch])
    actions = np.asarray([t[1] for t in minibatch])
    rewards = np.asarray([t[2] for t in minibatch])
    new_states = np.asarray([t[3] for t in minibatch])
    terminals = np.asarray([t[4] for t in minibatch])

    # Reformat the data, turn numpy arrays into tensors
    states_tensor = torch.as_tensor(states, dtype=torch.float32)
    actions_tensor = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
    rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1)
    new_states_tensor = torch.as_tensor(new_states, dtype=torch.float32)
    terminals_tensor = torch.as_tensor(terminals, dtype=torch.float32).unsqueeze(-1)
    
    # Calculate maximum target q-values of the next state, using the target network
    target_q_values = target(new_states_tensor)
    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

    # Use the update rule to calculate the target values
    targets = rewards_tensor + DISCOUNT_FACTOR * (1 - terminals_tensor) * max_target_q_values

    # Find the q-values of the performed actions, using the local network
    q_values = local(states_tensor)
    action_q_values = torch.gather(input=q_values, dim=1, index=actions_tensor)

    # Train the local network with the q_values of the performed actions and the target values.
    loss = nn.functional.smooth_l1_loss(action_q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

def perform_training(env, experience_replay, local, target, optimizer):
    '''Trains the local network. Uses a target network to improve the estimation of the temporal difference targets.
    After every 10 epochs (so, 110 steps as an epoch consists of 11 steps) the performance of the local network is 
    tested, and the results are recorded.'''
    
    results = []
    state = env.reset()
    exploration_rate = INITIAL_EXPLORATION_RATE

    for step in range(NUMBER_OF_STEPS):
        
        # Based on the exploration rate and local network, the action is chosen
        exploration_rate = get_exploration_rate(step, exploration_rate)
        action = action_selection(local, state, exploration_rate)
        # print("Training - Action: ", action)
        
        # The action is executed, and the next step is made. The results are stored in the experience replay buffer.
        next_state, reward, is_finished = env.step(action)
        experience_replay.store(state, action, reward, next_state, is_finished)
        state = next_state
        
        # After every epoch, the environmet is reset to start a new eopch.
        if is_finished:
            state = env.reset()
            
        # Train the local network
        perform_learning(experience_replay, local, target, optimizer)
        
        # Every TARGET_UPDATE_FREQ steps, the target network is updated.
        if step % TARGET_UPDATE_FREQ == 0:
            target.load_state_dict(local.state_dict())
        
        # Every 110 steps, so 10 epochs, the local network is tested and the performance is stored.
        if step % 110 == 0 and step != 0:
            average_reward = perform_testing(env, local, int(step/11))

            # append the average reward over 10 testing runs
            results.append(average_reward)
            
    return results


def run_environment():
    # Create the catch environment
    env = CatchEnv()

    # Create the networks and experience replay
    local = CNN(env, NUM_ACTIONS)
    target = CNN(env, NUM_ACTIONS)
    target.load_state_dict(local.state_dict())
    
    optimizer = torch.optim.Adam(local.parameters(), LEARNING_RATE)
    # optimizer = torch.optim.RMSprop(local.parameters(), LEARNING_RATE)
    
    experience_replay = ExperienceReplay()
    
    # First, perform observation runs
    run_initial_observations(env, experience_replay)
    
    # Then, train the local network while keeping track of the performance by means of testing every 10 epochs
    results = perform_training(env, experience_replay, local, target, optimizer)
    
    return results


if __name__ == "__main__":
    timestamp = time.time()
    testing_results = run_environment()
    log_number = 9
    np.save(f"group_56_catch_rewards_{log_number}.npy", testing_results)
    print("Done in {:.3f} seconds".format(time.time()-timestamp))