from skimage.transform import resize
import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import deque

# paramerters
# MAX_CAPACITY = 500

# BATCH_SIZE = 8

# LEARNING_RATE = 0.1 #alpha
# DISCOUNT_FACTOR = 0.1 #gamma

# NUM_ACTIONS = 3

# INITIAL_EXPLORATION_RATE = 1
# FINAL_EXPLORATION_RATE = 0.01
# DECAY_RATE = 0.001

# NUMBER_OF_EPOCHS = 2000
# NUMBER_OF_TESTING_EPOCHS = 10
# NUMBER_OF_OBSERVATION_EPOCHS = 32 #TODO: Waarop is deze 32 gebaseerd?

MAX_CAPACITY = 100000
MIN_REPLAY_SIZE = 500

BATCH_SIZE = 32

LEARNING_RATE = 0.1 #alpha
DISCOUNT_FACTOR = 0.99 #gamma

NUM_ACTIONS = 3

INITIAL_EXPLORATION_RATE = 1
FINAL_EXPLORATION_RATE = 0.01
DECAY_RATE = 0.001

NUMBER_OF_STEPS = 11000
NUMBER_OF_TESTING_EPOCHS = 10
NUMBER_OF_OBSERVATION_EPOCHS = 32*11 #TODO: Waarop is deze 32 gebaseerd? *11 toch?

class CatchEnv():
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
    def __init__(self, output_nodes):
        super(CNN, self).__init__()
        # self.layer1 = nn.Linear(4, 128)
        # self.layer2 = nn.Linear(128, 128)
        # self.layer3 = nn.Linear(128, 3)
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            
            nn.Flatten(start_dim=0),
            
            nn.Linear(in_features=2888, out_features=32),
            nn.ReLU(),
            
            nn.Linear(in_features=32, out_features=output_nodes)
        )
        
        # # conv > relu > pool
        # self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3) 
        # self.relu1 = nn.ReLU()
        # self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        # # conv > relu > pool
        # self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3) 
        # self.relu2 = nn.ReLU()
        # self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        # # fc > relu
        # self.fc1 = nn.Linear(in_features=2888, out_features=32)
        # self.relu3 = nn.ReLU()
        
        # # fc
        # self.fc2 = nn.Linear(in_features=32, out_features=output_nodes)
        
    def forward(self, input_state):
        # print("Shape: ", input_state.shape)
        # x = F.relu(self.conv1(input_state))
        # x = F.max_pool2d(x, kernel_size=(2,2), stride=(2,2))
        # # print("Shape: ", x.shape)
        # x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x, kernel_size=(2,2), stride=(2,2))
        # # print("Shape: ", x.shape)
        # x = torch.flatten(x)
        # # print("Shape: ", x.shape)
        # x = F.relu(self.fc1(x))
        # # print("Shape: ", x.shape)
        # x = self.fc2(x)
        
        return self.net(input_state)

# class DQN():
#     def __init__(self):
#         self.network = CNN(NUM_ACTIONS)
#         # self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr=LEARNING_RATE)
#         self.optimizer = torch.optim.Adam(self.network.parameters(), lr = LEARNING_RATE)

    
#     def train(self, minibatch):
#         # self.network.train()
        
#         Q_st_at = np.zeros(len(minibatch[0]))
#         r_t = np.zeros(len(minibatch[0]))
#         max_Q_st_1_a = np.zeros(len(minibatch[0]))
#         update = np.zeros(len(minibatch[0]))
        
#         for idx in range(len(minibatch[0])):
#             # print("Idx: ", idx)
#             example_state = minibatch[0][idx]
#             example_action = minibatch[1][idx]
#             example_reward = minibatch[2][idx]
#             example_next_state = minibatch[3][idx]
#             example_is_finished = minibatch[4][idx]
            
#             Q_st_at[idx] = self.network.forward(example_state.permute(2, 0, 1)).max().item()
#             # Q_st_at[idx] = self.network.forward(example_state).max().item()
            
#             r_t[idx] = example_reward.item()

#             max_Q_st_1_a[idx] = self.network.forward(example_next_state.permute(2, 0, 1)).max().item()
#             # max_Q_st_1_a[idx] = self.network.forward(example_next_state).max().item()

#             # print(" Q_st_at[idx]: ", Q_st_at[idx] )
#             # print(" r_t[idx]: ", r_t[idx])
#             # print(" max_Q_st_1_a[idx]: ", max_Q_st_1_a[idx])
#             # print(" update[idx]: ", update[idx])
            
#             update[idx] = Q_st_at[idx] + LEARNING_RATE * (r_t[idx] + (DISCOUNT_FACTOR * max_Q_st_1_a[idx]) - Q_st_at[idx])
            
            
#         print("Update:  ", update)
#         print("Q_st_at: ", Q_st_at)
        
        
        
#         # mse_loss = nn.MSELoss()
#         # loss = mse_loss(torch.tensor(update, requires_grad=True), torch.tensor(Q_st_at, requires_grad=True))
#         loss = nn.functional.smooth_l1_loss(torch.tensor(update, requires_grad=True), torch.tensor(Q_st_at, requires_grad=True))
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
        
#         print("Performed training, now retry")
        
#         Q_st_at = np.zeros(len(minibatch[0]))
#         r_t = np.zeros(len(minibatch[0]))
#         max_Q_st_1_a = np.zeros(len(minibatch[0]))
#         update = np.zeros(len(minibatch[0]))
        
#         for idx in range(len(minibatch[0])):
#             example_state = minibatch[0][idx]
#             example_action = minibatch[1][idx]
#             example_reward = minibatch[2][idx]
#             example_next_state = minibatch[3][idx]
#             example_is_finished = minibatch[4][idx]
            
#             Q_st_at[idx] = self.network.forward(example_state.permute(2, 0, 1)).max().item()
            
#             r_t[idx] = example_reward.item()

#             max_Q_st_1_a[idx] = self.network.forward(example_next_state.permute(2, 0, 1)).max().item()
            
#             # print(" Q_st_at[idx]: ", Q_st_at[idx] )
#             # print(" r_t[idx]: ", r_t[idx])
#             # print(" max_Q_st_1_a[idx]: ", max_Q_st_1_a[idx])
#             # print(" update[idx]: ", update[idx])
            
#             update[idx] = Q_st_at[idx] + LEARNING_RATE * (r_t[idx] + (DISCOUNT_FACTOR * max_Q_st_1_a[idx]) - Q_st_at[idx])
            
            
#         print("Update:  ", update)
#         print("Q_st_at: ", Q_st_at)
        
#         return
    
class ExperienceReplay():
    def __init__(self):
        # self.idx = 0
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
        return tensors
        
def action_selection(network, state, exploration_rate):
    
    if np.random.rand() < exploration_rate:
        return random.randint(0,2)
    else:
        reformatted_state = np.transpose(state, [2, 0, 1])
        temp = network.forward(torch.from_numpy(reformatted_state).float())
        # print("Temp: ", temp)
        return temp.argmax().item()
    
def get_exploration_rate(epoch): 
    if epoch < NUMBER_OF_OBSERVATION_EPOCHS:
        # print("For epoch ", epoch, ", we have an exploration rate of 1 ----------- observation phase")
        return 1
    else:
        exploration_rate = np.interp(epoch, [0, DECAY_RATE], [INITIAL_EXPLORATION_RATE, FINAL_EXPLORATION_RATE])
        # print("For epoch ", epoch, ", we have an exploration rate of ", exploration_rate, " ----------- training phase")
        return exploration_rate
    
def run_initial_observations(env, experience_replay):
    state = env.reset()
    
    for _ in range(MIN_REPLAY_SIZE):
        action = random.randint(0,2)
        
        next_state, reward, is_finished = env.step(action)
        
        experience_replay.store(state, action, reward, next_state, is_finished)  
        
        if is_finished:
            state = env.reset()
    
    
def perform_testing(env, model, epoch):
    total_reward = 0
    
    for run in range(NUMBER_OF_TESTING_EPOCHS):
        state = env.reset()
        is_finished = False
        
        while not is_finished:
            action = action_selection(model, state, 0)
            # print("Testing - Action: ", action)
            next_state, reward, is_finished = env.step(action)
            state = np.squeeze(next_state)
            
        total_reward = total_reward + reward
        
    state = env.reset()
        
    return total_reward/NUMBER_OF_TESTING_EPOCHS

def perform_learning(experience_replay, model, optimizer):
    minibatch = experience_replay.sample(BATCH_SIZE)
    
    Q_st_at = np.zeros(len(minibatch[0]))
    r_t = np.zeros(len(minibatch[0]))
    max_Q_st_1_a = np.zeros(len(minibatch[0]))
    update = np.zeros(len(minibatch[0]))
    
    for idx in range(len(minibatch[0])):
        # print("Idx: ", idx)
        example_state = minibatch[0][idx]
        example_action = minibatch[1][idx]
        example_reward = minibatch[2][idx]
        example_next_state = minibatch[3][idx]
        example_is_finished = minibatch[4][idx]
        
        Q_st_at[idx] = model.forward(example_state.permute(2, 0, 1)).max().item()
        # Q_st_at[idx] = self.network.forward(example_state).max().item()
        
        r_t[idx] = example_reward.item()

        max_Q_st_1_a[idx] = model.forward(example_next_state.permute(2, 0, 1)).max().item()
        # max_Q_st_1_a[idx] = self.network.forward(example_next_state).max().item()

        # print(" Q_st_at[idx]: ", Q_st_at[idx] )
        # print(" r_t[idx]: ", r_t[idx])
        # print(" max_Q_st_1_a[idx]: ", max_Q_st_1_a[idx])
        # print(" update[idx]: ", update[idx])
        
        update[idx] = Q_st_at[idx] + LEARNING_RATE * (r_t[idx] + (DISCOUNT_FACTOR * max_Q_st_1_a[idx]) - Q_st_at[idx])
        
    # print("Update:  ", update)
    # print("Q_st_at: ", Q_st_at)
        
    loss = nn.functional.smooth_l1_loss(torch.tensor(update, requires_grad=True), torch.tensor(Q_st_at, requires_grad=True))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Q_st_at = np.zeros(len(minibatch[0]))
    # r_t = np.zeros(len(minibatch[0]))
    # max_Q_st_1_a = np.zeros(len(minibatch[0]))
    # update = np.zeros(len(minibatch[0]))
    
    # for idx in range(len(minibatch[0])):
    #     # print("Idx: ", idx)
    #     example_state = minibatch[0][idx]
    #     example_action = minibatch[1][idx]
    #     example_reward = minibatch[2][idx]
    #     example_next_state = minibatch[3][idx]
    #     example_is_finished = minibatch[4][idx]
        
    #     Q_st_at[idx] = model.forward(example_state.permute(2, 0, 1)).max().item()
    #     # Q_st_at[idx] = self.network.forward(example_state).max().item()
        
    #     r_t[idx] = example_reward.item()

    #     max_Q_st_1_a[idx] = model.forward(example_next_state.permute(2, 0, 1)).max().item()
    #     # max_Q_st_1_a[idx] = self.network.forward(example_next_state).max().item()

    #     # print(" Q_st_at[idx]: ", Q_st_at[idx] )
    #     # print(" r_t[idx]: ", r_t[idx])
    #     # print(" max_Q_st_1_a[idx]: ", max_Q_st_1_a[idx])
    #     # print(" update[idx]: ", update[idx])
        
    #     update[idx] = Q_st_at[idx] + LEARNING_RATE * (r_t[idx] + (DISCOUNT_FACTOR * max_Q_st_1_a[idx]) - Q_st_at[idx])
        
    # print("Second try-------------------- ")
    # print("Update:  ", update)
    # print("Q_st_at: ", Q_st_at)
        

def perform_training(env, experience_replay, model, optimizer):
    
    results = []
    
    state = env.reset()

    for step in range(NUMBER_OF_STEPS):
        
        action = action_selection(model, state, get_exploration_rate(step))
        # print("Training - Action: ", action)
        next_state, reward, is_finished = env.step(action)
        experience_replay.store(state, action, reward, next_state, is_finished)
        
        state = next_state
        
        if is_finished:
            state = env.reset()
            
        perform_learning(experience_replay, model, optimizer)
        
        if step % 110 == 0 and step != 0:
            print("In epoch ", int(step/11))
            average_reward = perform_testing(env, model, int(step/11))

            # append the average reward over 10 testing runs
            results.append(average_reward)
            
    return results


def run_environment():
    env = CatchEnv()

    model = CNN(NUM_ACTIONS)
    experience_replay = ExperienceReplay()
    
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)

    run_initial_observations(env, experience_replay)
    
    results = perform_training(env, experience_replay, model, optimizer)
    
    return results


if __name__ == "__main__":
    timestamp = time.time()
    testing_results = run_environment()
    log_number = 0
    np.save(f"group_56_catch_rewards_{log_number}.npy", testing_results)
    print("Done in {:.3f} seconds".format(time.time()-timestamp))