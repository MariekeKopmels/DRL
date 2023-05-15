from skimage.transform import resize
import random
import numpy as np
import time
import torch
import torch.nn as nn
import math

# paramerters
MAX_CAPACITY = 500
LEARNING_RATE = 0.01

BATCH_SIZE = 8

LEARNING_RATE = 0.01 #alpha
DISCOUNT_FACTOR = 0.01 #gamma

NUM_ACTIONS = 3

INITIAL_EXPLORATION_RATE = 1
FINAL_EXPLORATION_RATE = 0.01
DECAY_RATE = 0.001

NUMBER_OF_EPOCHS = 2000
NUMBER_OF_TESTING_EPOCHS = 10
NUMBER_OF_OBSERVATION_EPOCHS = 32 #TODO: Waarop is deze 32 gebaseerd?

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
    

class DQN():
    def __init__(self):
        self.network = CNN(NUM_ACTIONS)
        self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr=LEARNING_RATE)
    
    def train(self, minibatch):
        
        Q_st_at = np.zeros(len(minibatch))
        r_t = np.zeros(len(minibatch))
        max_Q_st_1_a = np.zeros(len(minibatch))
        update = np.zeros(len(minibatch))
        
        for idx in range(len(minibatch)):
            example_state = minibatch[0][idx]
            example_action = minibatch[1][idx]
            example_reward = minibatch[2][idx]
            example_next_state = minibatch[3][idx]
            example_is_finished = minibatch[4][idx]
            
            Q_st_at[idx] = self.network.forward(example_state.permute(2, 0, 1)).max().item()
            
            r_t[idx] = example_reward.item()

            max_Q_st_1_a[idx] = self.network.forward(example_next_state.permute(2, 0, 1)).max().item()
            
            # print(" Q_st_at[idx]: ", Q_st_at[idx] )
            # print(" r_t[idx]: ", r_t[idx])
            # print(" max_Q_st_1_a[idx]: ", max_Q_st_1_a[idx])
            # print(" update[idx]: ", update[idx])
            
            update[idx] = Q_st_at[idx] + LEARNING_RATE * (r_t[idx] + (DISCOUNT_FACTOR * max_Q_st_1_a[idx]) - Q_st_at[idx])
            
            
        # print("Update: ", update)
        # print("Q_st_at: ", Q_st_at)
        
        mse_loss = nn.MSELoss()
        loss = mse_loss(torch.tensor(update, requires_grad=True), torch.tensor(Q_st_at, requires_grad=True))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return
    
class ExperienceReplay():
    def __init__(self):
        # self.idx = 0
        self.buffer = []
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
        exploration_rate = FINAL_EXPLORATION_RATE + (INITIAL_EXPLORATION_RATE - FINAL_EXPLORATION_RATE) * math.exp(-DECAY_RATE * epoch)
        # print("For epoch ", epoch, ", we have an exploration rate of ", exploration_rate, " ----------- training phase")
        return exploration_rate
    
    
def perform_testing(env, model):
    total_reward = 0
    
    for run in range(NUMBER_OF_TESTING_EPOCHS):
        env.reset()
        # TODO: Updaten naar 2?
        state, reward, terminal = env.step(1)
        
        while not terminal:
            action = action_selection(model.network, state, 0)
            next_state, reward, terminal = env.step(action)
            state = np.squeeze(next_state)
            
        total_reward = total_reward + reward
        
    return total_reward/NUMBER_OF_TESTING_EPOCHS


def run_environment():
    env = CatchEnv()
    results = []

    model = DQN()
    buffer = ExperienceReplay()

    for ep in range(NUMBER_OF_EPOCHS):
        print("In episode ", ep)
        env.reset()

        # TODO: Updaten naar 2? 
        state, reward, terminal = env.step(1) 

        while not terminal:
            
            # print("Type(state): ", type(state))
            # print("state.shape: ", state.shape)
            # choose and execute action
            action = action_selection(model.network, state, get_exploration_rate(ep))
            next_state, reward, terminal = env.step(action)
            # print("Reward obtained by the agent: {}".format(reward))
            
            # store states in the replay buffer
            if not terminal: 
                buffer.store(state, action, reward, next_state, 0)
            else: 
                buffer.store(state, action, reward, next_state, 1)
            
            # train the model
            if ep > NUMBER_OF_OBSERVATION_EPOCHS:
                model.train(buffer.sample(BATCH_SIZE))
            
            state = np.squeeze(next_state)
                  
        if ep % 10 == 0 and ep != 0:
            average_reward = perform_testing(env, model)
            results.append(average_reward)
            
    return results


if __name__ == "__main__":
    timestamp = time.time()
    testing_results = run_environment()
    log_number = 0
    np.save(f"group_56_catch_rewards_{log_number}.npy", testing_results)
    print("Done in {:.3f} seconds".format(time.time()-timestamp))