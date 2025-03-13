import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

display_size = 400
grid_size = 20
pygame.init()
screen = pygame.display.set_mode((display_size, display_size))
pygame.display.set_caption("Snake Game")
clock = pygame.time.Clock()

def generate_food(snake):
    while True:
        food = [random.randint(0, (display_size // grid_size) - 1) * grid_size,
                random.randint(0, (display_size // grid_size) - 1) * grid_size]
        if food not in snake:
            return food

# Neural Network for the Agent
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

# Reinforcement Learning Agent
class Agent:
    def __init__(self):
        self.model = DQN(4, 4)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
    
    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, 3)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            actions = self.model(state)
        return torch.argmax(actions).item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=128):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        
        target = rewards + (1 - dones) * self.gamma * torch.max(self.model(next_states), dim=1)[0]
        target_f = self.model(states).gather(1, actions).squeeze()
        
        self.optimizer.zero_grad()
        loss = self.criterion(target_f, target.detach())
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Snake Game Environment
class SnakeGame:
    def __init__(self):
        self.snake = [[100, 100]]
        self.food = generate_food(self.snake)
        self.direction = [grid_size, 0]
        self.done = False
        self.score = 0
    
    def step(self, action):
        if action == 0:
            self.direction = [-grid_size, 0]
        elif action == 1:
            self.direction = [grid_size, 0]
        elif action == 2:
            self.direction = [0, -grid_size]
        elif action == 3:
            self.direction = [0, grid_size]
        
        new_head = [self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1]]
        
        if (new_head in self.snake or
                new_head[0] < 0 or new_head[0] >= display_size or
                new_head[1] < 0 or new_head[1] >= display_size):
            self.done = True
            return self.get_state(), -10, self.done
        
        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.food = generate_food(self.snake)
            self.score += 1
            reward = 10
        else:
            self.snake.pop()
            reward = -0.1
        
        return self.get_state(), reward, self.done
    
    def get_state(self):
        return np.array([self.snake[0][0], self.snake[0][1], self.food[0], self.food[1]])
    
    def reset(self):
        self.__init__()
        return self.get_state()
    
    def render(self):
        screen.fill((0, 0, 0))
        for segment in self.snake:
            pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(segment[0], segment[1], grid_size, grid_size))
        pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(self.food[0], self.food[1], grid_size, grid_size))
        pygame.display.flip()
        clock.tick(60)

# Training Loop
if __name__ == "__main__":
    env = SnakeGame()
    agent = Agent()
    episodes = 1000
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while not env.done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            env.render()
            
        agent.replay()
        print(f"Episode {episode+1}: Score {env.score}, Total Reward {total_reward}, Epsilon {agent.epsilon:.4f}")
    
    print("Training complete!")
