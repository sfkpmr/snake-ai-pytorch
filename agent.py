# @author Simon Eklundh
# @author Max Nyström
# @author Marcus Wallén

import random
from collections import deque

import numpy as np
import torch

from game import SnakeGameAI, Direction, Point
from helper import plot
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(17, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # OUR CODE
        to_danger_l_2, to_danger_l_3, border_or_snake_l = game.to_collision(head, Point(-20, 0))
        to_danger_r_2, to_danger_r_3, border_or_snake_r = game.to_collision(head, Point(20, 0))
        to_danger_u_2, to_danger_u_3, border_or_snake_u = game.to_collision(head, Point(0, -20))
        to_danger_d_2, to_danger_d_3, border_or_snake_d = game.to_collision(head, Point(0, 20))
        if dir_l:
            forward2 = to_danger_l_2
            forward3 = to_danger_l_3
            border_or_snake_f = border_or_snake_l
            left2 = to_danger_d_2
            left3 = to_danger_d_3
            border_or_snake_l2 = border_or_snake_d
            right2 = to_danger_u_2
            right3 = to_danger_u_3
            border_or_snake_r2 = border_or_snake_u
        elif dir_r:
            forward2 = to_danger_r_2
            forward3 = to_danger_r_3
            border_or_snake_f = border_or_snake_r
            left2 = to_danger_u_2
            left3 = to_danger_u_3
            border_or_snake_l2 = border_or_snake_u
            right2 = to_danger_d_2
            right3 = to_danger_d_3
            border_or_snake_r2 = border_or_snake_d

        elif dir_u:
            forward2 = to_danger_u_2
            forward3 = to_danger_u_3
            border_or_snake_f = border_or_snake_u
            left2 = to_danger_l_2
            left3 = to_danger_l_3
            border_or_snake_l2 = border_or_snake_l
            right2 = to_danger_r_2
            right3 = to_danger_r_3
            border_or_snake_r2 = border_or_snake_r
        else:
            forward2 = to_danger_d_2
            forward3 = to_danger_d_3
            border_or_snake_f = border_or_snake_d
            left2 = to_danger_r_2
            left3 = to_danger_r_3
            border_or_snake_l2 = border_or_snake_r
            right2 = to_danger_l_2
            right3 = to_danger_l_3
            border_or_snake_r2 = border_or_snake_l

        state = [
            # Move direction not our code
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location not our code
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down

            # forward danger
            forward2,
            forward3,
            border_or_snake_f,
            # left danger
            left2,
            left3,
            border_or_snake_l2,
            # right danger
            right2,
            right3,
            border_or_snake_r2
        ]
        # END OUR CODE
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)
        # get move
        final_move = agent.get_action(state_old)
        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
