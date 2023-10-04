# pip install gymnasium[classic-control]
import numpy as np
import gymnasium as gym
import tensorflow as tf
from tensorflow import keras
from collections import deque

# Parameters
GAMMA = 0.99
LEARNING_RATE = 0.001
MEMORY_SIZE = 1000000
BATCH_SIZE = 64
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

class DQNAgent:
    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(keras.layers.Dense(24, activation="relu"))
        self.model.add(keras.layers.Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = np.random.choice(len(self.memory), BATCH_SIZE, replace=False)
        for i in batch:
            state, action, reward, next_state, done = self.memory[i]
            q_update = reward
            if not done:
                q_update = (reward + GAMMA * np.amax(self.model.predict(next_state)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="human")
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_agent = DQNAgent(observation_space, action_space)
    run = 0

    for _ in range(100):
        run += 1
        state, _ = env.reset()
        print(state)
        state = np.array(state)
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            step += 1
            env.render()
            action = dqn_agent.act(state)
            next_state, reward, done, info, _ = env.step(action)
            reward = reward if not done else -reward
            next_state = np.reshape(next_state, [1, observation_space])
            dqn_agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("Run: {}, exploration: {:.2f}, score: {}".format(run, dqn_agent.exploration_rate, step))
                break
            dqn_agent.experience_replay()
    env.close()