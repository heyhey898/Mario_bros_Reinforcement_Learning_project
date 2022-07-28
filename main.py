
import tensorflow as tf
from tensorflow import keras
import random
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from tqdm import tqdm
import pickle
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
import gym
import numpy as np
import collections
import cv2
import matplotlib.pyplot as plt
from newwrappers import wrapper

from numpy import array, where
from random import choice
# import tensorflow.compat.v1.keras.backend as K
# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()

import time
# import pylab as pl
# from IPython import display



class DQNSolver:

    def __init__(self, input_shape, n_actions, deeper):
        if deeper:
            self.inputs = keras.layers.Input(shape=(84, 84, 4), name="observations")
            self.layer_cnn_1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', strides=(2, 2))(self.inputs)
            self.layer_cnn_2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(self.layer_cnn_1)
            self.layer_cnn_3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', strides=(2, 2))(self.layer_cnn_2)
            self.layer_cnn_4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(self.layer_cnn_3)


            self.layer_flatten = keras.layers.Flatten()(self.layer_cnn_4)
            self.layer_dense1 = keras.layers.Dense(512, activation='relu')(self.layer_flatten)
            self.layer_dense2 = keras.layers.Dense(n_actions, activation="linear")(self.layer_dense1)
            self.model = tf.keras.Model(self.inputs, self.layer_dense2)

        else:
            '''
            swallow layer
            '''
            self.inputs = keras.layers.Input(shape=(84, 84, 4), name="observations")
            self.layer_cnn_1 = tf.keras.layers.Conv2D(32, kernel_size=8, activation='relu', strides=4)(self.inputs)
            self.layer_cnn_2 = tf.keras.layers.Conv2D(64, kernel_size=4, activation='relu',strides=2)(self.layer_cnn_1)
            self.layer_cnn_3 = tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', strides=1)(self.layer_cnn_2)


            self.layer_flatten = keras.layers.Flatten()(self.layer_cnn_3)
            self.layer_dense1 = keras.layers.Dense(512, activation='relu')(self.layer_flatten)
            self.layer_dense2 = keras.layers.Dense(n_actions, activation="linear")(self.layer_dense1)
            self.model = tf.keras.Model(self.inputs, self.layer_dense2)







class DQNAgent:

    def __init__(self, state_space, action_space, max_memory_size, batch_size, gamma, lr,
                 dropout, exploration_max, exploration_min, exploration_decay, double_dqn, deeper):

        # Define DQN Layers
        self.loopnum = 0
        self.state_space = state_space
        self.action_space = action_space
        self.double_dqn = double_dqn


        self.dqn = DQNSolver(state_space, action_space, deeper)
        self.dqn_target = DQNSolver(state_space, action_space, deeper)
        self.copy = 5000  # Copy the local model weights into the target network every 5000 steps
        self.step = 0


        # Create memory
        self.max_memory_size = max_memory_size

        self.STATE_MEM = np.zeros((max_memory_size, *self.state_space))
        self.ACTION_MEM = np.zeros((max_memory_size, 1))
        self.REWARD_MEM = np.zeros((max_memory_size, 1))
        self.NEWSTATE_MEM = np.zeros((max_memory_size, *self.state_space))
        self.DONE_MEM = np.zeros((max_memory_size, 1))
        self.ending_position = 0
        self.num_in_queue = 0

        self.memory_sample_size = batch_size

        # Learning parameters
        self.gamma = gamma
        self.l1 = tf.keras.losses.Huber() # Also known as Huber loss
        self.exploration_max = exploration_max
        self.exploration_rate = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay

    def remember(self, state, action, reward, newstate, done):
        """Store the experiences in a buffer to use later"""

        # state = np.expand_dims(state, 0)
        state = np.array(state)
        newstate = np.array(newstate)


        self.STATE_MEM[self.ending_position] = state
        self.ACTION_MEM[self.ending_position] = action
        self.REWARD_MEM[self.ending_position] = reward
        self.NEWSTATE_MEM[self.ending_position] = newstate
        self.DONE_MEM[self.ending_position] = done
        self.ending_position = (self.ending_position + 1) % self.max_memory_size
        self.num_in_queue = min(self.num_in_queue + 1, self.max_memory_size)

    def batch_experiences(self):
        """Randomly sample"""
        index = random.choices(range(self.num_in_queue), k=self.memory_sample_size)
        STATE = self.STATE_MEM[index]
        ACTION = self.ACTION_MEM[index]
        REWARD = self.REWARD_MEM[index]
        NEWSTATE = self.NEWSTATE_MEM[index]
        DONE = self.DONE_MEM[index]
        return STATE, ACTION, REWARD, NEWSTATE, DONE

    def act(self, state):
        """Epsilon-greedy action"""
        self.step += 1

        state = np.expand_dims(state, 0)
        state = np.array(state)


        # print("state",state.shape)
        if random.random() < self.exploration_rate:

            return random.randrange(self.action_space)

        resultvalue = self.dqn.model(state)
        resultvalue = array(resultvalue)



        return np.argmax(resultvalue, axis=1)[0]




    def copy_model(self):
        print(self.dqn_target.model.trainable_variables[0][1][0][0][0])
        print(self.dqn.model.trainable_variables[0][1][0][0][0])


        self.dqn_target.model.set_weights(self.dqn.model.get_weights())

    def experience_replay(self):
        if self.step % self.copy == 0:
            self.copy_model()
        self.loopnum += 1
        if self.memory_sample_size > self.num_in_queue:
            return

        # Sample a batch of experiences
        STATE, ACTION, REWARD, NEWSTATE, DONE = self.batch_experiences()



        STATE = np.array(STATE)
        # print("STATE.shape",STATE.shape)


        NEWSTATE = np.array(NEWSTATE)
        # print("NEWSTATE.shape",NEWSTATE.shape)



        DONE = DONE[:,0]
        REWARD = REWARD[:,0]
        ACTION = ACTION[:,0]

        NEWSTATE_result = self.dqn_target.model(NEWSTATE)
        if self.double_dqn:
            q = self.dqn.model(NEWSTATE)
            a = np.argmax(q, axis=1)
            NEWSTATE_result = np.array(NEWSTATE_result)

            target = REWARD + (1. - DONE)*(self.gamma * NEWSTATE_result[np.arange(len(NEWSTATE_result)),a])
        else:
            NEWSTATE_result = np.amax(NEWSTATE_result, axis=1)
            target = REWARD + (1. - DONE)*(self.gamma * NEWSTATE_result)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025)

        with tf.GradientTape() as tape:
            current = self.dqn.model(STATE)


            a_true = np.array(ACTION)
            current = tf.gather_nd(params=current,
                                       indices=tf.stack([tf.range(tf.shape(a_true)[0]), a_true], axis=1))




            loss = self.l1(current,target)

        if self.loopnum % 1000 == 0:
            print(current)
            print(target)
            print(self.dqn.model.trainable_variables[0][0][0][0][0])
            print(self.dqn.model.trainable_variables[0][1][0][0][0])

        gradients = tape.gradient(loss, self.dqn.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.dqn.model.trainable_variables))


        self.exploration_rate *= self.exploration_decay

        # Makes sure that exploration rate is always at least 'exploration min'
        self.exploration_rate = max(self.exploration_rate, self.exploration_min)


def show_state(env):
    """show how the agent play mario"""
    env.render()



def run(training_mode, double_dqn,deeper, num_episodes=1000, exploration_max=1.0):
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    # env = create_mario_env(env)  # Wraps the environment so that frames are grayscale

    env = wrapper(env)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    observation_space = (84,84,4)
    action_space = env.action_space.n
    agent = DQNAgent(state_space=observation_space,
                     action_space=action_space,
                     max_memory_size=30000,
                     batch_size=32,
                     gamma=0.90,
                     lr=0.00025,
                     dropout=0.2,
                     exploration_max=exploration_max,
                     exploration_min=0.02,
                     exploration_decay=0.99,
                     double_dqn=double_dqn,
                     deeper=deeper
                     )

    # Restart the enviroment for each episode
    num_episodes = num_episodes
    env.reset()

    total_rewards = []

    save_ep_num = 0
    ep_num_array = []
    total_rewards_mean_array = []
    for ep_num in tqdm(range(num_episodes)):
        # print("ep_num",ep_num)
        state = env.reset()
        # state = state.transpose(1, 2, 0)

        # state = tf.Tensor(state,state.shape, dtype=tf.float32)
        total_reward = 0
        steps = 0
        while True:
            if not training_mode:
                show_state(env)
            if ep_num %50 == 0:
                show_state(env)
            action = agent.act(state)

            steps += 1
            state_next, reward, terminal, info = env.step(action)

            total_reward += reward
            terminal = int(terminal)


            if training_mode:
                agent.remember(state, action, reward, state_next, terminal)
                agent.experience_replay()

            state = state_next
            if terminal:
                break

        total_rewards.append(total_reward)



        if ep_num != 0 and ep_num % 100 == 0:
            print(len(total_rewards))
            print("Episode {} score = {}, average score = {}".format(ep_num + 1, total_rewards[-1],
                                                                     np.mean(total_rewards)))


            ep_num_array.append(ep_num)
            total_rewards_mean_array.append(np.mean(total_rewards))
            reward_file = "total_rewards.npz"
            np.savez(reward_file,ep_num_array, total_rewards_mean_array)



            if training_mode:
                if ep_num % 300 == 0:
                    save_ep_num = ep_num


                print(agent.dqn.model.trainable_variables[0][0][0][0][0])
                print(agent.dqn.model.trainable_variables[0][1][0][0][0])
                agent.dqn.model.save('my_model'+str(save_ep_num)+'.h5')
                agent.dqn.model.save_weights('./checkpoints/my_checkpoint')
                if agent.double_dqn:
                    agent.dqn_target.model.save('my_model_target.h5')

            if training_mode:
                with open("ending_position.pkl", "wb") as f:
                    pickle.dump(agent.ending_position, f)
                with open("num_in_queue.pkl", "wb") as f:
                    pickle.dump(agent.num_in_queue, f)
                with open("total_rewards.pkl", "wb") as f:
                    pickle.dump(total_rewards, f)

                np.savez("STATE_MEM.npz", agent.STATE_MEM)
                np.savez("ACTION_MEM.npz", agent.ACTION_MEM)
                np.savez("REWARD_MEM.npz", agent.REWARD_MEM)

                np.savez("NEWSTATE_MEM.npz", agent.NEWSTATE_MEM)
                np.savez("DONE_MEM.npz", agent.DONE_MEM)

        num_episodes += 1

    print("Episode {} score = {}, average score = {}".format(ep_num + 1, total_rewards[-1], np.mean(total_rewards)))





    env.close()


# For training
run(training_mode=True, double_dqn=True,deeper=True, num_episodes=15000, exploration_max=1)

