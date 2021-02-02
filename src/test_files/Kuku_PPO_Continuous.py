# ================================================================
#
#   File name   : BipedalWalker-v3_PPO
#   Author      : PyLessons
#   Created date: 2020-10-18
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/Reinforcement_Learning
#   Description : BipedalWalker-v3 PPO continuous agent
#   TensorFlow  : 2.3.1
#
# ================================================================
import os

import random
import gym
import Box2D
import pylab
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
# from tensorboardX import SummaryWriter
# tf.config.experimental_run_functions_eagerly(True) # used for debuging and development
tf.compat.v1.disable_eager_execution()  # usually using this for fastest performance
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, Adadelta
from tensorflow.keras import backend as K
import copy
from scipy import signal

from threading import Thread, Lock
from multiprocessing import Process, Pipe
import time
from packaging import version

from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv

# tf.debugging.set_log_device_placement(True)
# print(tf.config.experimental.list_physical_devices())

########################################
# check tensorflow version
print("Tensorflow Version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This program requires Tensorflow 2.0 or above"
#######################################

######################################
# avoid CUDNN_STATUS_INTERNAL_ERROR
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
################################################
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

class Actor_Model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        self.state_size = input_shape
        self.action_size = action_space
        self.lr = lr
        self.upper_bound = 1.0
        self.model = self.build_net()

    def build_net(self):
        # Initialise weights
        last_init = tf.random_uniform_initializer(minval=-0.03, maxval=0.03)

        input = Input(shape=self.state_size)
        out = Dense(512, activation="relu", kernel_initializer=last_init)(input)
        out = Dense(256, activation="relu", kernel_initializer=last_init)(out)
        out = Dense(64, activation="relu", kernel_initializer=last_init)(out)
        output = Dense(self.action_size, activation="tanh")(out)

        output = output * self.upper_bound

        model = Model(inputs=input, outputs=output)
        model.compile(loss=self.ppo_loss_continuous,
                      optimizer=tf.keras.optimizers.Adam(self.lr))

        return model

    def ppo_loss_continuous(self, y_true, y_pred):
        advantages = y_true[:, :1]
        actions = y_true[:, 1:1 + self.action_size]
        logp_old_ph = y_true[:, 1 + self.action_size]

        epsilon = 0.2
        logp = self.gaussian_likelihood(actions, y_pred)
        ratio = K.exp(logp - logp_old_ph)

        p1 = ratio * advantages
        p2 = tf.where(advantages > 0, (1.0 + epsilon) * advantages, (1.0 - epsilon) * advantages)

        actor_loss = -K.mean(K.minimum(p1, p2))

        return actor_loss

    def gaussian_likelihood(self, actions, pred):
        log_std = -0.5 * np.ones(self.action_size, dtype=np.float32)
        pre_sum = -0.5 * (((actions - pred) / (K.exp(log_std) + 1e-8)) ** 2 + 2 * log_std + K.log(2 * np.pi))
        return K.sum(pre_sum, axis=1)

    def predict(self, state):
        return self.model.predict(state)


class Critic_Model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        self.state_size = input_shape
        self.action_size = action_space
        self.lr = lr

        self.model = self.build_net()

    def build_net(self):
        # Initialise weights
        last_init = tf.random_uniform_initializer(minval=-0.03, maxval=0.03)

        input = Input(self.state_size)
        old_values = Input(shape=(1,))

        out = Dense(512, activation="relu", kernel_initializer=last_init)(input)
        out = Dense(256, activation="relu", kernel_initializer=last_init)(out)
        out = Dense(64, activation="relu", kernel_initializer=last_init)(out)
        value = Dense(1, activation=None)(out)

        model = Model(inputs=[input, old_values], outputs=value)
        model.compile(loss=[self.critic_PPO2_loss(old_values)],
                      optimizer=tf.keras.optimizers.Adam(self.lr))

        return model

    def critic_PPO2_loss(self, values):
        def loss(y_true, y_pred):
            epsilon = 0.2
            clipped_value_loss = values + K.clip(y_pred - values, -epsilon, epsilon)
            v_loss1 = (y_true - clipped_value_loss) ** 2
            v_loss2 = (y_true - y_pred) ** 2

            # value_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))
            value_loss = K.mean((y_true - y_pred) ** 2)  # standard PPO loss
            return value_loss
        return loss

    def predict(self, state):
        return self.model.predict([state, np.zeros((state.shape[0], 1))])


class PPOAgent:
    # PPO Main Optimization Algorithm
    def __init__(self, env_name, model_name=""):
        # Initialization
        # Environment and PPO parameters
        self.env_name = env_name
        # self.env = gym.make(env_name)
        self.env = KukaDiverseObjectEnv(renders=False,
                                        isDiscrete=False,
                                        maxSteps=20,
                                        removeHeightHack=False)
        self.action_size = self.env.action_space.shape[0]
        self.state_size = self.env.observation_space.shape
        self.EPISODES = 200000  # total episodes to train through all environments
        self.episode = 0  # used to track the episodes total count of episodes played through all thread environments
        self.max_average = 0.5  # when average score is above 0 model will be saved
        self.lr = 0.00025
        self.epochs = 10  # training epochs
        self.shuffle = True
        self.Training_batch = 1024
        # self.optimizer = RMSprop
        self.optimizer = Adam

        self.replay_count = 0
        # self.writer = SummaryWriter(comment="_"+self.env_name+"_"+self.optimizer.__name__+"_"+str(self.lr))

        # Instantiate plot memory
        self.scores_, self.episodes_, self.average_ = [], [], []  # used in matplotlib plots

        # Create Actor-Critic network models
        self.Actor = Actor_Model(input_shape=self.state_size, action_space=self.action_size, lr=self.lr,
                                 optimizer=self.optimizer)
        self.Critic = Critic_Model(input_shape=self.state_size, action_space=self.action_size, lr=self.lr,
                                   optimizer=self.optimizer)

        self.Actor_name = f"{self.env_name}_PPO_Actor.h5"
        self.Critic_name = f"{self.env_name}_PPO_Critic.h5"
        # self.load() # uncomment to continue training from old weights

        # do not change bellow
        self.log_std = -0.5 * np.ones(self.action_size, dtype=np.float32)
        self.std = np.exp(self.log_std)

    def act(self, state):
        # Use the network to predict the next action to take, using the model
        pred = self.Actor.predict(state)

        low, high = -1.0, 1.0  # -1 and 1 are boundaries of tanh
        action = pred + np.random.uniform(low, high, size=pred.shape) * self.std
        action = np.clip(action, low, high)

        logp_t = self.gaussian_likelihood(action, pred, self.log_std)

        return np.squeeze(action)[0][0], logp_t[0][0]

    def gaussian_likelihood(self, action, pred, log_std):
        # https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/sac/policies.py
        pre_sum = -0.5 * (((action - pred) / (np.exp(log_std) + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        return np.sum(pre_sum, axis=1)

    def discount_rewards(self, reward):  # gaes is better
        # Compute the gamma-discounted rewards over an episode
        # We apply the discount and normalize it to avoid big variability of rewards
        gamma = 0.99  # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0, len(reward))):
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r)  # normalizing the result
        discounted_r /= (np.std(discounted_r) + 1e-8)  # divide by standard deviation
        return discounted_r

    def get_gaes(self, rewards, dones, values, next_values, gamma=0.99, lamda=0.90, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    def replay(self, states, actions, rewards, dones, next_states, logp_ts):
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        logp_ts = np.vstack(logp_ts)

        # Get Critic network predictions
        values = self.Critic.predict(states)
        next_values = self.Critic.predict(next_states)

        # Compute discounted rewards and advantages
        # discounted_r = self.discount_rewards(rewards)
        # advantages = np.vstack(discounted_r - values)

        print(len(values))
        print(len(states))
        print(len(rewards))

        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))
        '''
        pylab.plot(adv,'.')
        pylab.plot(target,'-')
        ax=pylab.gca()
        ax.grid(True)
        pylab.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.06)
        pylab.show()
        if str(episode)[-2:] == "00": pylab.savefig(self.env_name+"_"+self.episode+".png")
        '''
        # stack everything to numpy array
        # pack all advantages, predictions and actions to y_true and when they are received
        # in custom loss function we unpack it
        print(np.reshape(advantages, self.state_size).shape)
        print(len(actions))
        print(len(logp_ts))
        y_true = np.hstack([advantages, actions, logp_ts])

        # training Actor and Critic networks
        a_loss = self.Actor.model.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=self.shuffle)
        c_loss = self.Critic.model.fit([states, values], target, epochs=self.epochs, verbose=0, shuffle=self.shuffle)

        # calculate loss parameters (should be done in loss, but couldn't find working way how to do that with disabled eager execution)
        pred = self.Actor.predict(states)
        log_std = -0.5 * np.ones(self.action_size, dtype=np.float32)
        logp = self.gaussian_likelihood(actions, pred, log_std)
        approx_kl = np.mean(logp_ts - logp)
        approx_ent = np.mean(-logp)

        # self.writer.add_scalar('Data/actor_loss_per_replay', np.sum(a_loss.history['loss']), self.replay_count)
        # self.writer.add_scalar('Data/critic_loss_per_replay', np.sum(c_loss.history['loss']), self.replay_count)
        # self.writer.add_scalar('Data/approx_kl_per_replay', approx_kl, self.replay_count)
        # self.writer.add_scalar('Data/approx_ent_per_replay', approx_ent, self.replay_count)
        self.replay_count += 1

    def load(self):
        self.Actor.model.load_weights(self.Actor_name)
        self.Critic.model.load_weights(self.Critic_name)

    def save(self):
        self.Actor.model.save_weights(self.Actor_name)
        self.Critic.model.save_weights(self.Critic_name)

    pylab.figure(figsize=(18, 9))
    pylab.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.06)

    def PlotModel(self, score, episode, save=True):
        self.scores_.append(score)
        self.episodes_.append(episode)
        self.average_.append(sum(self.scores_[-50:]) / len(self.scores_[-50:]))
        if str(episode)[-2:] == "00":  # much faster than episode % 100
            pylab.plot(self.episodes_, self.scores_, 'b')
            pylab.plot(self.episodes_, self.average_, 'r')
            pylab.ylabel('Score', fontsize=18)
            pylab.xlabel('Steps', fontsize=18)
            try:
                pylab.grid(True)
                pylab.savefig(self.env_name + ".png")
            except OSError:
                pass
        # saving best models
        if self.average_[-1] >= self.max_average and episode > 100 and save:
            self.max_average = self.average_[-1]
            self.save()
            SAVING = "SAVING"
            # decreaate learning rate every saved model
            # self.lr *= 0.99
            # K.set_value(self.Actor.Actor.optimizer.learning_rate, self.lr)
            # K.set_value(self.Critic.Critic.optimizer.learning_rate, self.lr)
        else:
            SAVING = ""

        return self.average_[-1], SAVING

    def preprocess_state(self, state):
        state = np.asarray(state, dtype=np.float32) / 255.0  # convert into float array
        state = np.expand_dims(state, 0)
        return state

    def run_batch(self):
        state = self.env.reset()
        # state = np.reshape(state, [1, self.state_size[0]])
        state = self.preprocess_state(state)
        print(state.shape)
        done, score, SAVING = False, 0, ''
        while True:
            # Instantiate or reset games memory
            states, next_states, actions, rewards, dones, logp_ts = [], [], [], [], [], []
            for t in range(50):
                if self.episode % 100 == 0:
                    self.env.render()
                # Actor picks an action
                action, logp_t = self.act(state)
                # Retrieve new state, reward, and whether the state is terminal
                next_state, reward, done, _ = self.env.step(action)
                # Memorize (state, next_states, action, reward, done, logp_ts) for training
                states.append(state)
                # next_states.append(np.reshape(next_state, [1, self.state_size[0]]))
                next_state = self.preprocess_state(next_state)
                next_states.append(next_state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                logp_ts.append(logp_t)
                # Update current state shape
                # state = np.reshape(next_state, [1, self.state_size[0]])
                state = next_state
                score += reward
                if done:
                    self.episode += 1
                    average, SAVING = self.PlotModel(score, self.episode)
                    print("episode: {}/{}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, score,
                                                                                 average, SAVING))
                    # self.writer.add_scalar(f'Workers:{1}/score_per_episode', score, self.episode)
                    # self.writer.add_scalar(f'Workers:{1}/learning_rate', self.lr, self.episode)
                    # self.writer.add_scalar(f'Workers:{1}/average_score',  average, self.episode)

                    state, done, score, SAVING = self.env.reset(), False, 0, ''
                    # state = np.reshape(state, [1, self.state_size[0]])
                    state = self.preprocess_state(state)

            self.replay(states, actions, rewards, dones, next_states, logp_ts)
            if self.episode >= self.EPISODES:
                break

        self.env.close()

    def run_multiprocesses(self, num_worker=4):
        works, parent_conns, child_conns = [], [], []
        for idx in range(num_worker):
            parent_conn, child_conn = Pipe()
            work = Environment(idx, child_conn, self.env_name, self.state_size[0], self.action_size, True)
            work.start()
            works.append(work)
            parent_conns.append(parent_conn)
            child_conns.append(child_conn)

        states = [[] for _ in range(num_worker)]
        next_states = [[] for _ in range(num_worker)]
        actions = [[] for _ in range(num_worker)]
        rewards = [[] for _ in range(num_worker)]
        dones = [[] for _ in range(num_worker)]
        logp_ts = [[] for _ in range(num_worker)]
        score = [0 for _ in range(num_worker)]

        state = [0 for _ in range(num_worker)]
        for worker_id, parent_conn in enumerate(parent_conns):
            state[worker_id] = parent_conn.recv()

        while self.episode < self.EPISODES:
            # get batch of action's and log_pi's
            action, logp_pi = self.act(np.reshape(state, [num_worker, self.state_size[0]]))

            for worker_id, parent_conn in enumerate(parent_conns):
                parent_conn.send(action[worker_id])
                actions[worker_id].append(action[worker_id])
                logp_ts[worker_id].append(logp_pi[worker_id])

            for worker_id, parent_conn in enumerate(parent_conns):
                next_state, reward, done, _ = parent_conn.recv()

                states[worker_id].append(state[worker_id])
                next_states[worker_id].append(next_state)
                rewards[worker_id].append(reward)
                dones[worker_id].append(done)
                state[worker_id] = next_state
                score[worker_id] += reward

                if done:
                    average, SAVING = self.PlotModel(score[worker_id], self.episode)
                    print(
                        "episode: {}/{}, worker: {}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES,
                                                                                           worker_id, score[worker_id],
                                                                                           average, SAVING))
                    # self.writer.add_scalar(f'Workers:{num_worker}/score_per_episode', score[worker_id], self.episode)
                    # self.writer.add_scalar(f'Workers:{num_worker}/learning_rate', self.lr, self.episode)
                    # self.writer.add_scalar(f'Workers:{num_worker}/average_score',  average, self.episode)
                    score[worker_id] = 0
                    if (self.episode < self.EPISODES):
                        self.episode += 1

            for worker_id in range(num_worker):
                if len(states[worker_id]) >= self.Training_batch:
                    self.replay(states[worker_id], actions[worker_id], rewards[worker_id], dones[worker_id],
                                next_states[worker_id], logp_ts[worker_id])

                    states[worker_id] = []
                    next_states[worker_id] = []
                    actions[worker_id] = []
                    rewards[worker_id] = []
                    dones[worker_id] = []
                    logp_ts[worker_id] = []

        # terminating processes after a while loop
        works.append(work)
        for work in works:
            work.terminate()
            print('TERMINATED:', work)
            work.join()

    def test(self, test_episodes=100):  # evaluate
        self.load()
        for e in range(101):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size[0]])
            done = False
            score = 0
            while not done:
                self.env.render()
                action = self.Actor.predict(state)[0]
                state, reward, done, _ = self.env.step(action)
                state = np.reshape(state, [1, self.state_size[0]])
                score += reward
                if done:
                    average, SAVING = self.PlotModel(score, e, save=False)
                    print("episode: {}/{}, score: {}, average{}".format(e, test_episodes, score, average))
                    break
        self.env.close()


if __name__ == "__main__":
    # newest gym fixed bugs in 'BipedalWalker-v2' and now it's called 'BipedalWalker-v3'
    env_name = 'BipedalWalker-v3'
    # env_name = 'Pendulum-v0'
    # env_name = 'LunarLanderContinuous-v2'
    # env_name = 'MountainCarContinuous-v0'
    agent = PPOAgent(env_name)
    agent.run_batch()  # train as PPO
    agent.test()