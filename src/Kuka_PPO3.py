"""
Implementing Proximal Policy Optimization (PPO) for Kuka Environment
PPO_CLIP Algorithm
"""
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from FeatureNet import FeatureNetwork
from Replay_Buffer import Buffer
from collections import deque

import pickle

###########################
## TENSORFLOW Related Logistics
################################
# check tensorflow version
from packaging import version

print("Tensorflow Version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This program requires Tensorflow 2.0 or above"

# avoid CUDNN_STATUS_INTERNAL_ERROR
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


####################################

################
## ACTOR NETWORK
################
class PPOActor:
    def __init__(self, state_size, action_size,
                 learning_rate, epsilon,
                 action_upper_bound, feature_model):
        self.state_size = state_size  # shape: (w, h, c)
        self.action_size = action_size  # shape: (n, )
        self.lr = learning_rate
        self.epsilon = epsilon  # clipping factor
        self.upper_bound = action_upper_bound
        self.train_step_count = 0

        # create NN models
        self.feature_model = feature_model
        self.model = self._build_net(trainable=True)
        self.model_old = self._build_net(trainable=False)
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def _build_net(self, trainable=True):
        # input is a stack of 1-channel YUV images
        last_init = tf.random_uniform_initializer(minval=-0.03, maxval=0.03)
        state_input = tf.keras.layers.Input(shape=self.state_size)
        f = self.feature_model(state_input)
        f = tf.keras.layers.Dense(128, activation='relu', trainable=trainable)(f)
        f = tf.keras.layers.Dense(64, activation="relu", trainable=trainable)(f)
        net_out = tf.keras.layers.Dense(self.action_size[0], activation='tanh',
                                        kernel_initializer=last_init, trainable=trainable)(f)
        model = tf.keras.Model(state_input, net_out)
        net_out = net_out * self.upper_bound  # element-wise product
        model.summary()
        tf.keras.utils.plot_model(model, to_file='actor_net.png',
                                  show_shapes=True, show_layer_names=True)
        return model

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)

    def train(self, state_batch, action_old, advantages):
        self.train_step_count += 1
        with tf.GradientTape() as tape:
            actor_weights = self.model.trainable_variables
            actor_policy = self.model(state_batch)
            # actor_policy_old = self.model_old(state_batch)
            ratio = (actor_policy + 1e-9) / (action_old + 1e-9)
            p1 = ratio * advantages
            p2 = tf.clip_by_value(ratio, clip_value_min=1 - self.epsilon,
                                  clip_value_max=1 + self.epsilon) * advantages
            actor_loss = -tf.math.reduce_mean(tf.minimum(p1, p2))
        actor_grad = tape.gradient(actor_loss, actor_weights)

        # save the weights before update
        self.model_old.set_weights(self.model.get_weights())

        self.optimizer.apply_gradients(zip(actor_grad, actor_weights))
        return actor_loss

    def update_target(self):
        # self.model_old.set_weights(self.model.get_weights())
        pass


####################
# CRITIC NETWORK
##################
class PPOCritic:
    def __init__(self, state_size, action_size,
                 learning_rate,
                 gamma, feature_model):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(self.lr)
        self.feature_model = feature_model
        self.model = self._build_net(trainable=True)
        self.model_old = self._build_net(trainable=False)
        self.gamma = gamma
        self.train_step_count = 0

    def _build_net(self, trainable=True):
        # state input is a stack of 1-D YUV images
        state_input = tf.keras.layers.Input(shape=self.state_size)
        feature = self.feature_model(state_input)
        out = tf.keras.layers.Dense(128, activation="relu", trainable=trainable)(feature)
        out = tf.keras.layers.Dense(64, activation="relu", trainable=trainable)(out)
        out = tf.keras.layers.Dense(32, activation="relu", trainable=trainable)(out)
        net_out = tf.keras.layers.Dense(1, trainable=trainable)(out)

        # Outputs single value for a given state = V(s)
        model = tf.keras.Model(inputs=state_input, outputs=net_out)
        model.summary()
        tf.keras.utils.plot_model(model, to_file='critic_net.png',
                                  show_shapes=True, show_layer_names=True)
        return model

    def train(self, state_batch, returns):
        self.train_step_count += 1
        with tf.GradientTape() as tape:
            critic_weights = self.model.trainable_variables
            critic_value = self.model(state_batch)
            critic_loss = tf.math.reduce_mean(tf.square(returns - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_weights)
        self.optimizer.apply_gradients(zip(critic_grad, critic_weights))
        return critic_loss

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)


########################
## PPO AGENT
#########################
class KukaPPOAgent:
    def __init__(self, state_size, action_size,
                 lr_a, lr_c, batch_size, memory_capacity, lmbda,
                 epsilon, gamma, action_upper_bound,
                 update_freq=500, trg_epochs=10):
        self.state_size = state_size
        self.action_size = action_size
        self.actor_lr = lr_a
        self.critic_lr = lr_c
        self.batch_size = batch_size
        self.memory_capacity = memory_capacity
        self.gamma = gamma  # discount factor
        self.lmbda = lmbda  # required for estimating advantage (GAE)
        self.epsilon = epsilon  # clip_factor
        self.action_upper_bound = action_upper_bound

        self.start_episode = 0
        self.reward_list = []
        self.s = -1
        self.season_score_window = deque(maxlen=100)
        self.best_mean_reward = -np.inf

        self.training_step_count = 0
        self.update_rate = 1000  # update_freq
        self.max_epochs = trg_epochs

        self.feature = FeatureNetwork(self.state_size)
        self.buffer = Buffer(self.memory_capacity, self.batch_size)
        self.actor = PPOActor(self.state_size, self.action_size, self.actor_lr,
                              self.epsilon, self.action_upper_bound, self.feature)

        # critic estimates the advantage
        self.critic = PPOCritic(self.state_size, self.action_size,
                                self.critic_lr, self.gamma, self.feature)

    def policy(self, state):
        action = tf.squeeze(self.actor.model(state))
        action = action.numpy()
        return action

    def record(self, experience: tuple):
        self.buffer.record(experience)

    # Chose a better name or call it 'train'
    def experience_replay(self):
        actor_loss, critic_loss = 0, 0
        self.training_step_count += 1

        if self.training_step_count % self.update_rate == 0:
            self.s += 1
            print('Updating the Networks ...')
            s_batch, a_batch, r_batch, ns_batch, d_batch = self.buffer.get_all_samples()

            # number of splits
            n_splits = self.buffer.size // self.batch_size

            s_batch = tf.convert_to_tensor(s_batch, dtype=tf.float32)
            a_batch = tf.convert_to_tensor(a_batch, dtype=tf.float32)
            r_batch = tf.convert_to_tensor(r_batch, dtype=tf.float32)
            ns_batch = tf.convert_to_tensor(ns_batch, dtype=tf.float32)
            d_batch = tf.convert_to_tensor(d_batch, dtype=tf.float32)

            # Scoring for last 'update_freq' steps (season)
            season_score = np.sum(r_batch)
            self.season_score_window.append(season_score)
            mean_reward = np.mean(self.season_score_window)
            if mean_reward > self.best_mean_reward:
                print("New best mean updated: {:.3f} --> {:.3f}".format(self.best_mean_reward, mean_reward))
                self.best_mean_reward = mean_reward

            returns_batch, adv_batch = self.compute_advantages(r_batch, s_batch,
                                                               ns_batch, d_batch)

            n_sample = self.buffer.size // self.batch_size
            idx = np.arange(self.buffer.size)
            np.random.shuffle(idx)
            for epoch in range(self.max_epochs):
                for b in range(n_sample):
                    ind = idx[b * self.batch_size:(b + 1) * self.batch_size]
                    adv = np.asarray(adv_batch)[ind]
                    tv = np.asarray(returns_batch)[ind]
                    actions = np.asarray(a_batch)[ind]
                    states = np.asarray(s_batch)[ind]

                    # action_est = self.actor.model(np.asarray(s_batch)[ind])
                    # values = self.critic.model(np.asarray(s_batch)[ind])

                    actor_loss = self.actor.train(states, actions, adv)
                    critic_loss = self.critic.train(states, tv)

            # clear the buffer
            self.buffer.buffer.clear()
            self.buffer.size = 0
            self.epsilon *= 0.999  # Reduce clipping as time goes on
            print("Season: {}, Reward: {}, Mean Reward: {:.3f}, Best Mean Reward: {:.3f}".format(self.s,
                                                                                                 season_score,
                                                                                                 mean_reward,
                                                                                                 self.best_mean_reward))
        return actor_loss, critic_loss

    def compute_advantages(self, r_batch, s_batch, ns_batch, d_batch):
        s_values = self.critic.model(s_batch)  # input: tensor
        ns_values = self.critic.model(ns_batch)
        returns = []
        gae = 0  # generalized advantage estimate
        for i in reversed(range(len(r_batch))):
            delta = r_batch[i] + self.gamma * ns_values[i] * (1 - d_batch[i]) - s_values[i]
            gae = delta + self.gamma * self.lmbda * (1 - d_batch[i]) * gae
            returns.insert(0, gae + s_values[i])

        returns = np.array(returns)
        adv = returns - s_values.numpy()
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)  # output: numpy array
        return returns, adv

    def update_targets(self):
        pass

    def save_model(self, path, actor_filename, critic_filename,
                   replay_filename, episode_filename, episode,
                   reward_list_filename, ep_reward_list):
        actor_file = path + actor_filename
        critic_file = path + critic_filename
        replay_file = path + replay_filename
        episode_file = path + episode_filename
        reward_list_file = path + reward_list_filename

        self.actor.model.save_weights(actor_file)
        self.critic.model.save_weights(critic_file)
        with open(replay_file, 'wb') as file:
            pickle.dump(self.buffer, file)
        with open(episode_file, 'wb') as file:
            pickle.dump(episode, file)
        with open(reward_list_file, 'wb') as file:
            pickle.dump(ep_reward_list, file)

    def load_model(self, path, actor_filename, critic_filename,
                   replay_filename, episode_filename, reward_list_filename):
        actor_file = path + actor_filename
        critic_file = path + critic_filename
        replay_file = path + replay_filename
        episode_file = path + episode_filename
        reward_list_file = path + reward_list_filename

        self.actor.model.load_weights(actor_file)
        self.actor.model_old.load_weights(actor_file)
        self.critic.model.load_weights(critic_file)

        with open(replay_file, 'rb') as file:
            self.buffer = pickle.load(file)
        with open(episode_file, 'rb') as file:
            self.start_episode = pickle.load(file)
        with open(reward_list_file, 'rb') as file:
            self.reward_list = pickle.load(file)
