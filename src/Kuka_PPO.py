''' PPO Implementation with Actor-Critic
    based on the algorithm set out by OpenAI
    https://spinningup.openai.com/en/latest/algorithms/ppo.html '''

# This PPO implementation runs k number of episodes
# on a single policy and stores the experience in
# a temporary buffer. The policy is then updated
# a single time from a random sample from the buffer.
# The buffer is then cleared and the process repeats.

from Replay_Buffer import Buffer
from FeatureNet import FeatureNetwork
from OUActionNoise import OUActionNoise

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

import pickle


class KukaPPOAgent:
    def __init__(self, state_size, action_size,
                 lr_a, lr_c,
                 batch_size,
                 memory_capacity, lmbda,
                 epsilon, gamma, upper_bound,
                 lower_bound):
        self.state_size = state_size
        self.action_size = action_size
        self.actor_lr = lr_a
        self.critic_lr = lr_c
        self.batch_size = batch_size
        self.memory_capacity = memory_capacity
        self.gamma = gamma
        self.lmbda = lmbda
        self.epsilon = epsilon
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.train_step = 0

        self.done_ep = False
        self.ep_count = 0
        self.update_rate = 10

        self.start_episode = 0
        self.reward_list = []

        self.feature = FeatureNetwork(self.state_size)
        self.buffer = Buffer(self.memory_capacity, self.batch_size)
        self.actor = PPOActor(self.state_size, self.action_size,
                              self.actor_lr, self.epsilon,
                              self.upper_bound, self.feature)
        self.critic = PPOCritic(self.state_size, self.action_size,
                                self.critic_lr, self.gamma, self.feature)

        # Initialise old policy
        self.actor.model_old.set_weights(self.actor.model.get_weights())

    def policy(self, state):
        # Get action
        action = tf.squeeze(self.actor.model(state))
        return action

    # Not technically EP, just named so it runs in main
    def experience_replay(self):
        actor_loss, critic_loss = 0, 0
        if self.done_ep:
            self.ep_count += 1
            if self.ep_count % self.update_rate == 0:
                print("Updating policy...")
                s_batch, a_batch, r_batch, ns_batch, d_batch = self.buffer.sample()
                returns, advantages = self.compute_advantages(r_batch, s_batch, ns_batch, d_batch)
                actor_loss = self.actor.train(s_batch, advantages)
                critic_loss = self.critic.train(s_batch, returns)
                self.buffer.buffer.clear()
                self.buffer.size = 0
        return actor_loss, critic_loss

    def compute_advantages(self, r_batch, s_batch, ns_batch, d_batch):
        values = self.critic.model([s_batch])
        ns_values = self.critic.model([ns_batch])
        returns = []
        gae = 0
        for i in reversed(range(len(r_batch))):
            delta = r_batch[i] + self.gamma * ns_values[i] * (1 - d_batch[i]) - values[i]
            gae = delta + self.gamma * self.lmbda * (1 - d_batch[i]) * gae
            returns.insert(0, gae + values[i])

        adv = np.array(returns) - values
        return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

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


class PPOActor:
    def __init__(self, state_size, action_size, learning_rate,
                 epsilon, upper_bound, feature_model):
        print("Initialising Actor network")
        self.state_size = state_size  # shape: (w, h, c)
        self.action_size = action_size  # shape: (n, )
        self.lr = learning_rate
        self.epsilon = epsilon
        self.upper_bound = upper_bound
        self.train_step_count = 0

        # Create NN model
        self.feature_model = feature_model
        self.model = self.build_net()
        self.model_old = self.build_net()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def build_net(self):
        # Initialise weights
        last_init = tf.random_uniform_initializer(minval=-0.03, maxval=0.03)

        state_input = layers.Input(shape=self.state_size)
        feature = self.feature_model(state_input)

        l2 = layers.Dense(64, activation='relu')(feature)
        net_out = layers.Dense(self.action_size[0], activation='tanh',
                               kernel_initializer=last_init)(l2)

        net_out = net_out * self.upper_bound
        model = keras.Model(state_input, net_out)
        model.summary()
        keras.utils.plot_model(model, to_file='actor_net.png',
                               show_shapes=True, show_layer_names=True)
        return model

    def train(self, s_batch, advantages):
        self.train_step_count += 1
        with tf.GradientTape() as tape:
            actor_weights = self.model.trainable_variables
            actor_policy = self.model([s_batch])
            actor_policy_old = self.model_old([s_batch])
            ratio = (actor_policy + 1e-9) / (actor_policy_old + 1e-9)
            # ratio = K.exp(K.log(actor_policy + 1e-8) - K.log(actor_policy_old + 1e-8))
            p1 = ratio * advantages
            p2 = K.clip(ratio, min_value=1 - self.epsilon, max_value=1 + self.epsilon) * advantages
            actor_loss = -K.mean(K.minimum(p1, p2))

        actor_grad = tape.gradient(actor_loss, actor_weights)
        self.model_old = self.model
        self.optimizer.apply_gradients(zip(actor_grad, actor_weights))
        return actor_loss


class PPOCritic:
    def __init__(self, state_size, action_size,
                 learning_rate, gamma, feature_model):
        print("Initialising Critic network")
        self.state_size = state_size  # shape: (w, h, c)
        self.action_size = action_size  # shape: (n, )
        self.lr = learning_rate
        self.gamma = gamma
        self.train_step_count = 0

        # Create NN models
        self.feature_model = feature_model
        self.model = self.build_net()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def build_net(self):
        state_input = layers.Input(shape=self.state_size)

        feature = self.feature_model(state_input)
        state_out = layers.Dense(32, activation="relu")(feature)
        state_out = layers.Dense(32, activation="relu")(state_out)

        out = layers.Dense(128, activation="relu")(state_out)
        out = layers.Dense(64, activation="relu")(out)  # leakyRelu
        # out = layers.Dense(64, activation=tf.keras.layers.LeakyReLU())(out)  # leakyRelu

        net_out = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model(inputs=state_input, outputs=net_out)
        model.summary()
        keras.utils.plot_model(model, to_file='critic_net.png',
                               show_shapes=True, show_layer_names=True)
        return model

    def train(self, states, returns):
        self.train_step_count += 1
        with tf.GradientTape() as tape:
            critic_weights = self.model.trainable_variables
            critic_value = self.model([states])
            critic_loss = tf.math.reduce_mean(tf.square(returns - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_weights)
        self.optimizer.apply_gradients(zip(critic_grad, critic_weights))
        return critic_loss
