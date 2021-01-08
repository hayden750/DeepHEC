''' Twin Delayed DDPG Implementation with Actor-Critic '''

from Replay_Buffer import HER_Buffer
from FeatureNet import FeatureNetwork
from OUActionNoise import OUActionNoise

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pickle


class KukaDDPGHERAgent:
    def __init__(self, state_size, action_size,
                 replacement,
                 lr_a, lr_c,
                 batch_size,
                 memory_capacity,
                 gamma,
                 upper_bound, lower_bound, clip_noise):
        self.state_size = state_size
        self.action_size = action_size
        self.replacement = replacement
        self.actor_lr = lr_a
        self.critic_lr = lr_c
        self.batch_size = batch_size
        self.memory_capacity = memory_capacity
        self.gamma = gamma
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.clip_noise = clip_noise
        self.train_step = 0
        self.actor_loss = 0
        self.policy_decay = 2
        self.done_ep = False  # Not used here

        self.start_episode = 0
        self.reward_list = []

        self.feature = FeatureNetwork(self.state_size)
        self.buffer = HER_Buffer(self.memory_capacity, self.batch_size)

        std_dev = 0.2
        self.noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

        self.actor = HERActor(self.state_size, self.action_size, self.replacement,
                              self.actor_lr, self.upper_bound, self.feature)

        self.critic = HERCritic(self.state_size, self.action_size, self.replacement,
                                    self.critic_lr, self.gamma, self.feature)

        # Initially make weights for target and model equal
        self.actor.target.set_weights(self.actor.model.get_weights())
        self.critic.target.set_weights(self.critic.model.get_weights())

    def policy(self, state, goal):
        # Get action
        sampled_action = tf.squeeze(self.actor.model([state, goal]))
        # Get noise (scalar value)
        noise = self.noise()
        # Convert into the same shape as that of the action vector
        noise_vec = noise * np.ones(self.action_size)
        # Add noise to the action
        sampled_action = sampled_action.numpy() + noise_vec
        # Make sure that the action is within bounds
        valid_action = np.clip(sampled_action, self.lower_bound, self.upper_bound)
        return np.squeeze(valid_action)

    def experience_replay(self):
        # Sample batch from buffer
        s_batch, a_batch, r_batch, ns_batch, d_batch, g_batch = self.buffer.sample()
        # Compute targets
        y = self.compute_targets(r_batch, ns_batch, d_batch, g_batch)
        # Update Critics (Q functions)
        critic_loss = self.critic.train(s_batch, a_batch, y)
        # Update Actor (Policy) (less frequently than critics)
        actor_loss = self.actor.train(s_batch, g_batch, self.critic)
        self.train_step += 1

        return actor_loss, critic_loss

    def compute_targets(self, r_batch, ns_batch, d_batch, g_batch):
        # Target smoothing
        target_actions = self.compute_target_actions(ns_batch, g_batch)
        target_critic = self.critic.target([ns_batch, target_actions])
        y = r_batch + self.gamma * (1 - d_batch) * target_critic
        return y

    def compute_target_actions(self, ns_batch, g_batch):
        target_actions = self.actor.target([ns_batch, g_batch])

        # Get noise (scalar value)
        noise = np.clip(self.noise(), -self.clip_noise, self.clip_noise)
        # Convert into the same shape as that of the action vector
        noise_vec = noise * np.ones(self.action_size)
        # Add noise to the target actions
        target_actions = target_actions.numpy() + noise_vec
        # Make sure that the action is within bounds
        clipped_target_actions = np.clip(target_actions, self.lower_bound, self.upper_bound)

        return clipped_target_actions

    def update_targets(self):
        self.actor.update_target()
        self.critic.update_target()

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
        self.actor.target.load_weights(actor_file)
        self.critic.model.load_weights(critic_file)
        self.critic.target.load_weights(critic_file)

        with open(replay_file, 'rb') as file:
            self.buffer = pickle.load(file)
        with open(episode_file, 'rb') as file:
            self.start_episode = pickle.load(file)
        with open(reward_list_file, 'rb') as file:
            self.reward_list = pickle.load(file)


class HERActor:
    def __init__(self, state_size, action_size,
                 replacement, learning_rate,
                 upper_bound, feature_model):
        print("Initialising Actor network")
        self.state_size = state_size  # shape: (w, h, c)
        self.action_size = action_size  # shape: (n, )
        self.goal_size = state_size
        self.lr = learning_rate
        self.replacement = replacement
        self.upper_bound = upper_bound
        self.train_step_count = 0

        # Create NN models
        self.feature_model = feature_model
        self.model = self.build_net()
        self.target = self.build_net()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def build_net(self):
        # Initialise weights
        last_init = tf.random_uniform_initializer(minval=-0.03, maxval=0.03)

        # State Input
        state_input = layers.Input(shape=self.state_size)

        feature = self.feature_model(state_input)
        state_out = layers.Dense(32, activation="relu")(feature)
        state_out = layers.Dense(32, activation="relu")(state_out)

        # Goal Input
        goal_input = layers.Input(shape=self.goal_size)
        goal_out = layers.Dense(32, activation="relu")(feature)
        goal_out = layers.Dense(32, activation="relu")(goal_out)

        # Both are passed through separate layer before concatenating
        concat = layers.Concatenate()([state_out, goal_out])

        out = layers.Dense(128, activation="relu")(concat)
        out = layers.Dense(64, activation="relu")(out)

        net_out = layers.Dense(self.action_size[0], activation='tanh',
                               kernel_initializer=last_init)(out)

        net_out = net_out * self.upper_bound
        model = tf.keras.Model(inputs=[state_input, goal_input], outputs=net_out)
        model.summary()
        keras.utils.plot_model(model, to_file='actor_net.png',
                               show_shapes=True, show_layer_names=True)
        return model

    def train(self, state_batch, goal_batch, critic):
        self.train_step_count += 1
        with tf.GradientTape() as tape:
            actor_weights = self.model.trainable_variables
            actions = self.model([state_batch, goal_batch])
            critic_value = critic.model([state_batch, actions])
            # -ve value is used to maximize value function
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_weights)
        self.optimizer.apply_gradients(zip(actor_grad, actor_weights))
        return actor_loss

    def update_target(self):
        if self.replacement['name'] == 'hard':
            if self.train_step_count % \
                    self.replacement['rep_iter_a'] == 0:
                self.target.set_weights(self.model.get_weights())
        else:
            w = np.array(self.model.get_weights(), dtype=object)
            w_dash = np.array(self.target.get_weights(), dtype=object)
            new_wts = self.replacement['tau'] * w + \
                      (1 - self.replacement['tau']) * w_dash
            self.target.set_weights(new_wts)


class HERCritic:
    def __init__(self, state_size, action_size,
                 replacement,
                 learning_rate,
                 gamma, feature_model):
        print("Initialising Critic network")
        self.state_size = state_size  # shape: (w, h, c)
        self.action_size = action_size  # shape: (n, )
        self.goal_size = state_size
        self.lr = learning_rate
        self.gamma = gamma
        self.replacement = replacement
        self.train_step_count = 0

        # Create NN models
        self.feature_model = feature_model
        self.model = self.build_net()
        self.target = self.build_net()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def build_net(self):
        state_input = layers.Input(shape=self.state_size)

        feature = self.feature_model(state_input)
        state_out = layers.Dense(32, activation="relu")(feature)
        state_out = layers.Dense(32, activation="relu")(state_out)

        # Goal Input
        goal_input = layers.Input(shape=self.goal_size)
        goal_out = layers.Dense(32, activation="relu")(feature)
        goal_out = layers.Dense(32, activation="relu")(goal_out)

        # Action as input
        action_input = layers.Input(shape=self.action_size)
        action_out = layers.Dense(32, activation="relu")(action_input)

        # Both are passed through separate layer before concatenating
        concat = layers.Concatenate()([state_out, goal_out, action_out])

        out = layers.Dense(128, activation="relu")(concat)
        out = layers.Dense(64, activation="relu")(out)  # leakyRelu
        # out = layers.Dense(64, activation=tf.keras.layers.LeakyReLU())(out)  # leakyRelu

        net_out = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model(inputs=[state_input, goal_input, action_input], outputs=net_out)
        model.summary()
        keras.utils.plot_model(model, to_file='critic_net.png',
                               show_shapes=True, show_layer_names=True)
        return model

    def train(self, state_batch, action_batch, y):
        self.train_step_count += 1
        with tf.GradientTape() as tape:
            critic_weights = self.model.trainable_variables
            critic_value = self.model([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_weights)
        self.optimizer.apply_gradients(zip(critic_grad, critic_weights))
        return critic_loss

    def update_target(self):
        if self.replacement['name'] == 'hard':
            if self.train_step_count % \
                    self.replacement['rep_iter_a'] == 0:
                self.target.set_weights(self.model.get_weights())
        else:
            w = np.array(self.model.get_weights(), dtype=object)
            w_dash = np.array(self.target.get_weights(), dtype=object)
            new_wts = self.replacement['tau'] * w + \
                      (1 - self.replacement['tau']) * w_dash
            self.target.set_weights(new_wts)

