# Imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.backend as K
from collections import deque
import os
import datetime

# Local Imports
from feature import FeatureNetwork, AttentionFeatureNetwork


class Actor:
    def __init__(self, input_shape, action_space, lr, epsilon, feature):
        self.state_size = input_shape
        self.action_size = action_space
        self.lr = lr
        self.upper_bound = 1.0
        self.entropy_coeff = 0.01
        self.epsilon = epsilon  # Clipping value
        self.feature_model = feature
        self.model = self.build_net()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

        # additions
        logstd = tf.Variable(np.zeros(shape=self.action_size, dtype=np.float32))
        self.model.logstd = logstd
        self.model.trainable_variables.append(logstd)

    def build_net(self):
        # input is a stack of 1-channel YUV images
        last_init = tf.random_uniform_initializer(minval=-0.03, maxval=0.03)
        state_input = tf.keras.layers.Input(shape=self.state_size)
        f = self.feature_model(state_input)
        f = tf.keras.layers.Dense(128, activation='relu', trainable=True)(f)
        f = tf.keras.layers.Dense(64, activation="relu", trainable=True)(f)
        net_out = tf.keras.layers.Dense(self.action_size, activation='tanh',
                                        kernel_initializer=last_init, trainable=True)(f)

        net_out = net_out * self.upper_bound  # element-wise product
        model = tf.keras.Model(state_input, net_out)
        tf.keras.utils.plot_model(model, to_file='actor_net.png',
                                  show_shapes=True, show_layer_names=True)
        model.summary()

        return model

    def train(self, states, advantages, actions, old_pi, c_loss):

        with tf.GradientTape() as tape:

            mean = tf.squeeze(self.model(states))
            std = tf.squeeze(tf.exp(self.model.logstd))
            new_pi = tfp.distributions.Normal(mean, std)
            ratio = tf.exp(new_pi.log_prob(tf.squeeze(actions)) -
                           old_pi.log_prob(tf.squeeze(actions)))

            adv_stack = tf.stack([advantages, advantages, advantages], axis=1)

            p1 = ratio * adv_stack
            p2 = tf.clip_by_value(ratio, 1. - self.epsilon, 1. + self.epsilon) * adv_stack
            l_clip = K.mean(K.minimum(p1, p2))
            entropy = tf.reduce_mean(new_pi.entropy())

            actor_loss = - (l_clip - c_loss + self.entropy_coeff * entropy)
            actor_weights = self.model.trainable_variables

        # outside gradient tape
        actor_grad = tape.gradient(actor_loss, actor_weights)
        self.optimizer.apply_gradients(zip(actor_grad, actor_weights))

        return actor_loss.numpy()

    def __call__(self, state):
        mean = tf.squeeze(self.model(state))
        std = tf.squeeze(tf.exp(self.model.logstd))
        return mean, std  # returns tensors

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)


class Critic:
    def __init__(self, input_shape, action_space, lr, feature):
        self.state_size = input_shape
        self.action_size = action_space
        self.lr = lr
        self.feature_model = feature
        self.model = self.build_net()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def build_net(self):
        # state input is a stack of 1-D YUV images
        state_input = tf.keras.layers.Input(shape=self.state_size)
        feature = self.feature_model(state_input)
        out = tf.keras.layers.Dense(128, activation="relu", trainable=True)(feature)
        out = tf.keras.layers.Dense(64, activation="relu", trainable=True)(out)
        out = tf.keras.layers.Dense(32, activation="relu", trainable=True)(out)
        net_out = tf.keras.layers.Dense(1, trainable=True)(out)

        # Outputs single value for a given state = V(s)
        model = tf.keras.Model(inputs=state_input, outputs=net_out)
        tf.keras.utils.plot_model(model, to_file='critic_net.png',
                                  show_shapes=True, show_layer_names=True)
        model.summary()

        return model

    def train(self, states, returns):
        with tf.GradientTape() as tape:
            critic_weights = self.model.trainable_variables
            critic_value = tf.squeeze(self.model(states))
            critic_loss = tf.math.reduce_mean(tf.square(returns - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_weights)
        self.optimizer.apply_gradients(zip(critic_grad, critic_weights))
        return critic_loss.numpy()

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)


class PPOAgent:
    def __init__(self, env, SEASONS, success_value, lr_a, lr_c, epochs,
                 training_batch, batch_size, epsilon, gamma, lmbda, use_attention):
        self.env = env
        self.action_size = self.env.action_space.shape[0]
        self.state_size = self.env.observation_space.shape
        self.upper_bound = self.env.action_space.high
        self.SEASONS = SEASONS
        self.episode = 0
        self.replay_count = 0
        self.success_value = success_value
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.epochs = epochs
        self.training_batch = training_batch
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.lmbda = lmbda
        self.use_attention = use_attention

        self.shuffle = True

        # Create Actor-Critic network models
        if self.use_attention:
            self.feature = AttentionFeatureNetwork(self.state_size, lr_a)
        else:
            self.feature = FeatureNetwork(self.state_size, lr_a)

        self.actor = Actor(input_shape=self.state_size, action_space=self.action_size, lr=self.lr_a,
                           epsilon=self.epsilon, feature=self.feature)
        self.critic = Critic(input_shape=self.state_size, action_space=self.action_size, lr=self.lr_c,
                             feature=self.feature)

        # do not change
        self.log_std = -0.5 * np.ones(self.action_size, dtype=np.float32)
        self.std = np.exp(self.log_std)

    def policy(self, state):
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)

        mean, std = self.actor(tf_state)

        # pi = tfp.distributions.Normal(mean, std)
        # action = pi.sample()
        #
        # valid_action = tf.clip_by_value(action, -self.upper_bound, self.upper_bound)
        # return valid_action.numpy()

        action = mean + np.random.uniform(-self.upper_bound, self.upper_bound, size=mean.shape) * std
        action = np.clip(action, -self.upper_bound, self.upper_bound)

        return action

    def compute_advantages(self, r_batch, s_batch, ns_batch, d_batch):
        gamma = self.gamma
        lmbda = self.lmbda
        s_values = tf.squeeze(self.critic.model(s_batch))  # input: tensor
        ns_values = tf.squeeze(self.critic.model(ns_batch))
        returns = []
        gae = 0  # generalized advantage estimate
        for i in reversed(range(len(r_batch))):
            delta = r_batch[i] + gamma * ns_values[i] * (1 - d_batch[i]) - s_values[i]
            gae = delta + gamma * lmbda * (1 - d_batch[i]) * gae
            returns.insert(0, gae + s_values[i])

        returns = np.array(returns)
        adv = returns - s_values.numpy()
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)  # output: numpy array
        return adv, returns

    def replay(self, states, actions, rewards, dones, next_states):

        n_split = len(rewards) // self.batch_size

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)

        advantages, target = self.compute_advantages(rewards, states, next_states, dones)

        s_split = tf.split(states, n_split)
        a_split = tf.split(actions, n_split)
        t_split = tf.split(target, n_split)
        adv_split = tf.split(advantages, n_split)
        indexes = np.arange(n_split, dtype=int)

        # current policy
        mean, std = self.actor(states)
        pi = tfp.distributions.Normal(mean, std)

        a_loss_list = []
        c_loss_list = []

        np.random.shuffle(indexes)
        for _ in range(self.epochs):
            for i in indexes:
                old_pi = pi[i * self.batch_size: (i + 1) * self.batch_size]
                # Update critic
                c_loss = self.critic.train(s_split[i], t_split[i])
                c_loss_list.append(c_loss)
                # Update actor
                a_loss = self.actor.train(s_split[i], adv_split[i], a_split[i], old_pi, c_loss)
                a_loss_list.append(a_loss)

        self.replay_count += 1

        return np.mean(a_loss_list), np.mean(c_loss_list)

    # Validation Routine
    def validate(self, env, max_eps=50):

        ep_reward_list = []
        for ep in range(max_eps):
            obsv = env.reset()
            state = np.asarray(obsv, dtype=np.float32) / 255.0  # convert into float array
            t = 0
            ep_reward = 0
            while True:
                action = self.policy(state)
                next_obsv, reward, done, _ = env.step(action)
                next_state = np.asarray(next_obsv, dtype=np.float32) / 255.0
                state = next_state
                ep_reward += reward
                t += 1
                if done:
                    ep_reward_list.append(ep_reward)
                    break

        mean_ep_reward = np.mean(ep_reward_list)
        return mean_ep_reward

    def run(self):

        #####################
        # TENSORBOARD SETTINGS
        TB_LOG = True  # enable / disable tensorboard logging

        if TB_LOG:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = 'logs/train/' + current_time
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        ############################
        path = './'

        filename = path + 'result_ppo_clip.txt'
        if os.path.exists(filename):
            os.remove(filename)
        else:
            print('The file does not exist. It will be created.')

        state = self.env.reset()
        state = np.asarray(state, dtype=np.float32) / 255.0  # convert into float array
        done, score = False, 0
        best_score = -np.inf
        val_score = -np.inf
        val_scores = deque(maxlen=50)
        s = 0
        s_scores = deque(maxlen=50)  # Last n season scores
        while True:
            # Instantiate or reset games memory
            s_score = 0
            states, next_states, actions, rewards, dones = [], [], [], [], []
            for t in range(self.training_batch):  # self.training_batch
                action = self.policy(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.asarray(next_state, dtype=np.float32) / 255.0

                states.append(state)
                next_states.append(next_state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)

                state = next_state
                score += reward
                s_score += reward

                if done:
                    self.episode += 1
                    state, done, score = self.env.reset(), False, 0
                    state = np.asarray(state, dtype=np.float32) / 255.0

            a_loss, c_loss = self.replay(states, actions, rewards, dones, next_states)

            # Decay variables
            # self.actor.epsilon *= 0.999
            # self.actor.entropy_coeff *= 0.998

            # After season
            success_rate = s_score / sum(dones)
            s_scores.append(s_score)
            mean_s_score = np.mean(s_scores)
            if mean_s_score > best_score:
                self.save_model(path, 'actor_weights.h5', 'critic_weights.h5')
                print("Season ", s)
                print("Updated best score {}->{}, Model saved!".format(best_score, mean_s_score))
                best_score = mean_s_score
            s += 1

            if s % 10 == 0:
                print("Season {} score: {}, Mean score: {}".format(s, s_score, mean_s_score))
                val_score = self.validate(self.env)
                val_scores.append(val_score)
                mean_val_score = np.mean(val_scores)
                print("Season: {}, Validation score: {}, Mean Validation score: {}".format(s, val_score, mean_val_score))

            if TB_LOG:  # tensorboard logging
                with train_summary_writer.as_default():
                    tf.summary.scalar('1. Season score', s_score, step=s)
                    tf.summary.scalar('2. Average Season Score', mean_s_score, step=s)
                    tf.summary.scalar('3. Success rate', success_rate, step=s)
                    tf.summary.scalar('4. Validation score', val_score, step=s)
                    tf.summary.scalar('5. Actor Loss', a_loss, step=s)
                    tf.summary.scalar('6. Critic Loss', c_loss, step=s)

            if best_score > self.success_value:
                print("Problem solved in {} episodes with score {}".format(self.episode, best_score))
                break
            if s >= self.SEASONS:
                break

        self.env.close()

    def save_model(self, path, actor_filename, critic_filename):
        actor_file = path + actor_filename
        critic_file = path + critic_filename
        self.actor.save_weights(actor_file)
        self.critic.save_weights(critic_file)

    def load_model(self, path, actor_filename, critic_filename):
        actor_file = path + actor_filename
        critic_file = path + critic_filename
        self.actor.model.load_weights(actor_file)
        self.critic.model.load_weights(critic_file)