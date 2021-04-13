import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import tensorflow_probability as tfp

import Box2D
import gym

from collections import deque
from packaging import version
import copy
import random

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

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
################################################

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


class Buffer:
    def __init__(self, buffer_capacity, batch_size):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer = deque(maxlen=self.buffer_capacity)
        self.size = 0

    def __len__(self):
        return len(self.buffer)

    def record(self, state, action, reward, next_state, done):
        if self.size <= self.buffer_capacity:
            self.size += 1
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        valid_batch_size = min(len(self.buffer), self.batch_size)
        mini_batch = random.sample(self.buffer, valid_batch_size)

        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        for i in range(valid_batch_size):
            state_batch.append(mini_batch[i][0])
            action_batch.append(mini_batch[i][1])
            reward_batch.append(mini_batch[i][2])
            next_state_batch.append(mini_batch[i][3])
            done_batch.append(mini_batch[i][4])

        # convert to tensors
        state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float32)
        action_batch = tf.convert_to_tensor(action_batch, dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(next_state_batch, dtype=tf.float32)
        done_batch = tf.convert_to_tensor(done_batch, dtype=tf.float32)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def get_all_samples(self):
        s_batch = []
        a_batch = []
        r_batch = []
        ns_batch = []
        d_batch = []
        for i in range(len(self.buffer)):
            s_batch.append(self.buffer[i][0])
            a_batch.append(self.buffer[i][1])
            r_batch.append(self.buffer[i][2])
            ns_batch.append(self.buffer[i][3])
            d_batch.append(self.buffer[i][4])

        return s_batch, a_batch, r_batch, ns_batch, d_batch


class Actor:
    def __init__(self, input_shape, action_space, lr):
        self.state_size = input_shape
        self.action_size = action_space
        self.lr = lr
        self.upper_bound = 1.0
        self.model = self.build_net()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

        # additions
        logstd = tf.Variable(np.zeros(shape=self.action_size, dtype=np.float32))
        self.model.logstd = logstd
        self.model.trainable_variables.append(logstd)

    def build_net(self):
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        state_input = layers.Input(shape=self.state_size)
        l = layers.Dense(128, activation='relu')(state_input)
        l = layers.Dense(64, activation='relu')(l)
        l = layers.Dense(64, activation='relu')(l)
        net_out = layers.Dense(self.action_size, activation='tanh',
                               kernel_initializer=last_init)(l)
        net_out = net_out * self.upper_bound
        model = keras.Model(state_input, net_out)
        model.summary()
        return model

    def train(self, states, advantages, actions, old_pi, critic, b, s_batch):
        with tf.GradientTape() as tape:
            epsilon = 0.2

            mean = tf.squeeze(self.model(states))
            std = tf.squeeze(tf.exp(self.model.logstd))
            new_pi = tfp.distributions.Normal(mean, std)
            ratio = tf.exp(new_pi.log_prob(tf.squeeze(actions)) -
                           old_pi.log_prob(tf.squeeze(actions)))

            # Change stack amount for action size
            # adv_stack = tf.stack([advantages, advantages, advantages, advantages], axis=1)
            adv_stack = advantages

            p1 = ratio * adv_stack
            p2 = tf.clip_by_value(ratio, 1. - epsilon, 1. + epsilon) * adv_stack

            ppo_loss = K.mean(K.minimum(p1, p2))

            # Off-policy loss
            mean_off = tf.squeeze(self.model(s_batch))
            q_values = critic.model([s_batch, mean_off])
            sum_q_values = K.sum(K.mean(q_values))
            off_loss = ((b / len(s_batch)) * sum_q_values)

            actor_loss = -tf.reduce_sum(ppo_loss + off_loss)
            # actor_loss = -ppo_loss
            actor_weights = self.model.trainable_variables

        # outside gradient tape
        actor_grad = tape.gradient(actor_loss, actor_weights)
        self.optimizer.apply_gradients(zip(actor_grad, actor_weights))

        return actor_loss.numpy()

    def gaussian_likelihood(self, actions, pred):
        log_std = -0.5 * np.ones(self.action_size, dtype=np.float32)
        pre_sum = -0.5 * (((actions - pred) / (K.exp(log_std) + 1e-8)) ** 2 + 2 * log_std + K.log(2 * np.pi))
        return K.sum(pre_sum)

    def __call__(self, state):
        mean = tf.squeeze(self.model(state))
        std = tf.squeeze(tf.exp(self.model.logstd))
        return mean, std  # returns tensors


class Critic:
    def __init__(self, state_size, action_size,
                 replacement, learning_rate, gamma):
        print("Initialising Critic network")
        self.state_size = state_size  # shape: (w, h, c)
        self.action_size = action_size  # shape: (n, )
        self.lr = learning_rate
        self.gamma = gamma
        self.replacement = replacement
        self.train_step_count = 0

        # Create NN models
        self.model = self.build_net()
        # self.target = self.build_net()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def build_net(self):
        state_input = layers.Input(shape=self.state_size)
        state_out = layers.Dense(32, activation="relu")(state_input)

        # Action as input
        action_input = layers.Input(shape=self.action_size)
        action_out = layers.Dense(32, activation="relu")(action_input)

        # Both are passed through separate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(128, activation="relu")(concat)
        out = layers.Dense(64, activation="relu")(out)  # leakyRelu
        # out = layers.Dense(64, activation=tf.keras.layers.LeakyReLU())(out)  # leakyRelu

        net_out = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model(inputs=[state_input, action_input], outputs=net_out)
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

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)


class Baseline:
    def __init__(self, input_shape, action_space, lr):
        self.state_size = input_shape
        self.action_size = action_space
        self.lr = lr
        self.model = self.build_net()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def build_net(self):
        state_input = layers.Input(shape=self.state_size)
        out = layers.Dense(64, activation="relu")(state_input)
        out = layers.Dense(64, activation="relu")(out)
        out = layers.Dense(64, activation="relu")(out)
        net_out = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model(inputs=state_input, outputs=net_out)
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


class PPOAgent:
    def __init__(self, env, EPISODES, success_value, lr, epochs, training_batch, batch_size):
        self.env = env
        self.action_size = self.env.action_space.shape[0]
        self.state_size = self.env.observation_space.shape
        self.upper_bound = self.env.action_space.high
        self.EPISODES = EPISODES
        self.episode = 0
        self.replay_count = 0
        self.success_value = success_value
        self.lr = lr
        self.epochs = epochs
        self.training_batch = training_batch
        self.batch_size = batch_size

        self.shuffle = True

        self.replacement = [
            dict(name='soft', tau=0.005),
            dict(name='hard', rep_iter_a=600, rep_iter_c=500)
        ][0]
        self.gamma = 0.99
        self.buffer = Buffer(50000, self.batch_size)

        # Create Actor-Critic and Baseline network models
        self.actor = Actor(input_shape=self.state_size, action_space=self.action_size, lr=self.lr)
        self.critic = Critic(state_size=self.state_size, action_size=self.action_size,
                             replacement=self.replacement, learning_rate=self.lr, gamma=self.gamma)
        self.baseline = Baseline(input_shape=self.state_size, action_space=self.action_size, lr=self.lr)

        # do not change below
        self.log_std = -0.5 * np.ones(self.action_size, dtype=np.float32)
        self.std = np.exp(self.log_std)

    def policy(self, state):
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)

        mean, std = self.actor(tf_state)

        # action = mean + np.random.uniform(-self.upper_bound, self.upper_bound, size=mean.shape) * std
        action = mean + np.random.uniform(-self.upper_bound, self.upper_bound) * std
        action = np.clip(action, -self.upper_bound, self.upper_bound)

        return action

    def gaussian_likelihood(self, action, mean, log_std):
        # https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/sac/policies.py
        pre_sum = -0.5 * (((action - mean) / (np.exp(log_std) + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        return np.sum(pre_sum)

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

    def compute_advantages2(self, r_batch, s_batch, ns_batch, d_batch):
        gamma = 0.993
        lmbda = 0.5
        s_values = tf.squeeze(self.baseline.model(s_batch))  # input: tensor
        ns_values = tf.squeeze(self.baseline.model(ns_batch))
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

    def compute_targets(self, r_batch, ns_batch, d_batch):
        mean = self.actor.model(ns_batch)
        std = tf.exp(self.actor.model.logstd)

        target_critic = self.critic.model([ns_batch, mean])
        y = r_batch + self.gamma * (1 - d_batch) * target_critic
        return y

    # Might be wrong
    def compute_adv_bar(self, s_batch, a_batch):
        mean = self.actor.model(s_batch)
        # std = tf.exp(self.actor.model.logstd)

        # actions = mean + np.random.uniform(-self.upper_bound, self.upper_bound, size=mean.shape) * std
        # actions = np.clip(actions, -self.upper_bound, self.upper_bound)
        x = tf.squeeze(a_batch) - tf.squeeze(mean)
        y = tf.squeeze(self.critic.model([s_batch, mean]))
        adv_bar = y * x
        return adv_bar

    def replay(self, states, actions, rewards, dones, next_states):
        print("Training...")

        n_split = len(rewards) // self.batch_size
        n_samples = n_split * self.batch_size

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)

        advantages, target = self.compute_advantages2(rewards, states, next_states, dones)

        # current policy
        mean, std = self.actor(states)
        pi = tfp.distributions.Normal(mean, std)

        use_CV = True
        v = 0.2
        if use_CV:
            # Compute critic-based advantage
            adv_bar = self.compute_adv_bar(states, mean)
            ls = advantages - adv_bar
            b = 1
        else:
            ls = advantages
            b = v

        ls *= (1 - v)

        s_split = tf.split(states, n_split)
        a_split = tf.split(actions, n_split)
        t_split = tf.split(target, n_split)
        adv_split = tf.split(advantages, n_split)
        ls_split = tf.split(ls, n_split)
        indexes = np.arange(n_split, dtype=int)

        a_loss, c_loss = 0, 0

        np.random.shuffle(indexes)
        for _ in range(self.epochs):
            s_batch, a_batch, r_batch, ns_batch, d_batch = self.buffer.sample()
            for i in indexes:
                old_pi = pi[i * self.batch_size: (i + 1) * self.batch_size]
                # Update actor
                a_loss = self.actor.train(s_split[i], ls_split[i], a_split[i], old_pi, self.critic, b, s_batch)
                # Update baseline
                v_loss = self.baseline.train(s_split[i], t_split[i])

            # Update critic
            y = self.compute_targets(r_batch, ns_batch, d_batch)
            c_loss = self.critic.train(s_batch, a_batch, y)

        self.replay_count += 1

        return a_loss, c_loss

    def run_batch(self):
        state = self.env.reset()
        # state = np.asarray(state, dtype=np.float32) / 255.0  # convert into float array
        done, score, SAVING = False, 0, ''
        scores = deque(maxlen=100)
        best_score = -np.inf
        while True:
            # Instantiate or reset games memory
            states, next_states, actions, rewards, dones = [], [], [], [], []
            for t in range(self.training_batch):
                # if self.episode % 100 < 5:
                #     env.render()
                action = self.policy(state)
                next_state, reward, done, _ = self.env.step(action)
                # next_state = np.asarray(next_state, dtype=np.float32) / 255.0

                states.append(state)
                next_states.append(next_state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                self.buffer.record(state, action, reward, next_state, done)

                state = next_state
                score += reward

                if done:
                    self.episode += 1
                    scores.append(score)
                    average = np.mean(scores)
                    if average > best_score and self.episode > 100:
                        # print("Updated best score: {}->{}".format(best_score, average))
                        best_score = average
                        SAVING = "updated!"
                    print("episode: {}/{}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, score,
                                                                                 average, SAVING))

                    state, done, score, SAVING = self.env.reset(), False, 0, ''
                    # state = np.asarray(state, dtype=np.float32) / 255.0

            self.replay(states, actions, rewards, dones, next_states)
            if best_score > self.success_value:
                print("Problem solved in {} episodes with score {}".format(self.episode, best_score))
                break
            if self.episode >= self.EPISODES:
                break

        self.env.close()


if __name__ == "__main__":
    ##### Hyper-parameters
    EPISODES = 20000
    success_value = 250
    lr = 0.0002
    epochs = 10
    training_batch = 1024
    batch_size = 64

    # env = gym.make('Pendulum-v0')
    # env = gym.make('BipedalWalker-v3')
    # env = gym.make('LunarLanderContinuous-v2')
    env = gym.make('MountainCarContinuous-v0')
    agent = PPOAgent(env, EPISODES, success_value, lr, epochs, training_batch, batch_size)
    agent.run_batch()  # train as PPO
