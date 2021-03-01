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

    def train(self, states, advantages, actions, old_pi):

        with tf.GradientTape() as tape:
            epsilon = 0.2

            mean = tf.squeeze(self.model(states))
            std = tf.squeeze(tf.exp(self.model.logstd))
            new_pi = tfp.distributions.Normal(mean, std)
            ratio = tf.exp(new_pi.log_prob(tf.squeeze(actions)) -
                           old_pi.log_prob(tf.squeeze(actions)))

            # Change stack amount for action size
            adv_stack = tf.stack([advantages, advantages, advantages, advantages], axis=1)
            #adv_stack = advantages

            p1 = ratio * adv_stack
            p2 = tf.clip_by_value(ratio, 1. - epsilon, 1. + epsilon) * adv_stack

            actor_loss = -K.mean(K.minimum(p1, p2))
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

        # Create Actor-Critic network models
        self.actor = Actor(input_shape=self.state_size, action_space=self.action_size, lr=self.lr)
        self.critic = Critic(input_shape=self.state_size, action_space=self.action_size, lr=self.lr)

        # do not change below
        self.log_std = -0.5 * np.ones(self.action_size, dtype=np.float32)
        self.std = np.exp(self.log_std)

    def policy(self, state):
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)

        mean, std = self.actor(tf_state)

        action = mean + np.random.uniform(-self.upper_bound, self.upper_bound, size=mean.shape) * std
        #action = mean + np.random.uniform(-self.upper_bound, self.upper_bound) * std
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
        print("Training...")

        n_split = len(rewards) // self.batch_size
        n_samples = n_split * self.batch_size

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)

        advantages, target = self.compute_advantages2(rewards, states, next_states, dones)

        s_split = tf.split(states, n_split)
        a_split = tf.split(actions, n_split)
        t_split = tf.split(target, n_split)
        adv_split = tf.split(advantages, n_split)
        indexes = np.arange(n_split, dtype=int)

        # current policy
        mean, std = self.actor(states)
        pi = tfp.distributions.Normal(mean, std)

        a_loss, c_loss = 0, 0

        np.random.shuffle(indexes)
        for _ in range(self.epochs):
            for i in indexes:
                old_pi = pi[i * self.batch_size: (i + 1) * self.batch_size]
                # Update actor
                a_loss = self.actor.train(s_split[i], adv_split[i], a_split[i], old_pi)
                # Update critic
                c_loss = self.critic.train(s_split[i], t_split[i])

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
                if self.episode % 100 < 5:
                    env.render()
                action = self.policy(state)
                next_state, reward, done, _ = self.env.step(action)
                # next_state = np.asarray(next_state, dtype=np.float32) / 255.0

                states.append(state)
                next_states.append(next_state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)

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

    #env = gym.make('Pendulum-v0')
    env = gym.make('BipedalWalker-v3')
    #env = gym.make('LunarLanderContinuous-v2')
    #env = gym.make('MountainCarContinuous-v0')
    agent = PPOAgent(env, EPISODES, success_value, lr, epochs, training_batch, batch_size)
    agent.run_batch()  # train as PPO
