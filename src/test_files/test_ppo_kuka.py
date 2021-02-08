import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import tensorflow_probability as tfp

from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv

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


######################
# FEATURE NETWORK
#####################
class FeatureNetwork:
    def __init__(self, state_size, learning_rate=1e-3):
        print("Initialising Feature network")
        self.state_size = state_size
        self.lr = learning_rate
        # create NN models
        self.model = self._build_net()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def _build_net(self):
        img_input = layers.Input(shape=self.state_size)

        # shared convolutional layers
        conv1 = layers.Conv2D(15, kernel_size=5, strides=2,
                              padding="SAME", activation="relu")(img_input)
        bn1 = layers.BatchNormalization()(conv1)
        conv2 = layers.Conv2D(32, kernel_size=5, strides=2,
                              padding="SAME", activation='relu')(bn1)
        bn2 = layers.BatchNormalization()(conv2)
        conv3 = layers.Conv2D(32, kernel_size=5, strides=2,
                              padding="SAME", activation='relu')(bn2)
        bn3 = layers.BatchNormalization()(conv3)
        f1 = layers.Flatten()(bn3)
        fc1 = layers.Dense(128, activation='relu')(f1)
        fc2 = layers.Dense(64, activation='relu')(fc1)
        model = tf.keras.Model(inputs=img_input, outputs=fc2)
        print('shared feature network')
        model.summary()
        keras.utils.plot_model(model, to_file='feature_net.png',
                               show_shapes=True, show_layer_names=True)
        return model

    def __call__(self, state):
        return self.model(state)


class Actor:
    def __init__(self, input_shape, action_space, lr, feature):
        self.state_size = input_shape
        self.action_size = action_space
        self.lr = lr
        self.upper_bound = 1.0
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
        model.summary()

        return model

    def train(self, states, advantages, actions, old_pi):

        with tf.GradientTape() as tape:
            epsilon = 0.07

            mean = tf.squeeze(self.model(states))
            std = tf.squeeze(tf.exp(self.model.logstd))
            new_pi = tfp.distributions.Normal(mean, std)
            ratio = tf.exp(new_pi.log_prob(tf.squeeze(actions)) -
                           old_pi.log_prob(tf.squeeze(actions)))

            adv_stack = tf.stack([advantages, advantages, advantages], axis=1)

            p1 = ratio * adv_stack
            p2 = tf.clip_by_value(ratio, 1. - epsilon, 1. + epsilon) * adv_stack

            actor_loss = -K.mean(K.minimum(p1, p2))
            actor_weights = self.model.trainable_variables

        # outside gradient tape
        actor_grad = tape.gradient(actor_loss, actor_weights)
        self.optimizer.apply_gradients(zip(actor_grad, actor_weights))

        return actor_loss.numpy()

    def __call__(self, state):
        mean = tf.squeeze(self.model(state))
        std = tf.squeeze(tf.exp(self.model.logstd))
        return mean, std  # returns tensors


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
        self.feature = FeatureNetwork(self.state_size)
        self.actor = Actor(input_shape=self.state_size, action_space=self.action_size, lr=self.lr,
                           feature=self.feature)
        self.critic = Critic(input_shape=self.state_size, action_space=self.action_size, lr=self.lr,
                             feature=self.feature)

        # do not change below
        self.log_std = -0.5 * np.ones(self.action_size, dtype=np.float32)
        self.std = np.exp(self.log_std)

    def policy(self, state):
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)

        mean, std = self.actor(tf_state)

        action = mean + np.random.uniform(-self.upper_bound, self.upper_bound, size=mean.shape) * std
        action = np.clip(action, -self.upper_bound, self.upper_bound)

        return action

    def compute_advantages(self, r_batch, s_batch, ns_batch, d_batch):
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

        advantages, target = self.compute_advantages(rewards, states, next_states, dones)

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
        state = np.asarray(state, dtype=np.float32) / 255.0  # convert into float array
        done, score, SAVING = False, 0, ''
        scores = deque(maxlen=100)
        best_score = -np.inf
        while True:
            # Instantiate or reset games memory
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

                if done:
                    self.episode += 1
                    scores.append(score)
                    average = np.mean(scores)
                    if average > best_score and self.episode > 100:
                        print("Updated best score: {}->{}".format(best_score, average))
                        best_score = average
                        SAVING = "updated!"
                    print("episode: {}/{}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, score,
                                                                                 average, SAVING))

                    state, done, score, SAVING = self.env.reset(), False, 0, ''
                    state = np.asarray(state, dtype=np.float32) / 255.0

            self.replay(states, actions, rewards, dones, next_states)
            if best_score > self.success_value:
                print("Problem solved in {} episodes with score {}".format(self.episode, best_score))
                break
            if self.episode >= self.EPISODES:
                break

        self.env.close()


if __name__ == "__main__":

    ##### Hyper-parameters
    EPISODES = 50000
    success_value = 0.7
    lr = 0.0002
    epochs = 10
    training_batch = 1024
    batch_size = 128

    env = KukaDiverseObjectEnv(renders=False,
                               isDiscrete=False,
                               maxSteps=20,
                               removeHeightHack=False)
    agent = PPOAgent(env, EPISODES, success_value, lr, epochs, training_batch, batch_size)
    agent.run_batch()  # train as PPO




