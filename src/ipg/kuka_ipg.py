# Imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import os
import datetime
import random

from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from packaging import version


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
        conv1 = layers.Conv2D(16, kernel_size=5, strides=2,
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

    def train(self, states, advantages, actions, old_pi, critic, b, s_batch):

        with tf.GradientTape() as tape:
            mean = tf.squeeze(self.model(states))

            # On-policy ppo loss
            std = tf.squeeze(tf.exp(self.model.logstd))
            new_pi = tfp.distributions.Normal(mean, std)
            ratio = tf.exp(new_pi.log_prob(tf.squeeze(actions)) -
                           old_pi.log_prob(tf.squeeze(actions)))

            adv_stack = tf.stack([advantages, advantages, advantages], axis=1)

            p1 = ratio * adv_stack
            p2 = tf.clip_by_value(ratio, 1. - self.epsilon, 1. + self.epsilon) * adv_stack
            ppo_loss = K.mean(K.minimum(p1, p2))

            # Off-policy loss
            mean_off = self.model(s_batch)
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

    def __call__(self, state):
        mean = tf.squeeze(self.model(state))
        std = tf.squeeze(tf.exp(self.model.logstd))
        return mean, std  # returns tensors

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)


class DDPGCritic:
    def __init__(self, state_size, action_size,
                 replacement,
                 learning_rate,
                 gamma, feature_model):
        print("Initialising Critic network")
        self.state_size = state_size  # shape: (w, h, c)
        self.action_size = action_size  # shape: (n, )
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
            critic_values = self.model(states)
            critic_loss = tf.math.reduce_mean(tf.square(returns - critic_values))

        critic_grad = tape.gradient(critic_loss, critic_weights)
        self.optimizer.apply_gradients(zip(critic_grad, critic_weights))
        return critic_loss.numpy()

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)


class QPROPAgent:
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

        self.buffer = Buffer(100000, self.batch_size)
        replacement = [
            dict(name='soft', tau=0.005),
            dict(name='hard', rep_iter_a=600, rep_iter_c=500)
        ][0]

        # Create Actor-Critic network models
        if self.use_attention:
            self.feature = FeatureNetwork(self.state_size, lr_a)  # TODO add attention network
        else:
            self.feature = FeatureNetwork(self.state_size, lr_a)

        self.actor = Actor(input_shape=self.state_size, action_space=self.action_size, lr=self.lr_a,
                           epsilon=self.epsilon, feature=self.feature)
        self.critic = DDPGCritic(state_size=self.state_size, action_size=self.action_size, replacement=replacement,
                                 learning_rate=self.lr_c, gamma=self.gamma, feature_model=self.feature)
        self.baseline = Baseline(input_shape=self.state_size, action_space=self.action_size, lr=self.lr_c,
                                 feature=self.feature)

    def policy(self, state):
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)

        mean, std = self.actor(tf_state)

        action = mean + np.random.uniform(-self.upper_bound, self.upper_bound, size=mean.shape) * std
        action = np.clip(action, -self.upper_bound, self.upper_bound)

        return action

    def compute_advantages(self, r_batch, s_batch, ns_batch, d_batch):
        gamma = self.gamma
        lmbda = self.lmbda
        s_values = tf.squeeze(self.baseline.model(s_batch))  # input: tensor
        ns_values = tf.squeeze(self.baseline.model(ns_batch))
        returns = []
        gae = 0  # generalized advantage estimate
        for i in reversed(range(len(r_batch))):
            delta = r_batch[i] + gamma * ns_values[i] * (1 - d_batch[i]) - s_values[i]
            gae = delta + gamma * lmbda * (1 - d_batch[i]) * gae
            returns.insert(0, gae + s_values[i])

        returns = np.array(returns)
        adv = returns - s_values.numpy()  # Q - V
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)  # output: numpy array
        return adv, returns

    def compute_targets(self, r_batch, ns_batch, d_batch):
        mean = self.actor.model(ns_batch)

        target_critic = self.critic.model([ns_batch, mean])
        y = r_batch + self.gamma * (1 - d_batch) * target_critic
        return y

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

        a_loss_list = []
        c_loss_list = []
        # v_loss_list = []

        n_split = len(rewards) // self.batch_size

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)

        # Based on algorithm at https://arxiv.org/pdf/1706.00387.pdf

        # Fit baseline using collected experience and compute advantages
        # Update baseline
        advantages, returns = self.compute_advantages(rewards, states, next_states, dones)

        # If use control variate: Compute Critic-based advantages
        # and compute learning signal
        use_CV = False
        v = 0.2
        if use_CV:
            # Compute critic-based advantage
            adv_bar = self.compute_adv_bar(states, actions)
            ls = advantages - adv_bar
            b = 1
        else:
            ls = advantages
            b = v

        ls *= (1 - v)

        s_split = tf.split(states, n_split)
        a_split = tf.split(actions, n_split)
        t_split = tf.split(returns, n_split)
        adv_split = tf.split(advantages, n_split)
        ls_split = tf.split(ls, n_split)
        indexes = np.arange(n_split, dtype=int)

        # current policy
        mean, std = self.actor(states)
        pi = tfp.distributions.Normal(mean, std)

        a_loss, c_loss = 0, 0

        np.random.shuffle(indexes)
        for _ in range(self.epochs):
            s_batch, a_batch, r_batch, ns_batch, d_batch = self.buffer.sample()
            for i in indexes:
                old_pi = pi[i * self.batch_size: (i + 1) * self.batch_size]
                # Update actor
                a_loss = self.actor.train(s_split[i], ls_split[i], a_split[i], old_pi, self.critic, b, s_batch)
                a_loss_list.append(a_loss)
                # Update baseline
                v_loss = self.baseline.train(s_split[i], t_split[i])

            # Update critic
            y = self.compute_targets(r_batch, ns_batch, d_batch)
            c_loss = self.critic.train(s_batch, a_batch, y)
            c_loss_list.append(c_loss)

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

        filename = path + str(self.use_attention) + 'result_ppo_clip.txt'
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
                self.buffer.record(state, action, reward, next_state, done)

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

            with open(filename, 'a') as file:
                file.write('{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(s, s_score, mean_s_score, a_loss, c_loss))

            s += 1

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


""" main for running ipg agents """

if __name__ == "__main__":

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

    # #### Hyper-parameters
    SEASONS = 100
    success_value = 70
    lr_a = 0.0002  # 0.001
    lr_c = 0.0002  # 0.001
    epochs = 10
    training_batch = 1024  # 512
    batch_size = 64  # 128
    epsilon = 0.2  # 0.07
    gamma = 0.993  # 0.99
    lmbda = 0.7  # 0.9

    use_attention = False  # enable/disable for attention model

    env = KukaDiverseObjectEnv(renders=False,
                               isDiscrete=False,
                               maxSteps=20,
                               removeHeightHack=False)

    # PPO Agent
    agent = QPROPAgent(env, SEASONS, success_value, lr_a, lr_c, epochs, training_batch, batch_size, epsilon, gamma,
                       lmbda, use_attention)

    agent.run()  # train as PPO
