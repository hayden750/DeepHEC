# -*- coding: utf-8 -*-
"""DPPG_Kuka_HER.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18h_11EWSRNFn5CsWTc2YzA8KJsuGN6mE
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import datetime
from packaging import version

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
# device_name = tf.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#     raise SystemError('GPU device not found')
# print('Found GPU at: {}'.format(device_name))

#####################
# TENSORBOARD SETTINGS
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/train/' + current_time
graph_log_dir = 'logs/func/' + current_time
train_summary_writer = tf.summary.create_file_writer(train_log_dir)


# graph_summary_writer = tf.summary.create_file_writer(graph_log_dir)
# tf.summary.trace_on(graph=True, profiler=True)
######################


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
        return model

    def __call__(self, state):
        return self.model(state)

import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv

renders = False
env = KukaDiverseObjectEnv(renders=renders,
                           isDiscrete=False,
                           maxSteps=20,
                           removeHeightHack=False)
print('Shape of Observation space: ', env.observation_space.shape)
print('Shape of Action space: ', env.action_space.shape)
print('Reward Range: ', env.reward_range)
print('Action High value: ', env.action_space.high)
print('Action Low Value: ', env.action_space.low)

num_states = env.observation_space.shape
num_actions = env.action_space.shape

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

class OUActionNoise:
  def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
    self.theta = theta
    self.mean = mean
    self.std_dev = std_deviation
    self.dt = dt
    self.x_initial = x_initial
    self.reset()

  def __call__(self):
    x = (
        self.x_prev
         + self.theta * (self.mean - self.x_prev) * self.dt
         + self.std_dev * np.sqrt(self.dt) * np.random.normal(
             size=self.mean.shape)
    )

    # Store x into x_prev
    # Makes next noise dependent on current one
    self.x_prev = x
    return x
  
  def reset(self):
    if self.x_initial is not None:
      self.x_prev = self.x_initial
    else:
      self.x_prev = np.zeros_like(self.mean)

class Buffer:
  def __init__(self, buffer_capacity=100000, batch_size=64):
    # Number of "experiences" to store at max
    self.buffer_capacity = buffer_capacity
    self.batch_size = batch_size
    self.buffer_counter = 0

    # Instead of list of tuples as the exp.relay concept go
    # We use different np.arrays for each tuple element
    self.state_buffer = deque(maxlen=self.buffer_capacity)
    self.action_buffer = deque(maxlen=self.buffer_capacity)
    self.reward_buffer = deque(maxlen=self.buffer_capacity)
    self.next_state_buffer = deque(maxlen=self.buffer_capacity)
    self.goal_buffer = deque(maxlen=self.buffer_capacity)

  # Takes (s,a,r,s') observation tuple as input
  def record(self, obs_tuple):
    # Set index to zero if buffer_capacity is exceeded,
    # replacing old records
    index = self.buffer_counter % self.buffer_capacity

    # self.state_buffer[index] = obs_tuple[0]
    # self.action_buffer[index] = obs_tuple[1]
    # self.reward_buffer[index] = obs_tuple[2]
    # self.next_state_buffer[index] = obs_tuple[3]
    # self.goal_buffer[index] = obs_tuple[4]

    self.state_buffer.append(obs_tuple[0])
    self.action_buffer.append(obs_tuple[1])
    self.reward_buffer.append(obs_tuple[2])
    self.next_state_buffer.append(obs_tuple[3])
    self.goal_buffer.append(obs_tuple[4])

    self.buffer_counter += 1

  @tf.function
  def update(
      self, state_batch, action_batch, reward_batch, next_state_batch, goal_batch
  ):
    # Training and updating Actor & Critic networks
    state_batch = tf.cast(state_batch, float)
    action_batch = tf.cast(action_batch, float)
    reward_batch = tf.cast(reward_batch, float)
    next_state_batch = tf.cast(next_state_batch, float)
    goal_batch = tf.cast(goal_batch, float)
    with tf.GradientTape() as tape:
      target_actions = target_actor([next_state_batch, goal_batch], training=True)
      y = reward_batch + gamma * target_critic(
          [next_state_batch, goal_batch, target_actions], training=True
      )
      critic_value = critic_model([state_batch, goal_batch, action_batch], training=True)
      critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

    critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
    critic_optimizer.apply_gradients(
        zip(critic_grad, critic_model.trainable_variables)
    )

    with tf.GradientTape() as tape:
      actions = actor_model([state_batch, goal_batch], training=True)
      critic_value = critic_model([state_batch, goal_batch, actions], training=True)
      # Used '-value' as we want to maximize the value given
      # by the critic for our actions
      actor_loss = -tf.math.reduce_mean(critic_value)
    
    actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
    actor_optimizer.apply_gradients(
        zip(actor_grad, actor_model.trainable_variables)
    )

  # We compute the loss and update parameters
  def learn(self):
    # Get sampling range
    record_range = min(self.buffer_counter, self.buffer_capacity)
    # Randomly sample indices
    batch_indices = np.random.choice(record_range, self.batch_size)

    # Convert to tensors
    state_batch = tf.convert_to_tensor(np.asarray(self.state_buffer)[batch_indices])
    action_batch = tf.convert_to_tensor(np.asarray(self.action_buffer)[batch_indices])
    reward_batch = tf.convert_to_tensor(np.asarray(self.reward_buffer)[batch_indices])
    reward_batch = tf.cast(reward_batch, dtype=tf.float32)
    next_state_batch = tf.convert_to_tensor(
        np.asarray(self.next_state_buffer)[batch_indices])
    goal_batch = tf.convert_to_tensor(np.asarray(self.goal_buffer)[batch_indices])
    
    self.update(state_batch, action_batch, reward_batch, next_state_batch, goal_batch)
  
# This updates target parameters slowly
# Based on rate 'tau', which is much less than one
@tf.function
def update_target(target_weights, weights, tau):
  for (a, b) in zip(target_weights, weights):
    a.assign(b * tau + a * (1 - tau))

def get_actor():
  # Initialise weights between -3e-3 and 3-e3
  last_init = tf.random_uniform_initializer(minval=-0.03, maxval=0.03)

  # State Input
  state_input = layers.Input(shape=num_states)

  feature = feature_model(state_input)
  state_out = layers.Dense(32, activation="relu")(feature)
  state_out = layers.Dense(32, activation="relu")(state_out)

  # Goal Input
  goal_input = layers.Input(shape=num_states)
  goal_out = layers.Dense(32, activation="relu")(feature)
  goal_out = layers.Dense(32, activation="relu")(goal_out)

  # Both are passed through separate layer before concatenating
  concat = layers.Concatenate()([state_out, goal_out])

  out = layers.Dense(128, activation="relu")(concat)
  out = layers.Dense(64, activation="relu")(out)

  net_out = layers.Dense(num_actions[0], activation='tanh',
                          kernel_initializer=last_init)(out)

  net_out = net_out * upper_bound
  model = tf.keras.Model(inputs=[state_input, goal_input], outputs=net_out)
  model.summary()
  
  return model

def get_critic():

  # State Input
  state_input = layers.Input(shape=num_states)

  feature = feature_model(state_input)
  state_out = layers.Dense(32, activation="relu")(feature)
  state_out = layers.Dense(32, activation="relu")(state_out)

  # Goal Input
  goal_input = layers.Input(shape=num_states)
  goal_out = layers.Dense(32, activation="relu")(feature)
  goal_out = layers.Dense(32, activation="relu")(goal_out)

  # Action Input
  action_input = layers.Input(shape=num_actions)
  action_out = layers.Dense(32, activation="relu")(action_input)

  # Both are passed through separate layer before concatenating
  # concat = layers.Concatenate()([state_out, goal_out, action_out])
  concat = layers.Concatenate()([state_out, goal_out])
  concat = layers.Concatenate()([concat, action_out])

  out = layers.Dense(128, activation="relu")(concat)
  out = layers.Dense(64, activation="relu")(out)  # leakyRelu
  # out = layers.Dense(64, activation=tf.keras.layers.LeakyReLU())(out)  # leakyRelu

  net_out = layers.Dense(1)(out)

  # Outputs single value for give state-action
  model = tf.keras.Model(inputs=[state_input, goal_input, action_input], outputs=net_out)
  model.summary()

  return model

def policy(state, goal, noise_object):
  sampled_actions = tf.squeeze(actor_model([state, goal]))
  noise = noise_object()
  # Adding noise to action
  sampled_actions = sampled_actions.numpy() + noise

  # We make sure action is within bounds
  legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

  return np.squeeze(legal_action)

def test_model(n, test_step, prev_succ):
  test_reward = []
  for i in range(n):
    goal = env.reset()
    prev_state = env.reset()
    episodic_reward = 0
    step = 0
    while True:

      tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state, dtype=float), 0)
      tf_goal = tf.expand_dims(tf.convert_to_tensor(goal, dtype=float), 0)
      action = policy(tf_prev_state, tf_goal, ou_noise)
      state, reward, done, info = env.step(action)

      episodic_reward += reward

      if done:
          break

      prev_state = state
      step += 1

    test_reward.append(episodic_reward)

  successes = np.sum(test_reward)
  print("Test Run {}: Success Rate: {}/{}: Previous Test Success: {}/{}".format(test_step, successes, n, prev_succ, n))
  return successes

# Training Hyperparameters
std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), 
                         std_deviation=float(std_dev) * np.ones(1))


feature_model = FeatureNetwork(num_states)

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 50000
gamma = 0.99  # Discount factor for future rewards
tau = 0.005  # Used to update target networks

buffer = Buffer(50000, 64)

# Main
ep_reward_list = []
avg_reward_list = []
best_avg = -np.inf
test_step = -1
test_run_eps = 10
test_success = None

for ep in range(total_episodes):
  goal = env.reset()
  prev_state = env.reset()
  episodic_reward = 0
  ep_exp = []
  step = 0
  while True:

    tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state, dtype=float), 0)
    tf_goal = tf.expand_dims(tf.convert_to_tensor(goal, dtype=float), 0)
    action = policy(tf_prev_state, tf_goal, ou_noise)
    state, reward, done, info = env.step(action)
    

    buffer.record((prev_state, action, reward, state, goal))
    ep_exp.append((prev_state, action, reward, state, goal))
    episodic_reward += reward
    
    if done:
      break
    
    prev_state = state
    step += 1

  # End of episode

  ep_reward_list.append(episodic_reward)

  # Mean of last 50 episodes
  avg_reward = np.mean(ep_reward_list[-50:])
  success = np.sum(ep_reward_list[-50:])
  # Every 50 episodes see if best mean has improved
  if ep % 50 == 0:
    if avg_reward > best_avg and ep >= 50:
      print("Best averaged updated: {:.3f} --> {:.3f}".format(best_avg, avg_reward))
      best_avg = avg_reward
  # Perform test run every 200 episodes
  if ep % 200 == 0 and ep > 0:
    print("Performing test run...")
    test_step += 1
    test_success = test_model(test_run_eps, test_step, test_success)
  print("Episode: {}: Reward: {}, Success Rate: {}/50, Avg Reward: {}".format(ep, reward, success, avg_reward))
  avg_reward_list.append(avg_reward)

  # For each step of the finished episode
  for t in range(len(ep_exp)):
  # Remove k loop for final state strategy
    #for k in range(4):
      #future = np.random.randint(t, len(ep_exp))  # Future strategy
      #goal = ep_exp[future][3]
    goal = ep_exp[-1][3]  # Final state strategy
    state = ep_exp[t][0]
    action = ep_exp[t][1]
    next_state = ep_exp[t][3]
    done = np.array_equal(next_state, goal)
    reward = 1 if done else 0
    buffer.record((state, action, reward, next_state, goal))

  for i in range(10):
    buffer.learn()
  update_target(target_actor.variables, actor_model.variables, tau)
  update_target(target_critic.variables, critic_model.variables, tau)


# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()
