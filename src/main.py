''' Main script for running creating agents to run algorithms'''

from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
from packaging import version
import utils

# Import agent classes
# DDPG Agent
from Kuka_DDPG import KukaDDPGAgent
# TD3 Agent
from Kuka_TD_DDPG import KukaTD3Agent
# DDPG HER Agent
from Kuka_DDPG_HER import KukaDDPGHERAgent
# TD3 HER Agent
from Kuka_TD3_HER import KukaTD3HERAgent
# PPO Agent
# from Kuka_PPO import KukaPPOAgent
from Kuka_PPO2 import KukaPPOAgent2
from Kuka_PPO3 import KukaPPOAgent


if __name__ == "__main__":

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

    # start open/AI GYM environment
    renders = False
    # env = KukaCamGymEnv(renders=renders, isDiscrete=False)
    env = KukaDiverseObjectEnv(renders=renders,
                               isDiscrete=False,
                               maxSteps=20,
                               removeHeightHack=False)
    print('Shape of Observation space: ', env.observation_space.shape)
    print('Shape of Action space: ', env.action_space.shape)
    print('Reward Range: ', env.reward_range)
    print('Action High value: ', env.action_space.high)
    print('Action Low Value: ', env.action_space.low)

    ######################
    # Hyper-parameters
    ######################
    MAX_EPISODES = 20000

    LR_A = 0.001
    LR_C = 0.002
    GAMMA = 0.99

    replacement = [
        dict(name='soft', tau=0.005),
        dict(name='hard', rep_iter_a=600, rep_iter_c=500)
    ][0]  # you can try different target replacement strategies

    MEMORY_CAPACITY = 70000
    BATCH_SIZE = 128

    upper_bound = env.action_space.high
    lower_bound = env.action_space.low
    state_size = env.observation_space.shape  # (48, 48, 3)
    action_size = env.action_space.shape  # (3,)

    noise_clip = 0.5

    print('State_size: ', state_size)
    print('Action_size: ', action_size)

    # Create a Kuka Actor-Critic Agent

    # agent = KukaDDPGAgent(state_size, action_size,
    #                       replacement, LR_A, LR_C,
    #                       BATCH_SIZE,
    #                       MEMORY_CAPACITY,
    #                       GAMMA,
    #                       upper_bound, lower_bound)

    # agent = KukaTD3Agent(state_size, action_size,
    #                      replacement, LR_A, LR_C,
    #                      BATCH_SIZE,
    #                      MEMORY_CAPACITY,
    #                      GAMMA,
    #                      upper_bound, lower_bound, noise_clip)

    _HER = False
    # _HER = True
    # agent = KukaTD3HERAgent(state_size, action_size,
    #                          replacement, LR_A, LR_C,
    #                          BATCH_SIZE,
    #                          MEMORY_CAPACITY,
    #                          GAMMA,
    #                          upper_bound, lower_bound, noise_clip)

    lmdba = 0.95
    epsilon = 0.2  # Clipping value

    agent = KukaPPOAgent(state_size, action_size,
                         LR_A, LR_C, BATCH_SIZE,
                         MEMORY_CAPACITY, lmdba, epsilon,
                         GAMMA, upper_bound, lower_bound)

    # agent = KukaPPOAgent2(state_size, action_size,
    #                      LR_A, LR_C, BATCH_SIZE,
    #                      MEMORY_CAPACITY, lmbda, epsilon,
    #                      GAMMA, upper_bound, lower_bound)

    # print("Loading model...")
    # agent.load_model('./agent_checkpoint/',
    #                  'actor_weights.h5',
    #                  'critic_weights.h5',
    #                  'agent_buffer',
    #                  'episode_number',
    #                  'ep_reward_list')
    # print("Success!")

    actor_loss, critic_loss = 0, 0
    ep_reward_list = agent.reward_list
    avg_reward_list = []
    best_score = - np.inf
    start_episode = agent.start_episode

    # For HER
    ep_exp = []
    optimisation_steps = 10
    K = 4

    for episode in range(start_episode, MAX_EPISODES):
        agent.done_ep = False
        if _HER:
            goal = env.reset()
            goal = np.asarray(goal, dtype=np.float32) / 255.0  # convert into float array
        obsv = env.reset()
        state = np.asarray(obsv, dtype=np.float32) / 255.0  # convert into float array
        episodic_reward = 0
        frames = []
        ep_frames = []
        step = 0
        ep_exp = []
        while True:
            if episode > MAX_EPISODES - 3:
                frames.append(env.render(mode='rgb_array'))

            if _HER:
                tf_goal = tf.expand_dims(tf.convert_to_tensor(goal), 0)
                tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
                action = agent.policy(tf_state, tf_goal)
                next_obsv, reward, done, info = env.step(action)
                next_state = np.asarray(next_obsv, dtype=np.float32) / 255.0  # convert into float array
                tb_img = np.reshape(next_state, (-1,) + state_size)  # for tensorboard

                with train_summary_writer.as_default():
                    tf.summary.image("Training Image", tb_img, step=episode)
                    tf.summary.histogram("action_vector", action, step=step)

                episodic_reward += reward

                agent.buffer.record(state, action, reward, next_state, done, goal)
                ep_exp.append([state, action, reward, next_state, done, goal])

                if done:
                    agent.done_ep = True

            else:
                # convert the numpy array state into a tensor of size (1, 48, 48)
                tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)

                # take an action as per the policy
                action = agent.policy(tf_state)

                # obtain next state and rewards
                next_obsv, reward, done, info = env.step(action)
                next_state = np.asarray(next_obsv, dtype=np.float32) / 255.0  # convert into float array

                tb_img = np.reshape(next_state, (-1,) + state_size)  # for tensorboard

                with train_summary_writer.as_default():
                    tf.summary.image("Training Image", tb_img, step=episode)
                    tf.summary.histogram("action_vector", action, step=step)

                episodic_reward += reward

                # store experience
                agent.buffer.record(state, action, reward, next_state, done)

                if done:
                    agent.done_ep = True
                # train the network
                actor_loss, critic_loss = agent.experience_replay()

                # update the target model
                agent.update_targets()

            if actor_loss != 0 and critic_loss != 0:
                with train_summary_writer.as_default():
                    tf.summary.scalar('actor_loss', actor_loss, step=episode)
                    tf.summary.scalar('critic_loss', critic_loss, step=episode)

            state = next_state
            step += 1

            if done:
                # Checkpoint agent every 100 episodes
                if episode % 100 == 0:
                    print("Saving agent model at episode {}...".format(episode))
                    agent.save_model('./agent_checkpoint/',
                                     'actor_weights.h5',
                                     'critic_weights.h5',
                                     'agent_buffer',
                                     'episode_number',
                                     episode,
                                     'ep_reward_list',
                                     ep_reward_list)
                    print("Agent model saved!")
                if episodic_reward > best_score:
                    best_score = episodic_reward
                break

        # End of episode
        ep_reward_list.append(episodic_reward)
        avg_reward = np.mean(ep_reward_list[-100:])
        print("Episode {}: Exp Size = {}, Reward = {}, AvgReward = {} "
              .format(episode, agent.buffer.size, reward, avg_reward))
        avg_reward_list.append(avg_reward)

        with train_summary_writer.as_default():
            tf.summary.scalar('avg_reward', avg_reward, step=episode)

        if _HER:
            # For each step of the finished episode
            for t in range(len(ep_exp)):
                # Remove k loop for final state strategy
                for k in range(K):
                    future = np.random.randint(t, len(ep_exp))  # Future strategy
                    goal = ep_exp[future][3]
                    # goal = ep_exp[-1][3]  # Final state strategy
                    state = ep_exp[t][0]
                    action = ep_exp[t][1]
                    next_state = ep_exp[t][3]
                    done = np.array_equal(next_state, goal)
                    reward = 1 if done else 0
                    agent.buffer.record(state, action, reward, next_state, done, goal)

            for i in range(optimisation_steps):
                actor_loss, critic_loss = agent.experience_replay()
                agent.update_targets()

    env.close()

    # plot
    plt.plot(avg_reward_list)
    plt.xlabel('Episodes')
    plt.ylabel('Avg episodic reward')
    plt.grid()
    plt.savefig('./kuka_ddpg_tf2.png')
    plt.show()
    # save animation as GIF
    # save_frames_as_gif(frames)
