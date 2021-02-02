import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv

from kuka_ppo import PPOAgent
from collections import deque
import datetime


##################
def collect_trajectories(env, agent, max_episodes):
    ep_reward_list = []
    steps = 0
    for ep in range(max_episodes):
        state = env.reset()
        state = np.asarray(state, dtype=np.float32) / 255.0  # convert into float array
        t = 0
        ep_reward = 0
        while True:
            action = agent.policy(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.asarray(next_state, dtype=np.float32) / 255.0  # convert into float array
            agent.buffer.record(state, action, reward, next_state, done)
            ep_reward += reward
            state = next_state
            t += 1
            if done:
                ep_reward_list.append(ep_reward)
                steps += t
                break

    mean_ep_reward = np.mean(ep_reward_list)
    return steps, mean_ep_reward


# main training routine
def main(env, agent, path='./'):
    global train_summary_writer

    if agent.method == 'clip':
        outfile = open(path + 'result_clip.txt', 'w')
    else:
        outfile = open(path + 'result_klp.txt', 'w')

    #training
    total_steps = 0
    best_score = -np.inf
    for s in range(MAX_SEASONS):
        # collect trajectories
        t, s_reward = collect_trajectories(env, agent, TRG_EPISODES)

        # train the agent
        a_loss, c_loss, kld_value = agent.train(training_epochs=TRAIN_EPOCHS)
        total_steps += t
        print('Season: {}, Episodes: {} , Training Steps: {}, Mean Episodic Reward: {:.2f}'\
              .format(s, (s+1) * TRG_EPISODES, total_steps, s_reward))

        if TB_LOG:  # tensorboard logging
            with train_summary_writer.as_default():
                tf.summary.scalar('Mean reward', s_reward, step=s)
                tf.summary.scalar('Actor Loss', a_loss, step=s)
                tf.summary.scalar('Critic Loss', c_loss, step=s)
                tf.summary.scalar('KL Divergence', kld_value, step=s)
                tf.summary.scalar('Lambda', agent.actor.lam, step=s)

        #valid_score = validate(env, agent)
        if best_score < s_reward:
            best_score = s_reward
            agent.save_model(path, 'actor_weights.h5', 'critic_weights.h5')
            print('*** Season:{}, best score: {}. Model Saved ***'.format(s, best_score))

        # book keeping
        if agent.method == 'penalty':
            outfile.write('{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(s,
                                    s_reward, a_loss, c_loss, kld_value, agent.actor.lam))
        else:
            outfile.write('{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(s,
                                                        s_reward, a_loss, c_loss, kld_value))

        if best_score > 50:
            print('Problem is solved in {} seasons involving {} steps.'.format(s, total_steps))
            agent.save_model(path, 'actor_weights.h5', 'critic_weights.h5')
            break

    env.close()
    outfile.close()


# Test the model
def test(env, agent, path='./', max_eps=10):
    agent.load_model(path, 'actor_weights.h5', 'critic_weights.h5')

    ep_reward_list = []
    for ep in range(max_eps):
        obsv = env.reset()
        state = np.asarray(obsv, dtype=np.float32) / 255.0  # convert into float array
        ep_reward = 0
        t = 0
        while True:
            env.render()        # show animation
            action = agent.policy(state)
            next_obsv, reward, done, _ = agent.step(action)
            next_state = np.asarray(next_obsv, dtype=np.float32) / 255.0  # convert into float array
            ep_reward += reward
            t += 1
            state = next_state
            if done:
                ep_reward_list.append(ep_reward)
                print('Episode:{}, Reward:{}'.format(ep, ep_reward))
                break

    print('Avg Episodic Reward: ', np.mean(ep_reward_list))
    env.close()


# Validation Routine
def validate(env, agent, max_eps=20):

    ep_reward_list = []
    for ep in range(max_eps):
        obsv = env.reset()
        state = np.asarray(obsv, dtype=np.float32) / 255.0  # convert into float array
        t = 0
        ep_reward = 0
        while True:
            action = agent.policy(state)
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


####################################
### MAIN FUNCTION
################################
if __name__ == '__main__':

    ############################

    print('TFP Version:', tfp.__version__)
    print('Tensorflow version:', tf.__version__)
    print('Keras Version:', tf.keras.__version__)

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

    #####################
    # TENSORBOARD SETTINGS
    TB_LOG = True  # enable / disable tensorboard logging

    if TB_LOG:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/train/' + current_time
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    ############################

    # set random seed for reproducibility
    tf.random.set_seed(20)
    np.random.seed(20)

    ############### Hyper-parameters

    MAX_SEASONS = 5000  # total number of training seasons
    TRG_EPISODES = 100  # total number of episodes in each season
    TEST_EPISODES = 10  # total number of episodes for testing
    TRAIN_EPOCHS = 20  # training epochs in each season
    GAMMA = 0.9  # reward discount
    LR_A = 0.0002  # learning rate for actor
    LR_C = 0.0002  # learning rate for critic
    BATCH_SIZE = 128  # minimum batch size for updating PPO
    MAX_BUFFER_SIZE = 50000  # maximum buffer capacity > TRAIN_EPISODES * 200
    METHOD = 'clip'  # 'clip' or 'penalty'

    ##################
    KL_TARGET = 0.01
    LAM = 0.5
    EPSILON = 0.2

    # Kuka Environment
    env = KukaDiverseObjectEnv(renders=False,
                               isDiscrete=False,
                               maxSteps=20,
                               removeHeightHack=False)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape
    action_bound = env.action_space.high

    # create an agent
    agent = PPOAgent(state_dim, action_dim, BATCH_SIZE, MAX_BUFFER_SIZE,
                     action_bound,
                     LR_A, LR_C, GAMMA, LAM, EPSILON, KL_TARGET, METHOD)

    # training with seasons
    main(env, agent)

    # test
    # test(env, agent)
