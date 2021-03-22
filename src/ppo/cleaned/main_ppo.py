""" main for running ppo agents """

# Imports
import tensorflow as tf
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from packaging import version

# Local imports
from ppo_agent import PPOAgent
from ppo_esil import PPOESILAgent

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
    SEASONS = 1000
    success_value = 100
    lr_a = 0.0002  # 0.001
    lr_c = 0.0002  # 0.001
    epochs = 10
    training_batch = 1024  # 512
    batch_size = 128
    epsilon = 0.07  # 0.2
    gamma = 0.993  # 0.99
    lmbda = 0.7  # 0.9

    use_attention = True  # enable/disable for attention model

    env = KukaDiverseObjectEnv(renders=False,
                               isDiscrete=False,
                               maxSteps=20,
                               removeHeightHack=False)

    # PPO Agent
    agent = PPOAgent(env, SEASONS, success_value, lr_a, lr_c, epochs, training_batch, batch_size, epsilon, gamma,
                     lmbda, use_attention)
    # ESIL Agent
    # agent = PPOESILAgent(env, SEASONS, success_value, lr_a, lr_c, epochs, training_batch, batch_size, epsilon, gamma,
    #                      lmbda, use_attention)

    agent.run()  # train as PPO
