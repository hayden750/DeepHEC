import tensorflow as tf
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from ppo_agent import PPOAgent
from ipg_agent import IPGAgent
from ipg_her_agent import IPGHERAgent
import numpy as np
import matplotlib.pyplot as plt
from grad_cam import make_gradcam_heatmap, save_and_display_gradcam, grad_cam2


if __name__ == '__main__':

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

    # start open/AI GYM environment
    env = KukaDiverseObjectEnv(renders=False,   # True for testing
                               isDiscrete=False,
                               maxSteps=20,
                               removeHeightHack=False)

    upper_bound = env.action_space.high
    state_size = env.observation_space.shape  # (48, 48, 3)
    action_size = env.action_space.shape[0]  # (3,)

    # Create a Kuka Actor-Critic Agent
    agent = IPGAgent(env,
                     SEASONS=1,
                     batch_size=128,
                     epochs=10,
                     use_attention=True,
                     success_value=70,
                     training_batch=1024,
                     lr_a=2e-4,
                     lr_c=2e-4,
                     gamma=0.993,
                     lmbda=0.7,
                     epsilon=0.2,
                     use_mujo=False)

    # agent = IPGHERAgent(env,
    #                     SEASONS=1,
    #                     batch_size=128,
    #                     epochs=10,
    #                     use_attention=True,
    #                     success_value=70,
    #                     training_batch=1024,
    #                     lr_a=2e-4,
    #                     lr_c=2e-4,
    #                     gamma=0.993,
    #                     lmbda=0.7,
    #                     epsilon=0.2,
    #                     use_mujo=False)

    # load model weights
    agent.load_model('./', 'actor_weights.h5', 'critic_weights.h5', 'baseline_weights.h5')

    print(agent.feature.model.summary())

    for e in range(5):
        goal = env.reset()
        goal = np.asarray(goal, dtype=np.float32) / 255.0  # convert into float array
        obsv = env.reset()
        done = False
        t = 0
        while not done:
            state = np.asarray(obsv, dtype=np.float32) / 255.0  # convert into float array
            action = agent.policy(state)
            next_obsv, reward, done, _ = env.step(action)

            # generate heatmap
            heatmap = make_gradcam_heatmap(state, agent.feature.model, 'max_pooling2d')
            #heatmap = grad_cam2(state, agent.feature.model, agent.actor.model, 'attention_2', 'feature_net')
            new_img_array = save_and_display_gradcam(state, heatmap, cam_path='./gradcam/cam_l2_{}_{}.jpg'.format(e, t))
            fig, axes = plt.subplots(2, 2)
            axes[0][0].imshow(obsv)
            axes[0][0].axis('off')
            #axes[0][0].set_title('Original')
            axes[0][1].matshow(heatmap)
            axes[0][1].axis('off')
            #axes[0][1].set_title('Heatmap')
            axes[1][0].imshow(new_img_array)
            axes[1][0].axis('off')
            #axes[1][0].set_title('Superimposed')
            axes[1][1].axis('off')
            fig.tight_layout()
            plt.savefig('./gradcam/comb_l2_{}_{}.jpg'.format(e, t))
            plt.show()

            obsv = next_obsv
            t += 1

