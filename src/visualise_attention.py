import tensorflow as tf
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from ppo_agent import PPOAgent
from ipg_agent import IPGAgent
from ipg_her_agent import IPGHERAgent
import numpy as np
import matplotlib.pyplot as plt
from grad_cam import make_gradcam_heatmap, save_and_display_gradcam, grad_cam2


if __name__ == '__main__':

    # start open/AI GYM environment
    env = KukaDiverseObjectEnv(renders=False,   # True for testing
                               isDiscrete=False,
                               maxSteps=20,
                               removeHeightHack=False)

    upper_bound = env.action_space.high
    state_size = env.observation_space.shape  # (48, 48, 3)
    action_size = env.action_space.shape  # (3,)

    # Create a Kuka Actor-Critic Agent
    agent = IPGAgent(state_size, action_size,
                     batch_size=128,
                     epochs=10,
                     use_attention=True,
                     success_value=70,
                     training_batch=1024,
                     lr_a=2e-4,
                     lr_c=2e-4,
                     gamma=0.993,
                     lmbda=0.7,
                     epsilon=0.2)

    # agent = IPGHERAgent(state_size, action_size,
    #                     batch_size=128,
    #                     epochs=10,
    #                     use_attention=True,
    #                     success_value=70,
    #                     training_batch=1024,
    #                     lr_a=2e-4,
    #                     lr_c=2e-4,
    #                     gamma=0.993,
    #                     lmbda=0.7,
    #                     epsilon=0.2)

    # load model weights
    agent.load_model('./', 'actor_weights.h5', 'critic_weights.h5', 'baseline_weights.h5')

    print(agent.feature.model.summary())

    for e in range(5):
        obsv = env.reset()
        done = False
        t = 0
        while not done:
            state = np.asarray(obsv, dtype=np.float32) / 255.0  # convert into float array
            action = agent.policy(state)
            next_obsv, reward, done, _ = env.step(action)

            # generate heatmap
            #heatmap = make_gradcam_heatmap(state, agent.feature.model, 'multiply_2')
            heatmap = grad_cam2(state, agent.feature.model, agent.actor.model, 'attention_2', 'feature_net')
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

