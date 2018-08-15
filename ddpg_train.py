# Derived from keras-rl
import numpy as np
import sys

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate
from keras.optimizers import Adam

import numpy as np

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from osim.env import *
from osim.http.client import Client

from keras.optimizers import RMSprop

import argparse
import math

def get_list_dict(state_desc):
    

        # Augmented environment from the L2R challenge
    res = []
    pelvis = None

    for body_part in ["pelvis", "head","torso","toes_l","toes_r","talus_l","talus_r"]:
        if True and body_part in ["toes_r","talus_r"]:
            res += [0] * 9
            continue
        cur = []
        cur += state_desc["body_pos"][body_part][0:2]
        cur += state_desc["body_vel"][body_part][0:2]
        cur += state_desc["body_acc"][body_part][0:2]
        cur += state_desc["body_pos_rot"][body_part][2:]
        cur += state_desc["body_vel_rot"][body_part][2:]
        cur += state_desc["body_acc_rot"][body_part][2:]
        if body_part == "pelvis":
            pelvis = cur
            res += cur[1:]
        else:
            cur_upd = cur
            cur_upd[:2] = [cur[i] - pelvis[i] for i in range(2)]
            cur_upd[6:7] = [cur[i] - pelvis[i] for i in range(6,7)]
            res += cur

    for joint in ["ankle_l","ankle_r","back","hip_l","hip_r","knee_l","knee_r"]:
        res += state_desc["joint_pos"][joint]
        res += state_desc["joint_vel"][joint]
        res += state_desc["joint_acc"][joint]

    for muscle in sorted(state_desc["muscles"].keys()):
        res += [state_desc["muscles"][muscle]["activation"]]
        res += [state_desc["muscles"][muscle]["fiber_length"]]
        res += [state_desc["muscles"][muscle]["fiber_velocity"]]

    cm_pos = [state_desc["misc"]["mass_center_pos"][i] - pelvis[i] for i in range(2)]
    res = res + cm_pos + state_desc["misc"]["mass_center_vel"] + state_desc["misc"]["mass_center_acc"]

    return res

# Command line parameters
parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false', default=True)
parser.add_argument('--steps', dest='steps', action='store', default=100000, type=int)
parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
parser.add_argument('--model', dest='model', action='store', default="example.h5f")
parser.add_argument('--token', dest='token', action='store', required=False)
args = parser.parse_args()

# Load walking environment
env = ProstheticsEnv(visualize=True)
env.reset(project = True)
# env.change_model(model='3D', prosthetic=True, difficulty=2,seed=None)
# env.reset(project = True)


nb_actions = env.action_space.shape[0]

# Total number of steps in training
nallsteps = args.steps

# Create networks for DDPG
# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(32))
actor.add(Activation('relu'))
actor.add(Dense(32))
actor.add(Activation('relu'))
actor.add(Dense(32))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('sigmoid'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = concatenate([action_input, flattened_observation])
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

# Set up the agent for training
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.2, size=env.get_action_space_size())
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3,
                  delta_clip=1.)
# agent = ContinuousDQNAgent(nb_actions=env.noutput, V_model=V_model, L_model=L_model, mu_model=mu_model,
#                            memory=memory, nb_steps_warmup=1000, random_process=random_process,
#                            gamma=.99, target_model_update=0.1)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

# Training here 

agent.fit(env, nb_steps=nallsteps, visualize=False, verbose=1, nb_max_episode_steps=env.time_limit, log_interval=10000)
# After training is done, we save the final weights.
agent.save_weights(args.model, overwrite=True)

