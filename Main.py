import pickle

import gym
import numpy as np
import sys
import re
from deepq import deepq
from util.env_creator import get_env_type, build_env
from util.vec_env import VecEnv
import multiprocessing
from collections import defaultdict



def testGym():
    evn_name = "Pong-v0"
    env = gym.make(evn_name)
    env.reset()
    for _ in range(400):
        action = env.action_space.sample()
        env.step(action)
        env.render()
    env.close()



def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved

def train():
    env_type, env_id = get_env_type("CartPole-v0")
    env = build_env(env_id, 'deepq')
    model = deepq.learn(
        env,
        network='mlp',
        lr=1e-3,
        total_timesteps=100000,
        buffer_size=50000,
        checkpoint_freq=100,
        checkpoint_path="cartpole_model",
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        save_path="model/cartpole_model",
        callback=callback
    )

def play():
    env_type, env_id = get_env_type("CartPole-v0")
    env = build_env(env_id, 'deepq')
    model = deepq.learn(
        env,
        network='mlp',
        lr=1e-3,
        total_timesteps=100000,
        buffer_size=50000,
        checkpoint_freq=100,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        load_path="model/cartpole_model",
        callback=callback
    )
    obs, done = env.reset(), False
    if not isinstance(env, VecEnv):
        obs = np.expand_dims(np.array(obs), axis=0)
    state = model.initial_state if hasattr(model, 'initial_state') else None

    episode_rew = np.zeros(env.num_envs) if isinstance(env, VecEnv) else np.zeros(1)
    while True:
        if state is not None:
            actions, _, state, _ = model.step(obs)
        else:
            actions, _, _, _ = model.step(obs)

        obs, rew, done, _ = env.step(actions.numpy())
        if not isinstance(env, VecEnv):
            obs = np.expand_dims(np.array(obs), axis=0)
        episode_rew += rew
        env.render()
        done_any = done.any() if isinstance(done, np.ndarray) else done
        if done_any:
            for i in np.nonzero(done)[0]:
                print('episode_rew={}'.format(episode_rew[i]))
                episode_rew[i] = 0

    env.close()

    #act.save("cartpole_model.pkl")


if __name__ == "__main__":
    #train()
    play()

