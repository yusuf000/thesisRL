import gym
from deepq import deepq


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


def main():
    env = gym.make("CartPole-v0")
    act = deepq.learn(
        env,
        network='mlp',
        lr=1e-3,
        total_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        save_path="cartpole_model.pkl",
        callback=callback
    )
    #pickle.dump(act, open("model", 'wb'))
    #act.save("cartpole_model.pkl")


if __name__ == "__main__":
    main()
