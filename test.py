import gym
from gym.envs.box2d import CarRacing

from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

if __name__=='__main__':
    env = lambda :  CarRacing(
        grayscale=1,
        show_info_panel=0,
        discretize_actions="hard",
        frames_per_state=4,
        num_lanes=1,
        num_tracks=1,
        )

    #env = getattr(environments, env)
    env = DummyVecEnv([env])

    model = PPO2.load('car_racing_weights.pkl')

    model.set_env(env)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
