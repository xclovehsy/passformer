import gym
import numpy as np
import random
from compiler_gym.envs.llvm import make_benchmark

from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class PassformerObservation:
    llvm_ir: str
    autophase: List[int]

class llvm_wrapper(gym.Env):
    """
    benchmarks: Names of programs in cbench-v1 which are to be cycled through in training
    max_episode_steps: If specified: Number of maximum steps an episode can last up to. Otherwise no episode limit
    patience: If specified:  Number of consecutive steps with reward 0 for episode to terminate.
    allowed_actions: If specified: Subset of action space given as integers.
    """

    def __init__(self, benchmarks, max_episode_steps=None, steps_in_observation=False, patience=None,
                 allowed_actions=None):
        self.env = gym.make("llvm-autophase-ic-v0", benchmark="cbench-v1/{0}".format(benchmarks[0]))
        # self.env = gym.make("llvm-v0", benchmark="cbench-v1/{0}".format(benchmarks[0]))
        self.benchmarks = benchmarks

        # patience
        self.patience = patience
        self.fifo = []

        # Observation space
        self.limited_time = max_episode_steps is not None
        if self.limited_time:
            self.max_steps = max_episode_steps
            self.elapsed_steps = 0
        
        # Observation space is the observation space of the environment
        # self.observation_space = self.env.observation_space

        # Action space
        self.action_space = self.env.action_space


    def close(self):
        self.env.close()
        super().close()

    def switch_benchmark(self):
        idx = random.randint(0, -1 + len(self.benchmarks))
        print("Switched to {0}".format(self.benchmarks[idx]))
        self.env.close()
        self.env = gym.make("llvm-autophase-ic-v0", benchmark="cbench-v1/{0}".format(self.benchmarks[idx]))

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        if self.patience is not None:
            self.fifo.append(reward)
            while len(self.fifo) > self.patience:
                del self.fifo[0]
            all_zero = True
            for x in self.fifo:
                if x != 0:
                    all_zero = False
                    break
            if all_zero and len(self.fifo) >= self.patience:
                done = True

        if self.limited_time:
            self.elapsed_steps += 1
            if self.elapsed_steps >= self.max_steps:
                done = True

        llvm_ir =self.env.observation["Ir"]
        autophase = self.env.observation["Autophase"]
        
        return PassformerObservation(llvm_ir=llvm_ir, autophase=autophase), reward, done, info

    def reset(self):
        self.fifo = []
        if self.limited_time:
            self.elapsed_steps = 0

        observation = self.env.reset()
        llvm_ir = self.env.observation["Ir"]
        autophase = self.env.observation["Autophase"]

        return PassformerObservation(llvm_ir=llvm_ir, autophase=autophase)

if __name__ == "__main__":
    benchmarks = ["qsort"]
    # env = llvm_wrapper(benchmarks)
    env = llvm_wrapper(benchmarks, max_episode_steps=200, steps_in_observation=True)
    print("reset:", env.reset())
    # print("observation_space:", env.observation_space)
    print("step:", env.step(env.action_space.sample()))
    print("close:", env.close())
