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
    def __init__(self, benchmarks, is_from_bc=False):
        if is_from_bc:
            benchmark = make_benchmark(benchmarks[0])
            self.env = gym.make("llvm-autophase-ic-v0", benchmark=benchmark)
        else:
            self.env = gym.make("llvm-autophase-ic-v0", benchmark="cbench-v1/{0}".format(benchmarks[0]))
        self.benchmarks = benchmarks
        self.action_space = self.env.action_space


    def close(self):
        self.env.close()
        super().close()

    def switch_benchmark(self):
        idx = random.randint(0, -1 + len(self.benchmarks))
        print("Switched to {0}".format(self.benchmarks[idx]))
        self.env.close()
        if self.is_from_bc:
            benchmark = make_benchmark(self.benchmarks[idx])
            self.env = gym.make("llvm-autophase-ic-v0", benchmark=benchmark)
        else:
            self.env = gym.make("llvm-autophase-ic-v0", benchmark="cbench-v1/{0}".format(self.benchmarks[idx]))

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        llvm_ir =self.env.observation["Ir"]
        autophase = self.env.observation["Autophase"]
        return PassformerObservation(llvm_ir=llvm_ir, autophase=autophase), reward, done, info

    def multistep(self, actions: List[int]):
        observation, reward, done, info = self.env.multistep(actions)
        llvm_ir =self.env.observation["Ir"]
        autophase = self.env.observation["Autophase"]
        return PassformerObservation(llvm_ir=llvm_ir, autophase=autophase), reward, done, info
    
    def multistep_by_action_flags(self, actions: str):
        actions = [self.action_space.flags.index(action) for action in actions.split(" ") if action]
        return self.multistep(actions)

    def reset(self, bc=None):
        if bc is not None:
            benchmark = make_benchmark(bc)
            self.env.reset(benchmark=benchmark)
        else:
            self.env.reset()
        llvm_ir = self.env.observation["Ir"]
        autophase = self.env.observation["Autophase"]
        return PassformerObservation(llvm_ir=llvm_ir, autophase=autophase)

if __name__ == "__main__":
    # benchmarks = ["qsort"]
    # # env = llvm_wrapper(benchmarks)
    # env = llvm_wrapper(benchmarks, max_episode_steps=200, steps_in_observation=True)
    # print("reset:", env.reset())
    # # print("observation_space:", env.observation_space)
    # print("step:", env.step(env.action_space.sample()))
    # print("close:", env.close())

    benchmarks = ["/home/xucong24/Compiler/datasets/cbench-v1/benchmark_cbench-v1_adpcm.bc"]
    # env = llvm_wrapper(benchmarks)
    env = llvm_wrapper(benchmarks, is_from_bc=True)
    env.reset()
    # print("observation_space:", env.observation_space)
    # print("step:", env.multistep([env.action_space.sample() for _ in range(10)]))
    observation, reward, done, info = env.multistep_by_action_flags("-add-discriminators -adce")
    print(reward)
    print(env.env.commandline())
    env.close()
    
