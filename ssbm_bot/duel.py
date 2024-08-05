from melee import enums
from melee_env.env import MeleeEnv
from melee_env.agents.basic import *
from melee_env.agents.util import *

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from nn_list import *
import math
import random
import argparse
import time
import os

from Bot import nnAgent

observation_space = ObservationSpace()
action_space = ActionSpace()

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = './models/1711315097.0012596_0.pt'

agent = nnAgent(observation_space)
agent.net.load_state_dict(torch.load(model_path))

players = [agent, CPU(enums.Character.FOX, 9)] #CPU(enums.Character.KIRBY, 5)]
env = MeleeEnv("ssbm.iso", players, fast_forward=True, ai_starts_game=True)
env.start()

state, done = env.setup(enums.Stage.FINAL_DESTINATION)
while not done:
    for i in range(len(players)):
        players[i].act(state)
    obs, _, _, _ = agent.observation_space(state)
    state, done = env.step()