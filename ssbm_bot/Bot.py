import keras
import melee
import numpy as np
from abc import ABC, abstractmethod
from nn_list import GRU
from melee_env.agents.util import *
from melee import enums
import MovesList
from melee_env.agents.util import ObservationSpace, ActionSpace, from_action_space
import torch
from train import SEQ_LEN
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Bot:
    def __init__(self, model, controller: melee.Controller, opponent_controller: melee.Controller):
        self.opponent_controller = opponent_controller
        self.drop_every = 180
        self.model = model
        self.controller = controller
        self.frame_counter = 0
        self.observ = ObservationSpace()
        self.action_space = ActionSpace()
        self.delay = 0
        self.pause_delay = 0
        self.firefoxing = False
        
    @from_action_space
    def act(self, gamestate: melee.GameState):
        self.controller.release_all()

        player: melee.PlayerState = gamestate.players.get(self.controller.port)
        opponent: melee.PlayerState = gamestate.players.get(self.opponent_controller.port)

        if opponent.action in MovesList.dead_list and player.on_ground:
            return

        self.frame_counter += 1

        inp, _, _, _ = self.observ(gamestate, self.controller.port, self.opponent_controller.port)
        del self.states[SEQ_LEN-1]
        self.states.insert(0, np.array(inp))
        temp = torch.tensor(np.float32([self.states])).to(device=DEVICE)
        temp2 = torch.tensor(np.float32(np.array([self.actions]))).to(device=DEVICE)
        
        a = self.model(temp, temp2)

        action = torch.argmax(a.detach().cpu())
        
        del self.actions[SEQ_LEN-2]
        zero = np.zeros(45)
        zero[action] = 1
        self.actions.insert(0, zero)
        
        return action

class Agent(ABC):
    def __init__(self):
        self.agent_type = "AI"
        self.controller = None
        self.port = None  # this is also in controller, maybe redundant?
        self.action = 0
        self.press_start = False
        self.self_observation = None
        self.current_frame = 0

    @abstractmethod
    def act(self):
        pass

class nnAgent(Agent):
    def __init__(self, obs_space, controller, opponent_controller):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = GRU().to(self.device)
        self.character = enums.Character.FOX
        self.controller = controller
        self.opponent_controller = opponent_controller

        self.action_space = ActionSpace()
        self.observation_space = obs_space
        self.action = 0
        self.states = [np.zeros(36) for _ in range(SEQ_LEN)]
        self.actions = [np.zeros(45) for _ in range(SEQ_LEN-1)]

    @from_action_space    
    def act(self, gamestate):
        inp, _, _, _ = self.observation_space(gamestate, self.controller.port, self.opponent_controller.port)
        del self.states[0]
        self.states.append(np.array(inp))
        temp = torch.tensor(np.float32([self.states])).to(device=DEVICE)
        temp2 = torch.tensor(np.float32(np.array([self.actions]))).to(device=DEVICE)
        
        a = self.model(temp, temp2)

        action = torch.argmax(a.detach().cpu())
        
        del self.actions[0]
        zero = np.zeros(45)
        zero[action] = 1
        self.actions.insert(0, zero)
        
        action_sampled = action#int(torch.multinomial(a, 1)[0])
        self.action = action_sampled
        control = self.action_space(self.action)
        control(self.controller)
        return a[0][self.action].detach().cpu()