    #!/usr/bin/python3
import copy
from collections import deque

import melee

import os
import sys
import json

from tqdm import tqdm
import time
import numpy as np
import pickle

import Args
from DataHandler_meleenv import get_ports, controller_states_different, generate_output
import MovesList

from melee_env.agents.util import ObservationSpace, ActionSpace

args = Args.get_args()

def get_size(obj):
    """객체의 크기를 바이트 단위로 반환합니다."""
    return sys.getsizeof(pickle.dumps(obj))

def load_data(replay_paths: str, player_character: melee.Character, opponent_character: melee.Character, itr):
    max_size = 3 * 1024 * 1024 * 1024
    
    X_player = []
    Y_player = []
    R_player = []

    X_opponent = []
    Y_opponent = []
    R_opponent = []
    
    player_replay_id = []
    opponent_replay_id = []
    
    for idx, path in tqdm(enumerate(replay_paths)):
        observ = ObservationSpace()
        console = melee.Console(system="file",
                                allow_old_version=True,
                                path=path)
        try:
            console.connect()
        except:
            console.stop()
            print('console failed to connect', path, time.time())
            continue

        gamestate: melee.GameState = console.step()
        player_port, opponent_port = get_ports(gamestate, player_character=player_character,
                                               opponent_character=opponent_character)
        if player_port == -1:
            print('bad port', path, gamestate.players.keys(), time.time())

            continue

        player: melee.PlayerState = gamestate.players.get(player_port)
        opponent: melee.PlayerState = gamestate.players.get(opponent_port)
        
        player_prev_game_last_idx = len(X_player)
        oppon_prev_game_last_idx = len(X_opponent)
        X_player.append([])
        Y_player.append([])
        R_player.append([])

        X_opponent.append([])
        Y_opponent.append([])
        R_opponent.append([])
        while True:
            try:
                gamestate: melee.GameState = console.step()
            except:
                break
            if gamestate is None or gamestate.stage is None:
                break
            
            player: melee.PlayerState = gamestate.players.get(player_port)
            opponent: melee.PlayerState = gamestate.players.get(opponent_port)
            if player is None or opponent is None:
                break

            if player.action in MovesList.dead_list:
                continue

            # player
            inp, r, _, _ = observ(gamestate, player_port, opponent_port)
            action = generate_output(player)
            # print(action) if action >= 30 else None
            # out = np.zeros(45)
            # out[action] = 1
            X_player[idx].append(inp)
            Y_player[idx].append(action)
            R_player[idx].append(r)

            #opponent
            inp, r, _, _ = observ(gamestate, opponent_port, player_port)
            action = generate_output(opponent)
            # out = np.zeros(45)
            # out[action] = 1
            # X_opponent.append(inp)
            # Y_opponent.append(action)
            # R_opponent.append(r)
            
        
        # player_replay_id.extend([(idx%256) for _ in range(len(X_player)-player_prev_game_last_idx)])
        # opponent_replay_id.extend([(idx%256) for _ in range(len(X_opponent)-oppon_prev_game_last_idx)])
        # if (len(player_replay_id) != len(X_player) or len(player_replay_id) != len(Y_player)):
        #     import pdb;pdb.set_trace()
        # if (len(opponent_replay_id) != len(X_opponent) or len(opponent_replay_id) != len(Y_opponent)):
        #     import pdb;pdb.set_trace()
        
        # if idx % 300 == 299:
        #     dump={'X': np.array(X_player), 'Y': np.array(Y_player), 'R': np.array(R_player), "ID": player_replay_id}
        #     size = get_size(dump)
        #     print(size)
        #     if size > max_size:
        #         X_player = np.array(X_player)
        #         Y_player = np.array(Y_player)
        #         R_player = np.array(R_player)
        #         X_opponent = np.array(X_opponent)
        #         Y_opponent = np.array(Y_opponent)
        #         R_opponent = np.array(R_opponent)
        #         del dump
                
        #         return X_player, Y_player, R_player, X_opponent, Y_opponent, R_opponent, player_replay_id, opponent_replay_id, replay_paths[idx+1:]
        #     del dump
                
        X_player[idx] = np.array(X_player[idx])
        Y_player[idx] = np.array(Y_player[idx])
        R_player[idx] = np.array(R_player[idx])
        # X_opponent[idx] = np.array(X_opponent[idx])
        # Y_opponent[idx] = np.array(Y_opponent[idx])
        # R_opponent[idx] = np.array(R_opponent[idx])
    
    return X_player, Y_player, R_player, X_opponent, Y_opponent, R_opponent, player_replay_id, opponent_replay_id, []

def process_replays(replay_paths: list, c1: melee.Character, c2: melee.Character, s: melee.Stage, iteration : int, section=0):
    #print(f'Data/{c1.name}_{c2.name}_on_{s.name}_data.pkl')
    #print(f'Data/{c2.name}_{c1.name}_on_{s.name}_data.pkl')

    print(len(replay_paths))

    Xp, Yp, Rp, Xo, Yo, Ro, IDp, IDo, _ = load_data(replay_paths, c1, c2, iteration)

    for idx, replay_path in enumerate(replay_paths):    
        replay_name = replay_path.split("\\")[-1][:-4]
        data_file_path = f'./Data/{replay_name}_data.pkl' # 수정 필
        print(len(Xp[idx]))
        print(Xp[idx])
        with open(data_file_path, 'wb') as file:
            pickle.dump({'X': Xp[idx], 'Y': Yp[idx], 'R': Rp[idx]}, file)
        print("saved at " + data_file_path)
    # data_file_path = f'./Data/{c2.name}_{c1.name}_on_{s.name}_{section}_data.pkl' # 수정 필
    # with open(data_file_path, 'wb') as file:
    #     pickle.dump({'X': Xo, 'Y': Yo, 'R': Ro, "ID": IDo}, file)
        
    # if len(replay_paths)>0:
    #     process_replays(replay_paths, c1, c2, s, iteration, section+1)



if __name__ == '__main__':

    # Mass Generate
    f = open('/root/data/jay/training_data/(!QUESTIONMARK) Falco vs Peach [PS] Game_20200219T203538.slp', 'r')
    replays = json.load(f)
    characters = [melee.Character.FALCO, melee.Character.JIGGLYPUFF, melee.Character.MARTH, melee.Character.CAPTAIN_FALCON, melee.Character.FOX]
    #characters = list(melee.Character)
    total_n = 0
    succed_n = 0
    exceptions = []
    iteration = 0
    for e, c1 in enumerate(characters[-1:]):
        for c2 in characters:
            for s in [melee.Stage.FINAL_DESTINATION]:
                try:
                    if (os.path.exists(f'./Data/{c2.name}_{c1.name}_on_{s.name}_0_data.pkl')):
                        iteration += 1
                        continue
                    process_replays(replays[f'{c1.name}_{c2.name}'][s.name], c1, c2, s, iteration)
                    total_n += 1
                    succed_n += 1
                except Exception as exc:
                    if type(exc) is KeyError:
                        total_n += 1
                        exceptions.append(exc)
                    else:
                        raise exc
                iteration += 1                        
    
    print(f"Ratio: {succed_n}/{total_n}")
    print(exceptions)
    