from absl import app
from slippi import Game
from tqdm import tqdm
import json
import os
import argparse

replay_folder= './training_data'
replay_paths = []

for root, dirs, files in os.walk(replay_folder):
    for name in files:
        replay_paths.append(os.path.join(root, name))
filename = 'replays.json'
if not os.path.exists(filename):
    with open(filename, 'w+') as file:
        json.dump({}, file, indent=4)

f = open(filename, 'r')
j = json.load(f)

for path in tqdm(replay_paths):
    try:
        game = Game(path)
        stage = game.start.stage.name
        players = list(set(game.frames[10].ports) - {None})
        if len(players)!=2:
            continue
        p1_char_name = players[0].leader.post.character.name
        p2_char_name = players[1].leader.post.character.name
        key = f'{p1_char_name}_{p2_char_name}'
        if key not in j:
            j[key] = {}
        if game.start.stage.name not in j[key]:
            j[key][stage] = []
        j[key][stage].append(path)

        if p1_char_name != p2_char_name:
            key = f'{p2_char_name}_{p1_char_name}'
            if key not in j:
                j[key] = {}
            if stage not in j[key]:
                j[key][stage] = []
            j[key][stage].append(path)
    except:
        with open(filename, 'w') as file:
            json.dump(j, file, indent=2)
        continue
with open(filename, 'w') as file:
    json.dump(j, file, indent=2)
