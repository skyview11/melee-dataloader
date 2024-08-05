from torch.utils.data import Dataset
from configuration_dataset import DatasetConfig
import numpy as np
import os
import json
import torch

class SlippiDataset(Dataset):
    def __init__(self, config:DatasetConfig):
        self.config = config
        [erase_wrong(file) for file in self.get_mm_file_names()]
        self.mm_list = [read_memmap(file) for file in self.get_mm_file_names()]
        self.start_indices = [0]
        self.total_len = 0
        self.seq_len = config.seq_len
        
        for mm in self.mm_list[:-1]:
            self.start_indices.append(self.start_indices[-1] + mm.shape[0])
            self.total_len += mm.shape[0]
        self.start_indices = np.array(self.start_indices)
        self.total_len += self.mm_list[-1].shape[0]
        
    def get_mm_file_names(self):
        mm_base_path = self.config.dataset_basepath
        mm_file_names = []
        for file_name in os.listdir(mm_base_path):
            if file_name.endswith("dat"):
                mm_file_names.append(os.path.join(mm_base_path, file_name))
        return mm_file_names
    def __len__(self):
        return self.total_len
    def __getitem__(self, index):
        target_game_id = np.arange(len(self.mm_list))[index >= self.start_indices][-1]
        target_game = self.mm_list[target_game_id]
        start_idx = index - self.start_indices[index >= self.start_indices][-1]
        end_idx = min(start_idx + self.seq_len, start_idx + len(target_game))
        
        if (end_idx - start_idx) < self.seq_len: # add padding
            padding = np.zeros((self.seq_len - (end_idx - start_idx), target_game.shape[1]), dtype=target_game.dtype)
            final_mm = np.concatenate([target_game[start_idx:end_idx], padding], axis=0)
        else:
            final_mm = target_game[start_idx:end_idx]
        final_mm = torch.from_numpy(final_mm)
        print(final_mm.shape)
        return { # TODO HARDCODE
            "p1_state": final_mm[:, :49], 
            "p2_state": final_mm[:, 49:98], # 49 + 49 
            "p1_action": final_mm[:, 100:108], 
            "p2_action": final_mm[:, 108:116],
            "p1_reward": final_mm[:, 98].reshape(-1, 1), 
            "p2_reward": final_mm[:, 99].reshape(-1, 1)
        }
    
def erase_wrong(mem_file_name):
    with open(f"{mem_file_name}.conf", "r") as file:
        try:
            memmap_configs = json.load(file)
        except:
            os.system(f"rm -rf '{mem_file_name}'")
            os.system(f"rm -rf '{mem_file_name}.conf'")
            print("d")

def read_memmap(mem_file_name):
    with open(f"{mem_file_name}.conf", "r") as file:
        memmap_configs = json.load(file)    
        return np.memmap(mem_file_name, mode='r+', \
            shape=tuple(memmap_configs['shape']), \
  dtype=memmap_configs['dtype'])
        

if __name__ == "__main__":
    config = DatasetConfig(seq_len=10000)
    dataset = SlippiDataset(config)
    
    for id, data in enumerate(dataset):
        for k, v in data.items():
            print(k, v.shape)
        break
    