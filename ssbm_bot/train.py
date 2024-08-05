#!/usr/bin/python3
import copy
from collections import deque

import melee

import os
import json
from tqdm import tqdm
import time
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torch.optim import Adam
from torchvision.ops import sigmoid_focal_loss
import math

import pickle

import Args
import MovesList
import copy

args = Args.get_args()

SEQ_LEN = 4
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
NUM_WORKERS = 0

class CustomDataset(Dataset):
    def __init__(self, X, Y, data_len):
        super(CustomDataset, self).__init__()

        self.X = X
        self.Y = Y
        self.data_len = data_len

    def __getitem__(self, idx):

        feature_lst = []

        for temp_idx in range(idx, idx+SEQ_LEN):
            x = self.X[temp_idx]            
            feature_lst.append([x])

        feature_tensor = np.concatenate(feature_lst, axis=0)
        label_tensor = self.Y[idx+SEQ_LEN-1]
        return feature_tensor, label_tensor

    def __len__(self):
        return self.data_len
    

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(self, input_embed_size, target_embed_size, num_classes, num_layers, nhead, dropout=0.5, max_seq_length=5000):
        super(TransformerClassifier, self).__init__()

        # Initial embedding layer
        self.embedding = nn.Linear(input_embed_size, target_embed_size)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(target_embed_size, dropout, max_seq_length)

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=target_embed_size,
            nhead=nhead,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer
        self.output_layer = nn.Linear(target_embed_size, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_length, input_embed_size]
        
        # Embedding
        x = self.embedding(x)  # [batch_size, seq_length, target_embed_size]
        
        # Permute x to [seq_length, batch_size, target_embed_size] for Transformer and Positional Encoding
        x = x.permute(1, 0, 2)
        
        # Apply Positional Encoding
        x = self.positional_encoding(x)
        
        # Transformer Encoder
        x = self.transformer_encoder(x)
        
        # Permute x back to [batch_size, seq_length, target_embed_size]
        x = x.permute(1, 0, 2)
        
        # Average pooling across the sequence dimension
        x = x.mean(dim=1)
        
        # Output layer
        logits = self.output_layer(x)
        
        return logits


# Example usage
embed_size = 128
num_classes = 21
num_layers = 4
input_embed_size = 37  # Original embedding size
target_embed_size = 128 # Desired embedding size
nhead = 8

def create_model(X: np.ndarray, Y: np.ndarray, player_character: melee.Character, opponent_character: melee.Character,
                 stage: melee.Stage,
                 folder: str, lr: float, wd: float):
    input_embed_size = X.shape[1]
    dataset = CustomDataset(
        X, Y,
        len(X)-SEQ_LEN
        )

    train_ratio = 0.8
    valid_ratio = 1 - train_ratio
    
    data_size = len(dataset)
    train_size = int(data_size * train_ratio)
    valid_size = data_size - train_size
    
    train_set, valid_set = random_split(dataset, [train_size, valid_size])
    
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=BATCH_SIZE,
        drop_last=False,
        num_workers=NUM_WORKERS,
        )
    
    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=BATCH_SIZE,
        drop_last=False,
        num_workers=NUM_WORKERS,
        )
    
    model = TransformerClassifier(input_embed_size, target_embed_size, num_classes, num_layers, nhead).to(DEVICE)
    
    # train
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(params=model.parameters() ,lr=lr)#,weight_decay=wd)

    epochs = 10
    for epoch in range(epochs):
        loss_sum = 0
        correct = 0
        for b, data in enumerate(zip(X,Y)):
            if len(X)-SEQ_LEN <= b:
                break
            x, y = X[b:b+SEQ_LEN], Y[b+SEQ_LEN-1]
            x = torch.tensor(np.array([x]))
            y = torch.tensor([y])
            x = x.to(DEVICE).float()
            y = y.to(DEVICE).float()
            
            # print(x.shape, y.shape)
            # print(y[0])
            
            optimizer.zero_grad()
                        
            pred = model(x)
            
            loss = criterion(pred, y)
            loss.backward()
            loss_sum += loss.item()
            # if loss.item()>10:
            #     print("loss over 10")
            
            optimizer.step()
            
            pred_label = torch.argmax(pred)
            gt_label = torch.argmax(y)
            if pred_label == gt_label:
                correct += 1
            
            print_freq = 50
            if b % print_freq == print_freq - 1:
                print(f"Epoch: {epoch+1}, loss: {loss_sum/print_freq}, acc: {correct / print_freq * 100}%")
                loss_sum = 0
                correct = 0

    # folder = 'models'
    pickle_file_path = f'{folder}/{player_character.name}_v_{opponent_character.name}_on_{stage.name}.pt'

    if not os.path.isdir(folder):
        os.mkdir(f'{folder}/')

    with open(pickle_file_path, 'wb') as file:
        torch.save(model.state_dict(), pickle_file_path)


if __name__ == '__main__':
    player_character = melee.Character.FALCO
    opponent_character = melee.Character.JIGGLYPUFF
    stage = melee.Stage.FINAL_DESTINATION
    lr = 3e-3
    wd = 0.00

    raw = open(f'Data/{player_character.name}_{opponent_character.name}_on_{stage.name}_data.pkl', 'rb')
    data = pickle.load(raw)
    X = data['X']
    Y = data['Y']
    create_model(X, Y, player_character=player_character,
                 opponent_character=opponent_character, stage=stage, folder='models', lr=lr, wd=wd)
