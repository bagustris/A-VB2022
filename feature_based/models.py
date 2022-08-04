# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Copyright 2022 The Hume AI Authors. All Rights Reserved.
# Code available under a Creative Commons Attribution-Non Commercial-No Derivatives 4.0
# International Licence (CC BY-NC-ND) license.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import torch
from torch import nn

# import numpy as np

from end2you.models.audio import AudioModel
from end2you.models.rnn import RNN
# from pathlib import Path


class MLPClass(nn.Module):
    def __init__(self, feat_dimensions, output_len):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dimensions, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            # nn.Linear(64, 32),
            # nn.LayerNorm(32),
            # nn.LeakyReLU(),
            # nn.Linear(32, 16),
            # nn.LayerNorm(16),
            # nn.LeakyReLU(),
        )

        self.class_output = nn.Sequential(
            nn.Linear(64, 32), nn.LeakyReLU(), nn.Linear(32, output_len)
        )

    def forward(self, x):
        main_mlp = self.mlp(x)
        output = self.class_output(main_mlp)
        return output


class MLPReg(nn.Module):
    def __init__(self, feat_dimensions, output_len):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dimensions, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
        )

        self.high_output = nn.Sequential(
            nn.Linear(64, 32), nn.LeakyReLU(), nn.Linear(32, output_len)
        )

    def forward(self, x):
        main_mlp = self.mlp(x)
        output = torch.sigmoid(self.high_output(main_mlp))
        # output = torch.tanh(self.high_output(main_mlp))
        return output


class AudioRNNModel(nn.Module):
    
    def __init__(self,
                 input_size:int,
                 num_outs:int,
                 pretrained:bool = False,
                 model_name:str = None):
        """ Convolutional recurrent neural network model.
        
        Args:
            input_size (int): Input size to the model. 
            num_outs (int): Number of output values of the model.
            pretrained (bool): Use pretrain model (default `False`).
            model_name (str): Name of model to build (default `None`).
        """
        
        super(AudioRNNModel, self).__init__()
        audio_network = AudioModel(model_name=model_name, input_size=input_size)
        self.audio_model = audio_network.model
        num_out_features = audio_network.num_features
        self.rnn, num_out_features = self._get_rnn_model(num_out_features)
        self.linear = nn.Linear(num_out_features, num_outs)
        self.num_outs = num_outs
    
    def _get_rnn_model(self, input_size: int):
        """ Builder method to get RNN instace."""
        
        rnn_args = {
            'input_size': input_size,
            'hidden_size': 4,
            'num_layers': 1,
            'batch_first': True
        }
        return RNN(rnn_args, 'lstm'), rnn_args['hidden_size']
    
    def forward(self, x:torch.Tensor):
        """
        Args:
            x ((torch.Tensor) - BS x S x 1 x T)
        """
        
        batch_size, t = x.shape
        x = x.view(batch_size, 1, t)
        
        audio_out = self.audio_model(x)
        audio_out = audio_out.view(batch_size, -1)
        
        rnn_out, _ = self.rnn(audio_out)
        
        output = self.linear(rnn_out)
        
        return output

