import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image

import math
import random
import numpy as np
from typing import List, Optional
import json
import sys
import os


class KeyQueryAttention(nn.Module):
    def __init__(
        self, 
        dim_key: int, 
        dim_query: int, 
        d_k: int):
        super().__init__()

        self.d_k = d_k
        self.w_q = nn.Linear(dim_query, d_k)
        self.w_k = nn.Linear(dim_key, d_k)
        self.temperature = 1.0 / math.sqrt(d_k)

    def forward(self, query: Tensor, key: Tensor):
        query, key = self.w_q(query), self.w_k(key)
        att = torch.einsum('bij,bkj->bik', query, key) * self.temperature
        return att
    

class MLP(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_layer_dims: Optional[List[int]], 
        output_dim: int):
        super().__init__()

        layers = []
        layer_dims = (input_dim,) + tuple(hidden_layer_dims)
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            layers.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(inplace=True)
            ))
        layers.append(nn.Linear(layer_dims[-1], output_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    

class LayerNorm(nn.Module):

    def forward(self, input):
        return F.layer_norm(input, input.size()[1:])
    

class ImageEncoder(nn.Module):
    def __init__(self, dim_slot_embed: int):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2), nn.ELU(), LayerNorm(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ELU(), LayerNorm(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ELU(), LayerNorm(),
            nn.Flatten(),
            nn.Linear(2304, dim_slot_embed), nn.ELU(), LayerNorm()
        )

    def forward(self, x):
        return self.model(x)
    

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(input.size(0), *self.shape)
    

class Interpolate(nn.Module):
    def __init__(self, scale_factor: float, mode: str):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
    
    def forward(self, input):
        return F.interpolate(input, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
    

class ImageDecoder(nn.Module):
    def __init__(self, dim_slot_embed: int):
        super().__init__()
        
        '''self.model = nn.Sequential(
            LayerNorm(),
            nn.Linear(dim_slot_embed, 4096), nn.ReLU(), LayerNorm(),
            Reshape([64, 8, 8]),
            Interpolate(2, 'bilinear'), nn.ReplicationPad2d(2),
            nn.Conv2d(64, 32, kernel_size=4, stride=1, padding=0), nn.ReLU(), LayerNorm(),
            Interpolate(2, 'bilinear'), nn.ReplicationPad2d(2),
            nn.Conv2d(32, 16, kernel_size=4, stride=1, padding=0), nn.ReLU(), LayerNorm(),
            Interpolate(2, 'bilinear'),
            nn.Conv2d(16, 16, kernel_size=4, stride=1, padding=0), nn.ReLU(), LayerNorm(),
            nn.Conv2d(16, 1, kernel_size=4, stride=1, padding=0), nn.Sigmoid()
        )'''
        
        self.model = nn.Sequential(
            LayerNorm(),
            nn.Linear(dim_slot_embed, 4096), nn.ELU(), LayerNorm(),
            Reshape([64, 8, 8]),
            nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.PixelShuffle(2), nn.ELU(), LayerNorm(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.PixelShuffle(2), nn.ELU(), LayerNorm(),
            nn.Conv2d(16, 1 * 4, 5, stride=1, padding=2), nn.PixelShuffle(2), nn.Sigmoid()
        )
        
        
        
    def forward(self, x):
        return self.model(x)


class GroupLinear(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        num_blocks: int):
        super().__init__()
        
        limit = 1. / math.sqrt(output_dim)
        self.weight = nn.Parameter(torch.FloatTensor(num_blocks, input_dim, output_dim).uniform_(-limit, limit))
        self.bias = nn.Parameter(torch.FloatTensor(num_blocks, output_dim).uniform_(-limit, limit))

    def forward(self, x: Tensor):
        x = x.permute(1, 0, 2)
        x = torch.bmm(x, self.weight)
        x = x.permute(1, 0, 2)
        x = x + self.bias
        return x


def argmax_onehot(x: Tensor, dim: int):
    idx = x.argmax(dim=dim)
    onehot = torch.zeros_like(x).scatter_(dim, idx.unsqueeze(dim), 1.0)
    return onehot


def set_seed(seed: int, cuda: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def custom_argparser():
    argv_dict = {}
    for arg in sys.argv[1:]:
        key, sep, value = arg.partition('=')
        if value in ['True', 'False']:
            value = value == 'True'
        elif value.isdigit():
            value = int(value)
        elif "[" in value or "]" in value:
            value = json.loads(value)
        else:
            try:
                value = float(value)
            except ValueError:
                pass
        argv_dict[key[2:]] = value
    return argv_dict


@torch.no_grad()
def save_transformations(model, dataloader, device, num_samples: int):
    model.eval()
    test_operations = []
    
    temp_idx = num_samples
    for batch_idx, (images, operations) in enumerate(dataloader):
        images, operations = images.to(device), operations.to(device)
        
        model.rule_network.reset()
        
        test_image = images[0, 0, ...]
        test_operation = operations[0, 0, ...]

        outputs = model(images[:, :-1, ...], operations)
        
        transformed_test_image = outputs[0, ...]

        test_operations.append(test_operation)
        test_image_filepath = os.path.join('./NPS-code-to-run/res', f'test_image_{batch_idx}.png')
        save_image(test_image, test_image_filepath)
        transformed_test_image_filepath = os.path.join('./NPS-code-to-run/res', f'transformed_test_image_{batch_idx}.png')
        save_image(transformed_test_image, transformed_test_image_filepath)
        
        temp_idx -= 1
        if temp_idx == 0:
            break
        
    with open("./NPS-code-to-run/res/test_operations.txt", 'w') as output:
        for test_operation in test_operations:
            output.write(str(test_operation) + '\n')


def count_mapping(s: np.ndarray, t: np.ndarray, keys, return_string=True):
    assert len(s.shape) == len(t.shape) == 1
    assert len(s) == len(t)
    # assert s.dtype in (np.uint8, np.uint16, np.uint32, np.uint64)
    # assert t.dtype in (np.uint8, np.uint16, np.uint32, np.uint64)
    func = lambda _, a, b: dict(zip(
        *np.unique(b[np.where(a == _)[0]], return_counts=True)
    ))
    cnts_t = []
    for idx_s, key_s in enumerate(keys):
        cnt_t = func(idx_s, s, t)
        cnts_t.append(cnt_t)
    map_st = dict(zip(keys, cnts_t))
    if return_string:
        map_st = str(map_st).replace("'", '')
    return map_st
