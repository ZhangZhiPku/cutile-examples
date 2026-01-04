import math
import torch
import torch.nn as nn
import cuda.tile as ct
import numpy as np
from time import time

import torch

@ct.kernel
def build_freqs(
    o: ct.Array, hidden_dim: int, theta: float, 
    tile_size: ct.Constant, allow_tma: ct.Constant
):
    # o: [max_seq_len, hidden_dim // 2, [2, 2]]
    block_x, block_y = ct.bid(0), ct.bid(1)
    tile_o = ct.arange(tile_size, dtype=torch.float32) + block_x * tile_size
    tile_o = 1.0 / (ct.pow(theta, tile_o / (hidden_dim / 2)))
    tile_o = block_y * tile_o
    
    sin = ct.sin(tile_o).reshape((1, tile_size, 1, 1))
    cos = ct.cos(tile_o).reshape((1, tile_size, 1, 1))
    r1 = ct.cat((cos, -sin), axis=2)
    r2 = ct.cat((sin, cos), axis=2)
    rotation = ct.cat((r1, r2), axis=3)
    ct.store(o, (block_y, block_x, 0, 0), rotation, allow_tma=allow_tma)

@ct.kernel
def apply_rope(x: ct.Array, coord: ct.Array, o: ct.Array, freqs: ct.Array, tile_size: ct.Constant):
    # coord: [max_seq_len]
    # x, o: [seq, hidden_dim]
    # freqs: [max_seq_len, hidden_dim // 2, [2, 2]]
    block_x, block_y = ct.bid(0), ct.bid(1)
    pos = ct.load(coord, (block_y, ), (1, ), allow_tma=False)
    tile_x = ct.load(x, (block_y, block_x), (1, tile_size), allow_tma=False).astype(ct.float32)
    tile_f = ct.load(freqs, (pos.item(), block_x, 0, 0), (1, tile_size // 2, 2, 2), allow_tma=False) # enable l2 cache
    tile_x = tile_x.reshape((-1, 1, 2))
    tile_f = tile_f.reshape((-1, 2, 2))
    
    tile_o = ct.full(shape=tile_x.shape, fill_value=0.0, dtype=torch.float32)
    tile_o = ct.mma(tile_x, tile_f, tile_o).reshape((1, tile_size))
    ct.store(o, (block_y, block_x), tile_o.astype(o.dtype), allow_tma=False)