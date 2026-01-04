import torch
import cuda.tile as ct


@ct.kernel
def batch_average_mse_loss(
    x: ct.Array, y: ct.Array, o: ct.Array, dodx: ct.Array, 
    tile_size: ct.Constant, allow_tma: ct.Constant=True):
    # x, y, dodx: [M, N]
    M, N = x.shape[0], x.shape[1]
    inv_term = 1 / (M * N)
    
    # use block x on N is better
    block_x, block_y = ct.bid(0), ct.bid(1)

    tile_x = ct.load(
        x, (block_y, block_x), (1, tile_size), 
        allow_tma=allow_tma, padding_mode=ct.PaddingMode.ZERO
    ).astype(ct.float32)

    tile_y = ct.load(
        y, (block_y, block_x), (1, tile_size), 
        allow_tma=allow_tma, padding_mode=ct.PaddingMode.ZERO
    ).astype(ct.float32)
    
    loss = ct.sum(ct.pow(tile_x - tile_y, 2)) * inv_term
    
    # gradient: do / dx
    tile_dodx = 2 * inv_term * (tile_x - tile_y)
    ct.store(dodx, (block_y, block_x), tile_dodx.astype(dodx.dtype))
    ct.atomic_add(o, (0, ), loss.astype(o.dtype))