import torch
import cuda.tile as ct

@ct.function
def tile_norm(x: ct.Tile, tile_size: ct.Constant, eps: float=1e-6) -> ct.Tile:
    mean = ct.sum(x) / tile_size
    x = x - mean
    rsqrt = ct.rsqrt(ct.sum(x * x) / tile_size + eps)
    x = x * rsqrt
    return x

@ct.kernel
def ct_norm(x: ct.Array, y: ct.Array, tile_size: ct.Constant):
    # x, y [M, N]
    # N == tile_size
    block_id = ct.bid(0)
    
    tile_x = ct.load(x, index=(block_id, 0), shape=(1, tile_size))
    tile_x = tile_norm(tile_x, tile_size)
    
    ct.store(y, (block_id, 0), tile_x.astype(x.dtype))

x = torch.randn(size=[1024, 1024], device="cuda", dtype=torch.float16)
y = torch.empty_like(x)
tile_size: int = 1024

num_blocks = x.shape[0]
ct.launch(
    torch.cuda.current_stream(), (num_blocks, ), 
    ct_norm, (x, y, tile_size)
)

print(y)

