import torch
import cuda.tile as ct

@ct.function
def tile_rnd(seed: ct.Tile) -> ct.Tile:
    return seed * 1103515245 + 12345

@ct.kernel
def device_rnd(rnd: ct.Tile, tile_size: ct.Constant):
    block_idx = ct.bid(0)
    seed = ct.arange(tile_size, dtype=ct.int32) + block_idx * tile_size
    ct.store(rnd, (block_idx, ), tile_rnd(seed))

n, tile_size = 2048, 512
num_blocks = (n // tile_size, )
out = torch.empty(size=[n], device="cuda", dtype=torch.int)

ct.launch(
    torch.cuda.current_stream(), num_blocks, 
    device_rnd, (out, tile_size)
)

print(out)