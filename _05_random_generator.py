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

@ct.kernel
def sample(x: ct.Array, y: ct.Array, tile_size: ct.Constant, seed: int):
    # x, y: 1d array
    block_x = ct.bid(0)
    seed = ct.bid(0) + seed
    seed = (seed * 1103515245 + 12345) % tile_size
    tile_x = ct.load(x, (block_x * tile_size + seed, ), (1, ), allow_tma=False)
    ct.store(y, (block_x, ), tile_x)

if __name__ == "__main__":
    N, tile_size = 16, 4
    x = torch.rand(size=(N, ), device="cuda", dtype=torch.float32)
    s = torch.empty(size=(N // tile_size, ), device="cuda", dtype=torch.float32)
    ct.launch(torch.cuda.current_stream(), (N // tile_size, ), sample, (x, s, tile_size, 10086))
