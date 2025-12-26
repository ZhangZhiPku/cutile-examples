import torch
import cuda.tile as ct

@ct.function
def tile_quant(x: ct.Tile) -> tuple[ct.Tile, ct.Tile]:
    absmax = ct.max(x) # no abs avaliable
    absmax.astype(ct.float32)
    scale = absmax / 128
    x = x / scale
    return ct.astype(x, ct.int8), scale

@ct.kernel
def ct_quant(x: ct.Array, y: ct.Array, s: ct.Array, tile_size: ct.Constant):
    # x, y: [M, N]
    # s: [M, ]
    block_id = ct.bid(0)
    
    tile_x = ct.load(x, index=(block_id, 0), shape=(1, tile_size))
    tile_x, tile_s = tile_quant(tile_x)
    
    ct.store(y, (block_id, 0), tile_x)
    ct.store(s, (block_id, ), tile_s)

# problem define
m, n = 1024, 1024
num_blocks, tile_size = m, n

x = torch.rand(size=[m, n], device="cuda", dtype=torch.half) - 0.5
y = torch.empty_like(x)
scale = torch.empty(size=[n, ], device="cuda", dtype=torch.float32)

ct.launch(
    torch.cuda.current_stream(), (num_blocks, ), 
    ct_quant, (x, y, scale, tile_size)
)

print(y)

