import torch
import cuda.tile as ct
import math

# use exp2 is much faster than exp
INV_LOG_2 = 1.0 / math.log(2)

@ct.kernel
def softmax(x: ct.Array, y: ct.Array, tile_size: ct.Constant):
    # x, y: [M, N]
    block_idx = ct.bid(0)

    tile_x = ct.load(x, (block_idx, 0), (1, tile_size)) * INV_LOG_2
    tile_x = ct.exp2(tile_x - ct.max(tile_x))
    
    expsum = ct.sum(tile_x)
    ct.store(y, (block_idx, 0), ct.truediv(tile_x, expsum))
    
# problem define
m, n = 15, 256
num_blocks, tile_size = m, n

x = torch.randn(size=[m, n], device="cuda", dtype=torch.float16)
y = torch.empty_like(x)

ct.launch(
    torch.cuda.current_stream(), (num_blocks, ), 
    softmax, (x, y, tile_size)
)

print(y)
print(torch.softmax(x, dim=-1))