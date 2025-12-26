import torch
import cuda.tile as ct

@ct.kernel
def ct_sum(x: ct.Array, y: ct.Array, tile_size: ct.Constant):
    # x, y: 1d array with any length
    block_id = ct.bid(0)
    tile_x = ct.load(x, index=(block_id, ), shape=(tile_size, ))
    tile_x = ct.sum(tile_x.astype(ct.float32))
    ct.atomic_add(y, (0, ), tile_x.astype(y.dtype))

x = torch.randn(size=[1024 * 1024], device="cuda", dtype=torch.float16)
y = torch.zeros(size=(1, ), device="cuda", dtype=torch.float16)

tile_size: int = 256
num_elements = x.numel()
num_blocks = (num_elements // tile_size)

ct.launch(
    torch.cuda.current_stream(), (num_blocks, ), 
    ct_sum, (x, y, tile_size)
)

print(y)