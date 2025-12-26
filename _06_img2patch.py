import math
import torch
import cuda.tile as ct

@ct.kernel
def img2patch(
    x: ct.Array, y: ct.Array, coord: ct.Array,
    patch_size_x: ct.Constant, 
    patch_size_y: ct.Constant
):
    block_x, block_y, block_z = ct.bid(0), ct.bid(1), ct.bid(2)
    num_block_y = ct.num_blocks(1)
    
    tile = ct.load(x, (block_z, block_x, block_y), (1, patch_size_x, patch_size_y), padding_mode=ct.PaddingMode.ZERO)
    tile = ct.reshape(tile, shape=(1, patch_size_x * patch_size_y))
    
    ct.store(y, (block_x * block_y, 0), tile)
    
    local_coord_x = ct.full(shape=(1, 1), fill_value=(block_x), dtype=ct.int32)
    local_coord_y = ct.full(shape=(1, 1), fill_value=(block_y), dtype=ct.int32)
    ct.store(coord, (block_x + block_y * num_block_y, 0), ct.cat((local_coord_x, local_coord_y), axis=1))

n, c, h, w = 1, 3, 31, 31
patch_size_x: int = 32
patch_size_y: int = 32
num_patch_x: int = ct.cdiv(w, patch_size_x)
num_patch_y: int = ct.cdiv(h, patch_size_y)
x = torch.rand(size=(c, h, w), device="cuda", dtype=torch.float)
y = torch.empty(size=(num_patch_x * num_patch_y, patch_size_x * patch_size_y * c), device="cuda", dtype=torch.float)
coord = torch.empty(size=(num_patch_x * num_patch_y, 2), dtype=torch.int32, device="cuda")
grid = (num_patch_x, num_patch_y, c)

ct.launch(
    torch.cuda.current_stream(), grid, img2patch, 
    (x, y, coord, patch_size_x, patch_size_y)
)
print(y)