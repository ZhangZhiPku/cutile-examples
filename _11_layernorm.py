import cuda.tile as ct
import torch

@ct.kernel
def layernorm(
    x: ct.Array, w: ct.Array, b: ct.Array, o: ct.Array, 
    eps: float, tile_size: ct.Constant, 
    allow_tma: ct.Constant
):
    block_x = ct.bid(0)
    tile_x = ct.load(x, (block_x, 0), (1, tile_size), allow_tma=allow_tma)
    tile_x = ct.astype(tile_x, ct.float32)
    
    local_mean = ct.sum(tile_x) / tile_size
    tile_x = tile_x - local_mean

    local_rsqrt = ct.rsqrt(ct.sum(tile_x * tile_x) / tile_size + eps)
    tile_x = tile_x * local_rsqrt

    # apply w, b
    tile_w = ct.load(w, (0, ), (tile_size, ), allow_tma=allow_tma).astype(ct.float32).reshape((1, tile_size))
    tile_b = ct.load(b, (0, ), (tile_size, ), allow_tma=allow_tma).astype(ct.float32).reshape((1, tile_size))
    tile_x = tile_x * tile_w + tile_b

    ct.store(o, (block_x, 0), tile_x, allow_tma=allow_tma)

if __name__ == "__main__":
    M, N = 16384, 1024
    for i in range(100):
        x = torch.empty(size=[M, N], device="cuda", dtype=torch.bfloat16)
        w = torch.empty(size=[N], device="cuda", dtype=torch.bfloat16)
        b = torch.empty(size=[N], device="cuda", dtype=torch.bfloat16)
        o = torch.empty_like(x)
        ct.launch(
            torch.cuda.current_stream(), (M, 1, 1),
            layernorm, (x, w, b, o, 1e-7, N)
        )