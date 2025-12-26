import cuda.tile as ct
import torch

@ct.kernel
def rmsnorm(x: ct.Array, w: ct.Array, o: ct.Array, eps: float, tile_size: ct.Constant):
    block_x = ct.bid(0)
    
    tile_x = ct.load(x, (block_x, 0), (1, tile_size), allow_tma=False)
    tile_w = ct.load(w, (0, ), (tile_size, ), allow_tma=False).reshape((1, tile_size))

    output_dtype = tile_x.dtype
    tile_x = tile_x.astype(ct.float32)
    
    square_mean = ct.sum(tile_x * tile_x * (1 / tile_size)) + eps
    square_root = ct.rsqrt(square_mean)
    tile_o = tile_x * square_root * tile_w.astype(ct.float32)

    tile_o = tile_o.astype(output_dtype)
    ct.store(o, (block_x, 0), tile_o, allow_tma=False)

if __name__ == "__main__":
    M, N = 16384, 1024
    for i in range(100):
        x = torch.empty(size=[M, N], device="cuda", dtype=torch.bfloat16)
        w = torch.empty(size=[N], device="cuda", dtype=torch.bfloat16)
        real = torch.nn.functional.rms_norm(input=x, normalized_shape=(N, ), weight=w, eps=1e-7)

        x = torch.empty(size=[M, N], device="cuda", dtype=torch.bfloat16)
        w = torch.empty(size=[N], device="cuda", dtype=torch.bfloat16)
        o = torch.empty_like(x)
        ct.launch(
            torch.cuda.current_stream(), (M, 1, 1),
            rmsnorm, (x, w, o, 1e-7, N)
        )