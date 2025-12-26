import cuda.tile as ct
import torch

@ct.kernel
def silu_fuse_mul(
    x: ct.Array, gate: ct.Array, o: ct.Array,
    tile_size: ct.Constant
):
    block_x = ct.bid(0)
    tile_x = ct.load(x, (block_x, ), (tile_size, )).astype(ct.float32)
    tile_g = ct.load(gate, (block_x, ), (tile_size, )).astype(ct.float32)
    tile_o = tile_x / (1 + ct.exp(tile_g))
    ct.store(o, (block_x, ), tile_o.astype(o.dtype))

if __name__ == "__main__":
    M, N, K = 32, 12288, 3072
    for i in range(100):
        x = torch.empty(size=[M, K], device="cuda", dtype=torch.bfloat16)
        g = torch.empty(size=[M, N], device="cuda", dtype=torch.bfloat16)
        pred = torch.empty(size=[M, N], device="cuda", dtype=torch.bfloat16)

        ct.launch(
            torch.cuda.current_stream(), 
            (ct.cdiv(M * N, 1024), ),
            silu_fuse_mul, (x.flatten(), g.flatten(), pred.flatten(), 1024)
        )
        # print(real - pred)