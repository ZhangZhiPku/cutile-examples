import cuda.tile as ct
import torch

@ct.kernel
def linear(
    x: ct.Array, wT: ct.Array, o: ct.Array,
    sx: ct.Array, swT: ct.Array,
    tileM: ct.Constant, tileN: ct.Constant, tileK: ct.Constant
):
    k = x.shape[-1]
    block_x, block_y = ct.bid(0), ct.bid(1)
    
    num_tile_k = ct.cdiv(k, tileK)
    accumulator = ct.full((tileM, tileN), 0, dtype=ct.float32)
    
    for k_iter in range(num_tile_k):
        tile_x = ct.load(x, (block_x, k_iter), (tileM, tileK), padding_mode=ct.PaddingMode.ZERO)
        tile_w = ct.load(wT, (k_iter, block_y), (tileK, tileN), order="F", padding_mode=ct.PaddingMode.ZERO)
        accumulator = ct.mma(tile_x, tile_w, accumulator)
    
    tile_sx  = ct.load(sx, (block_x), (1, ))
    tile_swT = ct.load(swT, (block_y), (1, ))
    accumulator = accumulator * tile_sx * tile_swT
    accumulator = ct.astype(accumulator, ct.bfloat16)

    ct.store(o, (block_x, block_y), accumulator)

M, N, K = 4096, 4096, 4096
tileM, tileN, tileK = 128, 128, 128
x = torch.rand(size=[M, K], device="cuda", dtype=torch.bfloat16)
w = torch.rand(size=[N, K], device="cuda", dtype=torch.bfloat16)
x = x.to(dtype=torch.float8_e4m3fn)
w = w.to(dtype=torch.float8_e4m3fn)
xs = torch.rand(size=[ct.cdiv(M, tileM)], device="cuda", dtype=torch.float32)
ws = torch.rand(size=[ct.cdiv(N, tileN)], device="cuda", dtype=torch.float32)
pred = torch.empty(size=[M, N], device="cuda", dtype=torch.bfloat16)

ct.launch(
    torch.cuda.current_stream(), 
    (ct.cdiv(M, tileM), ct.cdiv(N, tileN)),
    linear, (x, w, pred, xs, ws, tileM, tileN, tileK)
)