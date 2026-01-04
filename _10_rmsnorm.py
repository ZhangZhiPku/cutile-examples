import cuda.tile as ct
import torch

@ct.kernel
def ct_layernorm(
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

@ct.kernel
def ct_rmsnorm(x: ct.Array, w: ct.Array, o: ct.Array, rsqrt: ct.Array, eps: float, tile_size: ct.Constant):
    block_x = ct.bid(0)
    
    tile_x = ct.load(x, (block_x, 0), (1, tile_size), allow_tma=False)
    tile_w = ct.load(w, (0, ), (tile_size, ), allow_tma=False).reshape((1, tile_size))

    output_dtype = tile_x.dtype
    tile_x = tile_x.astype(ct.float32)
    
    square_mean = ct.sum(tile_x * tile_x) * (1 / tile_size) + eps
    square_root = ct.rsqrt(square_mean)
    tile_o = tile_x * square_root * tile_w.astype(ct.float32)

    tile_o = tile_o.astype(output_dtype)
    ct.store(o, (block_x, 0), tile_o, allow_tma=False)
    # for backward
    ct.store(rsqrt, (block_x, ), square_root, allow_tma=False)

@ct.kernel
def ct_rmsnorm_bwd_dy_dw(
    g: ct.Array, x: ct.Array, rsqrt: ct.Array, dw: ct.Array,
    tileM: ct.Constant, tileN: ct.Constant,
):
    M = g.shape[0]
    block_x, block_y = ct.bid(0), ct.bid(1)
    tile_g = ct.load(g, (block_y, block_x), (tileM, tileN), allow_tma=False, padding_mode=ct.PaddingMode.ZERO).astype(ct.float32)
    tile_x = ct.load(x, (block_y, block_x), (tileM, tileN), allow_tma=False, padding_mode=ct.PaddingMode.ZERO).astype(ct.float32)
    tile_r = ct.load(rsqrt, (block_y, ), (tileM, ), allow_tma=False, padding_mode=ct.PaddingMode.ZERO).reshape((tileM, 1)).astype(ct.float32)

    tile_x = tile_g * tile_x * tile_r
    tile_x = ct.sum(tile_x, axis=0, keepdims=False)

    ct.atomic_add(dw, ct.arange(tileN, dtype=ct.int32) + block_x * tileN, tile_x)

@ct.kernel
def ct_rmsnorm_bwd_dy_dx(
    g: ct.Array, x: ct.Array, w: ct.Array, 
    rsqrt: ct.Array, dx: ct.Array,
    tile_size: ct.Constant
):
    block_x = ct.bid(0)
    tile_g = ct.load(g, (block_x, 0), (1, tile_size), allow_tma=False).astype(ct.float32)
    tile_x = ct.load(x, (block_x, 0), (1, tile_size), allow_tma=False).astype(ct.float32)
    tile_w = ct.load(w, (block_x, ), (tile_size, ), allow_tma=False).reshape((1, tile_size)).astype(ct.float32)
    tile_r = ct.load(rsqrt, (block_x, ), (tile_size, ), allow_tma=False).reshape((1, tile_size)).astype(ct.float32)
    
    tile_t = tile_g * tile_w * tile_x
    tile_t = ct.sum(tile_t, axis=1, keepdims=True) * (1 / tile_size)
    tile_dx = tile_g.astype(ct.float32) * tile_w.astype(ct.float32) - tile_t * tile_x.astype(ct.float32)
    tile_dx = tile_dx * tile_r
    ct.store(dx, (block_x, 0), tile_dx, allow_tma=False)
