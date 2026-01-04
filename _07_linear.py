import cuda.tile as ct
import torch


@ct.kernel
def ct_matmul(
    A: ct.Array, B: ct.Array, o: ct.Array,
    transposeA: ct.Constant, transposeB: ct.Constant,
    tileM: ct.Constant, tileN: ct.Constant, tileK: ct.Constant
):
    k = A.shape[0] if transposeA else A.shape[-1]
    block_x, block_y = ct.bid(0), ct.bid(1)
    num_tile_k = ct.cdiv(k, tileK)
    accumulator = ct.full((tileM, tileN), 0, dtype=ct.float32)
    
    for k_iter in range(num_tile_k):
        tileA = ct.load(
            A, (block_x, k_iter), (tileM, tileK), 
            padding_mode=ct.PaddingMode.ZERO, order="F" if transposeA else "C")
        tileB = ct.load(
            B, (k_iter, block_y), (tileK, tileN),
            padding_mode=ct.PaddingMode.ZERO, order="F" if transposeB else "C")
        accumulator = ct.mma(tileA, tileB, accumulator)
    
    accumulator = ct.astype(accumulator, o.dtype)
    ct.store(o, (block_x, block_y), accumulator)


@ct.kernel
def ct_pertensor_quantized_matmul_fp8(
    A: ct.Array, B: ct.Array, o: ct.Array, sA: ct.Array, sB: ct.Array,
    transposeA: ct.Constant, transposeB: ct.Constant,
    tileM: ct.Constant, tileN: ct.Constant, tileK: ct.Constant
):
    k = A.shape[0] if transposeA else A.shape[-1]
    block_x, block_y = ct.bid(0), ct.bid(1)
    num_tile_k = ct.cdiv(k, tileK)
    accumulator = ct.full((tileM, tileN), 0, dtype=ct.float32)
    tileSA = ct.load(sA, (0, ), (1, ), allow_tma=False).item()
    tileSB = ct.load(sB, (0, ), (1, ), allow_tma=False).item()
    
    for k_iter in range(num_tile_k):
        tileA = ct.load(
            A, (block_x, k_iter), (tileM, tileK), 
            padding_mode=ct.PaddingMode.ZERO, order="F" if transposeA else "C")
        tileB = ct.load(
            B, (k_iter, block_y), (tileK, tileN),
            padding_mode=ct.PaddingMode.ZERO, order="F" if transposeB else "C")
        accumulator = ct.mma(tileA, tileB, accumulator)
    
    accumulator = accumulator * tileSA * tileSB
    ct.store(o, (block_x, block_y), accumulator.astype(o.dtype))
    