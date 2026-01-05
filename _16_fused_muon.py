import torch
import cuda.tile as ct

@ct.kernel
def symm_matmul(
    X: ct.Array, O: ct.Array, alpha: float,
    tileMN: ct.Constant, tileK: ct.Constant
):
    # O = alpha * X @ X.T
    
    K = X.shape[-1]
    block_x, block_y = ct.bid(0), ct.bid(1)
    
    # 只计算上三角，其他地方不用算
    if block_y > block_x: return
    
    num_tile_k = ct.cdiv(K, tileK)
    accumulator = ct.full((tileMN, tileMN), 0.0, dtype=ct.float32)

    for k in range(num_tile_k):

        tile_x = ct.load(X, (block_x, k), (tileMN, tileK))
        tile_t = ct.load(X, (k, block_y), (tileK, tileMN), order="F")

        accumulator = ct.mma(tile_x, tile_t, accumulator)
    
    accumulator = accumulator * alpha
    accumulator = accumulator.astype(O.dtype)
    ct.store(O, (block_x, block_y), accumulator)
    if block_x != block_y:
        ct.store(O, (block_y, block_x), accumulator.transpose(0, 1))

@ct.kernel
def symm_matmul_bias(
    X: ct.Array, Y: ct.Array, O: ct.Array, 
    alpha: float, beta: float, 
    tileMN: ct.Constant, tileK: ct.Constant
):
    # O = alpha * X @ X.T + beta * Y
    
    K = X.shape[-1]
    block_x, block_y = ct.bid(0), ct.bid(1)
    
    # 只计算上三角，其他地方不用算
    if block_y > block_x: return
    
    num_tile_k = ct.cdiv(K, tileK)
    accumulator = ct.load(Y, (block_x, block_y), (tileMN, tileMN)).astype(ct.float32)
    accumulator = accumulator * beta / alpha

    for k in range(num_tile_k):

        tile_x = ct.load(X, (block_x, k), (tileMN, tileK))
        tile_t = ct.load(X, (k, block_y), (tileK, tileMN), order="F")

        accumulator = ct.mma(tile_x, tile_t, accumulator)
    
    accumulator = accumulator * alpha
    accumulator = accumulator.astype(O.dtype)
    ct.store(O, (block_x, block_y), accumulator)
    if block_x != block_y:
        ct.store(O, (block_y, block_x), accumulator.transpose(0, 1))


def muon_iteration(
    X: torch.Tensor, a: float, b: float, c: float, steps: int,
    tileMN: int = 64, tileK: int = 128
):
    assert X.ndim == 2
    M, K = X.shape

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for i in range(steps):
        # Y = b * X @ X.T
        Y = torch.empty(size=(M, M), dtype=X.dtype, device=X.device)
        ct.launch(
            torch.cuda.current_stream(),
            (ct.cdiv(M, tileMN), ct.cdiv(M, tileMN), 1),
            symm_matmul, (X, Y, b, tileMN, tileK)
        )
        # Z = c / b * Y @ Y.T + a * I
        Z = torch.eye(n=M, device="cuda", dtype=X.dtype)
        ct.launch(
            torch.cuda.current_stream(),
            (ct.cdiv(M, tileMN), ct.cdiv(M, tileMN), 1),
            symm_matmul_bias, (Y, Z, Z, c / b / b, a, tileMN, tileK)
        )
        X = (Z + Y) @ X
    return X