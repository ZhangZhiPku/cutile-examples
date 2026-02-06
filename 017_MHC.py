import torch
import math
import cuda.tile as ct

INV_LOG_2 = 1.0 / math.log(2)
@ct.function
def sigmoid_exp2(tile: ct.Tile):
    return 1 / (1 + ct.exp2(tile))

@ct.function
def sinkhorn_exp2(tile: ct.Tile, iter: int=20):
    tile = ct.exp2(tile)
    for i in range(iter):
        tile = tile / ct.sum(tile, axis=-1, keepdims=True)
        tile = tile / ct.sum(tile, axis=-2, keepdims=True)
    return tile

@ct.kernel
def FusedRmsNormSplitKGemm_N32Stream4(
    X, # [M, K] 
    W, # [K, N]
    O, # [k, M, 32]
    TileM: ct.Constant, 
    TileK: ct.Constant,
    SplitedK: ct.Constant
):
    """
    ### 融合RMSNorm+SplitK GEMM核，N维度固定32、Stream数固定4
    Tile划分：M维度按TileM切分，K维度先按SplitedK切分子块再按TileK切分Tile，N维度固定TileN=32
    功能：完成X@W的SplitK GEMM计算，同时融合计算RMSNorm所需的平方和，结果存入工作空间O
    
    :param X: [M, K] 输入特征矩阵
    :param W: [K, N] 权重矩阵
    :param O: [k, M, 32] 输出工作空间（k=K/SplitedK向上取整）
    :param TileM: 编译期常量，M/K维度Tile大小
    :param TileK: 编译期常量，M/K维度Tile大小
    :param SplitedK: 编译期常量，K维度SplitK分块大小
    """
    
    TileN: ct.Constant = 32
    Stream: ct.Constant = 4

    NumTileK: ct.Constant = ct.cdiv(SplitedK, TileK)
    bid_x, bid_y = ct.bid(0), ct.bid(1)

    accmulator = ct.full(shape=(TileM, TileN), fill_value=0.0, dtype=ct.float32)
    tile_sqaure_sum = ct.full(shape=(TileM, 1), fill_value=0.0, dtype=ct.float32)
    for iter_k in range(NumTileK):
        tile_x = ct.load(
            X, index=(bid_x, bid_y * NumTileK + iter_k), 
            shape=(TileM, TileK), padding_mode=ct.PaddingMode.ZERO, 
            allow_tma=True, latency=8
        )

        tile_w = ct.load(
            W, index=(bid_y * NumTileK + iter_k, 0), 
            shape=(TileK, TileN), padding_mode=ct.PaddingMode.ZERO, 
            allow_tma=True, latency=8
        )

        accmulator = ct.mma(tile_x, tile_w, accmulator)
        
        # rmsnorm
        tile_x = tile_x.astype(ct.float32)
        tile_sqaure_sum += ct.sum(tile_x * tile_x, axis=1, keepdims=True)
    
    ct.store(O, (bid_y, bid_x, 0), accmulator.reshape((1, TileM, TileN)))
    ct.store(O, (bid_y, bid_x, Stream*2+Stream*Stream), tile_sqaure_sum.reshape((1, TileM, 1)))

@ct.kernel
def FusedFinalizeSplitK_N32Stream4(
    X,   # [k, M, 32]
    AlphaBeta, # [4+4+4*4+1+1+1]
    H_pre, H_res, H_pos, 
    iter: int, TileM: ct.Constant
):
    """
    ### SplitK GEMM结果收尾核，N固定32、Stream数固定4，融合激活/缩放/Sinkhorn归一化
    Tile划分：M维度按TileM切分
    功能：聚合SplitK GEMM的分布式结果，依次做缩放、sigmoid_exp2激活、Sinkhorn归一化，生成三类特征变换矩阵
    
    :param X: [k, M, 32] SplitK GEMM输出的工作空间张量
    :param AlphaBeta: [25] 融合的缩放/偏置超参张量（4+4+16+1+1+1=25）
        需要依次拼接 beta_res, beta_pre, beta_pos, alpha_res, alpha_pre, alpha_pos
    :param H_pre: [M, 4] 输出预处理特征变换矩阵
    :param H_res: [M, 4, 4] 输出残差特征变换矩阵
    :param H_pos: [M, 4] 输出位置特征变换矩阵
    :param iter: Sinkhorn 归一化的迭代次数
    :param TileM: 编译期常量，M维度的Tile切分大小
    """
    TileN: ct.Constant = 32
    Stream: ct.Constant = 4

    bid_x = ct.bid(0)
    raw = ct.load(AlphaBeta, (0, ), (TileN,), allow_tma=False, latency=1)
    
    Offset = Stream * 2 + Stream * Stream
    alpha_res = ct.extract((raw), (Offset), (1, ))
    alpha_pre = ct.extract((raw), (Offset+1), (1, ))
    alpha_pos = ct.extract((raw), (Offset+2), (1, ))
    beta_res = ct.extract((raw), (0), (Stream * Stream, )).reshape((1, Stream, Stream))
    beta_pre = ct.extract((raw), (Stream), (Stream, )).reshape((1, Stream))
    beta_pos = ct.extract((raw), (Stream + 1), (Stream, )).reshape((1, Stream))
    k = X.shape[0]

    x_acc = ct.full((1, TileM, TileN), 0.0, dtype=ct.float32)
    for iter_k in range(k):
        tile_x = ct.load(X, (iter_k, bid_x, 0), (1, TileM, TileN), allow_tma=False)
        x_acc += tile_x

    x_acc = INV_LOG_2 * x_acc.reshape((TileM, TileN))
    tile_res = ct.extract(x_acc, (0, 0), (TileM, Stream * Stream))
    tile_pre = ct.extract(x_acc, (0, Stream), (TileM, Stream))
    tile_pos = ct.extract(x_acc, (0, Stream + 1), (TileM, Stream))

    tile_pre = 1 * alpha_pre * sigmoid_exp2(tile_pre) + beta_pre
    tile_pos = 2 * alpha_pos * sigmoid_exp2(tile_pre) + beta_pos
    ct.store(H_pre, (bid_x, 0), tile_pre, allow_tma=False)
    ct.store(H_pos, (bid_x, 0), tile_pos, allow_tma=False)

    tile_res = sinkhorn_exp2(alpha_res * tile_res.reshape((TileM, Stream, Stream)) + beta_res, iter)
    ct.store(H_res, (bid_x, 0, 0), tile_res, allow_tma=False)

@ct.kernel
def ApplyResidual_Stream4(
    X_pos, # [M, K // 4]
    X_res, # [M, K]
    H_pos, # [M, 4, 4]
    H_res, # [M, 4]
    O, # [M, K]
    tile_size: ct.Constant
):
    """
    ### 残差融合应用核，Stream 数固定 4，融合位置特征与残差特征
    Tile划分：K维度按tile_size切分，每个 Tile 包含 tile_size * Stream 个特征，M 维度按整行切分
    功能：将H_res / H_pos变换矩阵分别作用于 X_pos / X_res，融合位置特征与残差特征得到最终输出
    
    :param X_pos: [M, K // 4] 输入位置特征矩阵
    :param X_res: [M, K] 输入残差特征矩阵
    :param H_pos: [M, 4, 4] 位置特征变换矩阵
    :param H_res: [M, 4] 残差特征变换矩阵
    :param O: [M, K] 输出融合后的最终特征矩阵
    :param tile_size: 编译期常量，K维度的Tile切分基础大小
    """
    bid_x, bld_y = ct.bid(0), ct.bid(1)
    Stream: ct.Constant = 4
    
    h_res = ct.load(H_res, (bid_x, 0, 0), (1, Stream, Stream), allow_tma=False, latency=1)
    h_pos = ct.load(H_pos, (bid_x, 0), (1, Stream), allow_tma=False, latency=1)
    
    x_res = ct.load(X_res, (bid_x, bld_y), (1, tile_size * Stream), allow_tma=True, latency=8)
    x_acc = ct.full((Stream, tile_size), 0.0, ct.float32)

    h_res = h_res.reshape((Stream, Stream))
    for i in range(Stream):
        x_acc += h_res.extract((0, i), (Stream, 1)) * x_res.reshape((Stream, tile_size))

    x_pos = ct.load(X_pos, (bid_x, bld_y), (1, tile_size), allow_tma=True, latency=8)
    x_pos = h_pos.reshape((Stream, 1)) * x_pos.reshape((1, tile_size))

    o = x_acc + x_pos
    ct.store(O, (bid_x, bld_y), o.astype(O.dtype), allow_tma=True, latency=8)

@ct.kernel
def ApplyPreTransform_Stream4(
    X: ct.Array, H_pre: ct.Array, O: ct.Array, tile_size: ct.Constant
):
    bid_x, bid_y = ct.bid(0), ct.bid(1)
    tile_x = ct.load(X, (bid_x, bid_y), (1, tile_size))
    tile_h = ct.load(H_pre, (bid_x, 0), (1, 4))
    tile_x = tile_x.reshape((1, 4, tile_size // 4))
    tile_h = tile_h.reshape((1, 4, 1))
    tile_y = ct.sum(tile_x * tile_h, axis=1)
    ct.store(O, (bid_x, bid_y), tile_y.astype(O.dtype))

def ApplyPreTransform(X: torch.Tensor, H_pre: torch.Tensor, tile_size: int=1024):
    M, K = X.shape
    assert K % 4 == 0
    Y = torch.empty(size=[M, K // 4], device=X.device, dtype=X.dtype)

    ct.launch(
        torch.cuda.current_stream(), 
        (M, ct.cdiv(K, tile_size)),
        ApplyPreTransform_Stream4,
        (X, H_pre, Y, tile_size)
    )
    return Y

def FusedRmsNormSplitKGemm(
    X: torch.Tensor, W: torch.Tensor, workspace: torch.Tensor, 
    TileM: int = 128, TileK: int = 128, SplitK: int = 2048
):
    """
    ### FusedRmsNormSplitKGemm 核的 Python 封装接口
    功能：执行融合 RMSNorm 的 SplitK GEMM
    
    :param X: [M, K] 输入特征张量
    :param W: [K, N] 权重张量（N ≤ 32）
    :param workspace: [k, M, 32] 输出工作空间张量（k=K / SplitK向上取整）
    :param TileM: M维度的Tile切分大小，默认128
    :param TileK: K维度的Tile切分大小，默认128
    :param SplitK: K维度的SplitK分块大小，默认2048
    :return: [k, M, 32] 填充后的工作空间张量
    """
    M, K = X.shape
    K_, N = W.shape
    
    assert K == K_
    assert N <= 32
    
    ct.launch(
        torch.cuda.current_stream(), 
        (ct.cdiv(M, TileM), ct.cdiv(K, SplitK)),
        FusedRmsNormSplitKGemm_N32Stream4,
        (X, W, workspace, TileM, TileK, SplitK)
    )
    return workspace

def FusedFinalizeSplitK(
    workspace: torch.Tensor, fused_alpha_beta: torch.Tensor, 
    sinkhorn_iter: int = 20, TileM: int = 16
):
    """
    ### FusedFinalizeSplitK核的Python封装接口
    功能：聚合SplitK GEMM的分布式结果，依次做缩放、sigmoid_exp2激活、Sinkhorn归一化，生成三类特征变换矩阵
    
    :param workspace: [k, M, 32] SplitK GEMM输出的工作空间张量
    :param fused_alpha_beta: [M, N+3] 融合的缩放/偏置超参张量
    :param sinkhorn_iter: Sinkhorn归一化迭代次数，默认20
    :param TileM: M维度的Tile切分大小，默认16
    :return: 元组(H_pre, H_res, H_pos)，分别为[M,4]、[M,4,4]、[M,4]的变换矩阵
    """
    assert workspace.ndim == 3
    _, M, N = workspace.shape
    assert N <= 32
    
    H_pre = torch.empty(size=[M, 4], device=workspace.device, dtype=torch.float32)
    H_res = torch.empty(size=[M, 4, 4], device=workspace.device, dtype=torch.float32)
    H_pos = torch.empty(size=[M, 4], device=workspace.device, dtype=torch.float32)
    
    ct.launch(
        torch.cuda.current_stream(),
        (ct.cdiv(M, TileM), ),
        FusedFinalizeSplitK_N32Stream4,
        (workspace, fused_alpha_beta, H_pre, H_res, H_pos, sinkhorn_iter, TileM)
    )
    return H_pre, H_res, H_pos

def ApplyResidual(
    X_pos: torch.Tensor, X_res: torch.Tensor,
    H_pos: torch.Tensor, H_res: torch.Tensor,
    TileK: int = 2048
):
    """
    ### ApplyResidual_Stream4核的Python封装接口
    功能：将H_res / H_pos变换矩阵分别作用于 X_pos / X_res，融合位置特征与残差特征得到最终输出
    
    :param X_pos: [M, K//4] 输入位置特征张量
    :param X_res: [M, K] 输入残差特征张量
    :param H_pos: [M,4] 位置特征变换矩阵
    :param H_res: [M,4,4] 残差特征变换矩阵
    :param TileK: K维度的Tile切分大小，默认2048
    :return: [M, K] 融合后的最终特征张量
    """
    M, K = X_pos.shape
    M_, K_ = X_res.shape
    assert M == M_
    assert K * 4 == K_
    
    O = torch.empty_like(X_res)
    ct.launch(
        torch.cuda.current_stream(),
        (M, ct.cdiv(K_, TileK)),
        ApplyResidual_Stream4,
        (X_pos, X_res, H_pos, H_res, O, TileK)
    )


class MHCBlock4(torch.nn.Module):
    def __init__(self, dim: int, device="cuda"):
        super().__init__()
        # initialize alpha beta
        alpha = torch.ones(size=[3], device=device)
        beta = torch.zeros(size=[24], device=device)
        self.alpha_beta = torch.cat([beta, alpha], dim=0)
        self.W = torch.randn(size=[dim*4, 32], device=device)
    
    def forward(self, X: torch.Tensor, splitK: int=2048):
        M, K = X.shape

        NumSplitK = ct.cdiv(K, splitK)
        workspace = torch.empty(size=[NumSplitK, M, 32], device="cuda", dtype=torch.float32)

        workspace = FusedRmsNormSplitKGemm(X, self.W, workspace)
        H_pre, H_res, H_pos = FusedFinalizeSplitK(workspace, self.alpha_beta)

        Y = ApplyPreTransform(X, H_pre)
        # Add your logic here:
        #
        #
        #
        # : )
        X = ApplyResidual(Y, X, H_pos, H_res)
        return X

    def reference_logic(self, X: torch.Tensor):
        M, K = X.shape
        
        H = torch.matmul(X, self.W)
        H_res, H_pre, H_pos = H[:, 0:16], H[:, 16:20], H[:, 20:24]
        a_res, a_pre, a_pos = self.alpha_beta[24:25], self.alpha_beta[25:26], self.alpha_beta[26:27]
        b_res, b_pre, b_pos = self.alpha_beta[0:16], self.alpha_beta[16:20], self.alpha_beta[20:24]
        
        H_pre = a_pre * torch.sigmoid(H_pre) + b_pre
        H_res = a_res * H_res + b_res
        H_pos = a_pos * 2 * torch.sigmoid(H_pos) + b_pos
        H_pre = H_pre.reshape(M, 4, 1)
        H_res = H_res.reshape(M, 4, 4)
        H_pos = H_pos.reshape(M, 1, 4)
        
        X = X.reshape(M, 4, -1)
        
        X_res = torch.bmm(H_res, X)
        X_pre = torch.bmm(H_pre, X)
        X_pos = torch.bmm(H_pos, X_pre)
        
        return X_pos + X_res
    

MHC = MHCBlock4(dim=1024)
X = torch.randn(size=[1042, 1024*4], device="cuda", dtype=torch.float32)
pred = MHC(X)
real = MHC.reference_logic(X)
print(pred - real)
