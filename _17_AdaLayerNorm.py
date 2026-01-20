import cuda.tile as ct
import torch

@ct.function
def modulate(
    tile_x: ct.Tile, tile_shift: ct.Tile, tile_scale: ct.Tile
) -> ct.Tile:
    tile_x = ct.astype(tile_x, ct.float32)
    tile_shift = ct.astype(tile_shift, ct.float32)
    tile_scale = ct.astype(tile_scale, ct.float32)

    return tile_x * (1 + tile_scale) + tile_shift

@ct.function
def layernorm(
    tile_x: ct.Tile, 
    tile_w: ct.Tile, 
    tile_b: ct.Tile, 
    tile_size: ct.Constant,
    normalize_dim: int, eps: float
) -> ct.Tile:
    tile_x = ct.astype(tile_x, ct.float32)
    tile_w = ct.astype(tile_w, ct.float32)
    tile_b = ct.astype(tile_b, ct.float32)

    mask = ct.arange(tile_size, dtype=ct.int32) < normalize_dim
    inv_normalize_dim = (1 / normalize_dim)
    mean = ct.sum(tile_x) * inv_normalize_dim
    tile_x = tile_x - mean

    rsqrt = ct.rsqrt(ct.sum(tile_x * tile_x * mask) * inv_normalize_dim + eps)
    tile_x = tile_x * rsqrt

    return tile_x * tile_w + tile_b


@ct.kernel
def _AdaLayerNorm(
    x: ct.Array, w: ct.Array, b: ct.Array,
    shift: ct.Array, scale: ct.Array, eps: float,
    o: ct.Array, allow_tma: ct.Constant, 
    normalize_dim: int, tile_size: ct.Constant
):
    """
    ### Brief:
    AdaLayerNorm（Adaptive Layer Normalization / Adaptive LayerNorm）是一类“条件化归一化”
    模块：先对输入做标准 LayerNorm（带可学习的 weight/bias），再根据条件输入生成的
    per-sample 调制参数对归一化后的激活做仿射变换：
    
       y = LN(x; w, b)
       o = y * (1 + scale) + shift
    
    该结构最常见的出处是扩散模型/条件生成模型中的 Transformer 变体，尤其是：
    - Stable Diffusion 系列中的 DiT（Diffusion Transformer）使用的 AdaLN/AdaLN-Zero
      思路：用 timestep/class/text 等条件通过 MLP 产生 (shift, scale)，以调制每一层的
      归一化输出，从而将条件信息注入网络。
    - 后续大量 diffusion Transformer（以及部分条件化 Transformer/生成模型）沿用这一范式，
      统称为 AdaLayerNorm / AdaLN / AdaLN-Zero（区别主要在初始化与是否额外引入门控）。
    
    本 kernel 实现的是“LN + 调制(modulation)”的融合版本：对 [batch, seqlen, dim] 的 x，
    先在 dim 维做 LayerNorm（使用 w,b），再使用来自 [batch, dim] 的 shift/scale 做逐样本调制，
    并写回输出 o。ema 用于 layernorm 内部的数值稳定项（等价于 eps/ema 之类的稳定系数）。

    ### FORMULA:
    x = layernorm(x, w, b)
    o = modulate(x, shift, scale)
    return o
    
    ### WHERE:
    x, o: [batch, seqlen, normalize_dim]
    w, b: [normalize_dim]
    scale, shift: [batch, normalize_dim]
    
    ### BLOCK MAPPING:
    3d grid, mapping on o:
    [batch(block_z 1:1), seqlen(block_y 1:1), dim(block_x 1:tile_size)]
    """
    block_x, block_y, block_z = ct.bid(2), ct.bid(1), ct.bid(0)

    tile_x = ct.load(x, (block_z, block_y, block_x), (1, 1, tile_size), allow_tma=allow_tma, padding_mode=ct.PaddingMode.ZERO)
    tile_x = tile_x.reshape((tile_size, ))
    tile_w = ct.load(w, (block_x, ), (tile_size, ), allow_tma=False)
    tile_b = ct.load(b, (block_x, ), (tile_size, ), allow_tma=False, padding_mode=ct.PaddingMode.ZERO)

    tile_x = layernorm(tile_x, tile_w, tile_b, tile_size, normalize_dim, eps)
    tile_shift = ct.load(shift, (block_z, block_x), (1, tile_size), allow_tma=False)
    tile_scale = ct.load(scale, (block_z, block_x), (1, tile_size), allow_tma=False)
    tile_shift = tile_shift.reshape((tile_size, ))
    tile_scale = tile_scale.reshape((tile_size, ))

    tile_x = modulate(tile_x, tile_shift, tile_scale)
    tile_x = tile_x.reshape((1, 1, tile_size))
    ct.store(o, (block_z, block_y, block_x), tile_x.astype(o.dtype), allow_tma=allow_tma)


def AdaLayerNorm(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor,
    shift: torch.Tensor, scale: torch.Tensor, eps: float
) -> torch.Tensor:
    """
    ### Brief:
    AdaLayerNorm（Adaptive Layer Normalization / Adaptive LayerNorm）是一类“条件化归一化”
    模块：先对输入做标准 LayerNorm（带可学习的 weight/bias），再根据条件输入生成的
    per-sample 调制参数对归一化后的激活做仿射变换：
    
       y = LN(x; w, b)
       o = y * (1 + scale) + shift
    """
    if not x.is_cuda: raise ValueError("AdaLayerNorm: x must be a CUDA tensor")
    if not w.is_cuda: raise ValueError("AdaLayerNorm: w must be a CUDA tensor")
    if not b.is_cuda: raise ValueError("AdaLayerNorm: b must be a CUDA tensor")
    if not shift.is_cuda: raise ValueError("AdaLayerNorm: shift must be a CUDA tensor")
    if not scale.is_cuda: raise ValueError("AdaLayerNorm: scale must be a CUDA tensor")

    if not x.is_contiguous(): raise ValueError("AdaLayerNorm: x must be contiguous")
    if not w.is_contiguous(): raise ValueError("AdaLayerNorm: w must be contiguous")
    if not b.is_contiguous(): raise ValueError("AdaLayerNorm: b must be contiguous")
    if not shift.is_contiguous(): raise ValueError("AdaLayerNorm: shift must be contiguous")
    if not scale.is_contiguous(): raise ValueError("AdaLayerNorm: scale must be contiguous")

    if x.dim() != 3: raise ValueError("AdaLayerNorm: x must have shape [batch, seqlen, dim]")
    if w.dim() != 1: raise ValueError("AdaLayerNorm: w must have shape [dim]")
    if b.dim() != 1: raise ValueError("AdaLayerNorm: b must have shape [dim]")
    if shift.dim() != 2: raise ValueError("AdaLayerNorm: shift must have shape [batch, dim]")
    if scale.dim() != 2: raise ValueError("AdaLayerNorm: scale must have shape [batch, dim]")

    o = torch.empty_like(x)
    batch, seqlen, dim = x.shape
    tile_size = 4096
    ct.launch(
        torch.cuda.current_stream(),
        (batch, seqlen, ct.cdiv(dim, tile_size)), 
        _AdaLayerNorm, (x, w, b, shift, scale, eps, o, False, dim, tile_size)
    )
    return o

if __name__ == "__main__":
    batch, seqlen, dim = 1, 49600, 3072
    x = torch.rand(size=[batch, seqlen, dim], dtype=torch.float32, device="cuda")
    w = torch.rand(size=[dim], dtype=torch.float32, device="cuda")
    b = torch.rand(size=[dim], dtype=torch.float32, device="cuda")
    shift = torch.rand(size=[batch, dim], dtype=torch.float32, device="cuda")
    scale = torch.rand(size=[batch, dim], dtype=torch.float32, device="cuda")

    # reference:
    for i in range(5):
        ref = torch.nn.functional.layer_norm(x, (dim, ), w, b, eps=1e-7)
        ref = ref * (1 + scale.reshape(batch, 1, dim)) + shift.reshape(batch, 1, dim)

        pred = AdaLayerNorm(x, w, b, shift, scale, 1e-7)
    print(pred - ref)
