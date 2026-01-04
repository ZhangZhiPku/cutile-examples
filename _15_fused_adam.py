import cuda.tile as ct
import torch

@ct.kernel
def ct_fused_adamw(
    w: ct.Array, g: ct.Array, m: ct.Array, v: ct.Array, lr: float,
    beta1: float=0.9, beta2: float=0.999, eps: float = 1e-8, decay: float = 0.001,
    tile_size: ct.Constant = 1024, allow_tma: bool = False
):
    block_x = ct.bid(0)
    
    tile_g = ct.load(g, (block_x, ), (tile_size, )).astype(ct.float32)
    tile_m = ct.load(m, (block_x, ), (tile_size, )).astype(ct.float32)
    tile_v = ct.load(v, (block_x, ), (tile_size, )).astype(ct.float32)
    
    tile_m = beta1 * tile_m + (1 - beta1) * tile_g
    tile_v = beta2 * tile_m + (1 - beta2) * (tile_g * tile_g)
    
    ct.store(m, (block_x, ), tile_m.astype(m.dtype))
    ct.store(v, (block_x, ), tile_v.astype(v.dtype))
    
    # ref: https://docs.pytorch.org/docs/2.6/generated/torch.optim.AdamW.html
    term_m = 1 / (1 - beta1)
    tile_m = tile_m * term_m
    term_v = 1 / (1 - beta2)
    tile_v = tile_v * term_v
    
    tile_w = ct.load(w, (block_x, ), (tile_size, )).astype(ct.float32)
    tile_w = tile_w - lr * (tile_m * ct.rsqrt(term_v + eps) - tile_w * decay)

    ct.store(w, (block_x, ), tile_w.astype(w.dtype))

def fused_adamw(
    w: torch.Tensor, g: torch.Tensor, m: torch.Tensor, v: torch.Tensor, lr: float,
    beta1: float=0.9, beta2: float=0.999, eps: float = 1e-8, decay: float = 0.001,
    tile_size: ct.Constant = 1024, allow_tma: bool = False
):
    ct.launch(
        torch.cuda.current_stream(), (ct.cdiv(w.numel(), tile_size)), ct_fused_adamw, 
        (w, g, m, v, lr, beta1, beta2, eps, decay, tile_size, allow_tma)
    )