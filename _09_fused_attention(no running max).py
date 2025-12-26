import cuda.tile as ct
import torch
import math

INV_LOG_2 = 1.0 / math.log(2)

@ct.kernel
def flash_attention(
    Q: ct.Array, K: ct.Array, V: ct.Array, O: ct.Array, kv_len: int,
    head_dim: ct.Constant, tileKV: ct.Constant, tileQ: ct.Constant
):
    # Q, O: [q_len, num_head, head_dim]
    # K, V: [kv_len, num_head, head_dim]
    q_idx, head_idx = ct.bid(0), ct.bid(1)

    num_tile_kv = ct.cdiv(kv_len, tileKV)
    tile_q = ct.load(Q, (q_idx, head_idx, 0), (tileQ, 1, head_dim), padding_mode=ct.PaddingMode.ZERO)
    tile_q = tile_q * INV_LOG_2
    
    # online softmax
    accumulator = ct.full((tileQ, head_dim), 0.0, dtype=ct.float32)
    logits_exp_sum = ct.full(shape=(tileQ, 1), fill_value=0.0, dtype=ct.float32)

    for kv_iter in range(num_tile_kv):
        logits = ct.full(shape=(tileQ, tileKV), fill_value=0.0, dtype=ct.float32)
        # QK gemm
        tile_k = ct.load(K, (kv_iter, head_idx, 0), 
                         (tileKV, 1, head_dim), order="C", 
                         padding_mode=ct.PaddingMode.ZERO)
        tile_k = ct.transpose(tile_k, axis0=0, axis1=2)
        logits = ct.mma(
            ct.reshape(tile_q, (tileQ, head_dim)), 
            ct.reshape(tile_k, (head_dim, tileKV)), 
            logits
        )
        
        # mask out of range kv sequence
        position = ct.arange(tileKV, dtype=ct.int32) + (kv_iter * tileKV)
        mask = ct.where(position < kv_len, 0, -10000000.0) # TODO make it -inf
        logits = logits + mask.reshape((1, tileKV))
        
        logits = ct.exp2(logits, flush_to_zero=True)
        logits_exp_sum += ct.sum(logits, axis=1, keepdims=True)

        tile_v = ct.load(
            V, (kv_iter, head_idx, 0), (tileKV, 1, head_dim), 
            padding_mode=ct.PaddingMode.ZERO
        )
        accumulator = ct.mma(
            logits.astype(tile_v.dtype), 
            ct.reshape(tile_v, (tileKV, head_dim)), 
            accumulator
        )
    
    accumulator = ct.truediv(
        accumulator, logits_exp_sum, 
        flush_to_zero=True, rounding_mode=ct.RoundingMode.APPROX
    )
    accumulator = ct.reshape(accumulator, (tileQ, 1, head_dim))
    ct.store(O, (q_idx, head_idx, 0), accumulator.astype(O.dtype))

num_heads, q_len, kv_len, head_dim = 32, 423, 5356, 64
tileQ, tileKV = 64, 128
Q = torch.randn(size=[q_len, num_heads, head_dim], device="cuda", dtype=torch.bfloat16)
K = torch.randn(size=[kv_len, num_heads, head_dim], device="cuda", dtype=torch.bfloat16)
V = torch.randn(size=[kv_len, num_heads, head_dim], device="cuda", dtype=torch.bfloat16)
O = torch.empty(size=[q_len, num_heads, head_dim], device="cuda", dtype=torch.bfloat16)

ct.launch(
    torch.cuda.current_stream(), 
    (ct.cdiv(q_len, tileQ), num_heads), 
    flash_attention, 
    (Q, K, V, O, kv_len, head_dim, tileKV, tileQ)
)

print(O)