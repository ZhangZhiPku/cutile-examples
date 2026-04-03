import torch
import cuda.tile as ct
import math
from typing import Optional

@ct.kernel
def mla_splited_kernel(
    q_nope: ct.Array,
    q_rope: ct.Array,
    k_nope: ct.Array,
    k_rope: ct.Array,
    block_table: ct.Array,
    cu_seqlens: ct.Array,
    o: ct.Array,
    lse: ct.Array,
    scale,
    N_SPLITS: ct.Constant,
    BLOCK_SIZE: ct.Constant,
    KV_LORA_RANK: ct.Constant,
    ROPE_DIM: ct.Constant,
    TILE_H: ct.Constant,
    TILE_N: ct.Constant
):
    block_head_idx = ct.bid(0)
    block_batch_idx = ct.bid(1)
    block_splits = ct.bid(2)
    
    cur_batch_seqlen = ct.load(cu_seqlens, (block_batch_idx, ), (1, ))
    kv_block_num = ct.cdiv(cur_batch_seqlen, TILE_N)
    kv_block_per_split = ct.cdiv(kv_block_num, N_SPLITS)
    
    # [1, TILE_H, KV_LORA_RANK]
    q_nope_tile = ct.load(q_nope, (block_batch_idx, block_head_idx, 0), (1, TILE_H, KV_LORA_RANK), padding_mode=ct.PaddingMode.ZERO).astype(ct.float32)
    # [1, TILE_H, ROPE_DIM]
    q_rope_tile = ct.load(q_rope, (block_batch_idx, block_head_idx, 0), (1, TILE_H, ROPE_DIM), padding_mode=ct.PaddingMode.ZERO).astype(ct.float32)
    
    kv_block_start_idx = block_splits * kv_block_per_split
    kv_block_end_idx = ct.minimum(kv_block_start_idx + kv_block_per_split, kv_block_num)
    
    m_i = ct.full((TILE_H), fill_value=float("-inf"), dtype=ct.float32)
    l_i = ct.full((TILE_H), fill_value=0.0, dtype=ct.float32)
    
    acc = ct.zeros((TILE_H, KV_LORA_RANK), dtype=ct.float32)
    
    
    for kv_block_idx in range(kv_block_start_idx.item(), kv_block_end_idx.item()):
        physical_kv_block_idx = ct.load(block_table, (block_batch_idx, kv_block_idx), (1,1))
        physical_kv_block_idx = physical_kv_block_idx.reshape((1, )).item()
        
        # [TILE_N, KV_LORA_RANK]
        k_nope_tile = ct.load(k_nope, (physical_kv_block_idx, 0), (TILE_N, KV_LORA_RANK), padding_mode=ct.PaddingMode.ZERO).astype(ct.float32)
        # [TILE_N, ROPE_DIM]
        k_rope_tile = ct.load(k_rope, (physical_kv_block_idx, 0), (TILE_N, ROPE_DIM), padding_mode=ct.PaddingMode.ZERO).astype(ct.float32)
        
        # qk: [TILE_H, TILE_N]
        qk = q_nope_tile.reshape((TILE_H, KV_LORA_RANK)) @ k_nope_tile.transpose(0, 1) + q_rope_tile.reshape((TILE_H, ROPE_DIM)) @ k_rope_tile.transpose(0, 1)
        qk *= scale
        
        mask_n = kv_block_idx * TILE_N + ct.arange(TILE_N, dtype=ct.int32) < cur_batch_seqlen # [TILE_N]
        qk = ct.where(mask_n.reshape((1, TILE_N)), qk, float("-inf"))
        
        m_i_j = ct.max(qk, axis=1) # [TILE_H]
        m_i_new = ct.maximum(m_i, m_i_j) # [TILE_H]
        
        alpha = ct.exp2(m_i - m_i_new) # [TILE_H]
        
        qk = ct.exp2(qk - m_i_new.reshape((TILE_H, 1)))
        acc *= alpha.reshape((TILE_H, 1))
        
        acc = ct.mma(qk, k_nope_tile, acc=acc)
        l_i = l_i * alpha + ct.sum(qk, axis=1) # [TILE_H]
        
        m_i = m_i_new
        
    acc = acc / l_i.reshape((TILE_H, 1))
    acc = acc.reshape((1, TILE_H, 1, KV_LORA_RANK))
    l_i = ct.log2(l_i) + m_i
    
    ct.store(o, (block_batch_idx, block_head_idx, block_splits, 0), acc.astype(o.dtype))
    ct.store(lse, (block_batch_idx, block_head_idx, block_splits), l_i.reshape((1, TILE_H, 1)).astype(lse.dtype))
    
    
def mla_splited(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    o_splited: torch.Tensor,
    lse_splited: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens: torch.Tensor,
    n_splits: int,
    scale: float,
    block_size: int,
    causal: bool = False,
):
    bs_l, num_heads, kv_lora_rank = q_nope.shape
    _, _, q_rope_dim = q_rope.shape
    _, max_num_blocks = block_table.shape
    
    TILE_H = 2
    TILE_N = block_size
    
    grid = (ct.cdiv(num_heads, TILE_H), bs_l, n_splits)
    ct.launch(torch.cuda.current_stream(), grid, mla_splited_kernel, 
              (
                  q_nope,
                  q_rope,
                  k_nope,
                  k_rope,
                  block_table,
                  cu_seqlens,
                  o_splited,
                  lse_splited,
                  scale,
                  n_splits,
                  block_size,
                  kv_lora_rank,
                  q_rope_dim,
                  TILE_H,
                  TILE_N
              ))
@ct.kernel
def merging_attention_states_kernel(
    o_splited: ct.Array,
    lse_splited: ct.Array,
    o: ct.Array,
    lse: ct.Array,
    cu_seqlens: ct.Array,
    N_SPLITS: ct.Constant,
    HEAD_DIM: ct.Constant
):
    block_batch_idx = ct.bid(1)
    block_head_idx = ct.bid(0)
    
    cur_batch_seqlen = ct.load(cu_seqlens, (block_batch_idx, ), (1, ))
    m = ct.full((1, ), fill_value=-float("inf"), dtype=ct.float32)
    se = ct.zeros((1, ), dtype=ct.float32)
    
    acc = ct.zeros((HEAD_DIM, ), dtype=ct.float32)
    
    for split_idx in range(N_SPLITS):
        kv_per_split = ct.cdiv(cur_batch_seqlen, N_SPLITS)
        start = split_idx * kv_per_split
        end = ct.minimum(start + kv_per_split, cur_batch_seqlen)
        
        if start.item() < end.item(): # pass empty splits
            o_i = ct.load(o_splited, (block_batch_idx, block_head_idx, split_idx, 0), (1, 1, 1, HEAD_DIM), padding_mode=ct.PaddingMode.ZERO)
            lse_i = ct.load(lse_splited, (block_batch_idx, block_head_idx, split_idx), (1, 1, 1)).reshape((1, ))
            m_new = ct.maximum(m, lse_i)
            
            acc *= ct.exp2(m - m_new)
            scale_i = ct.exp2(lse_i - m_new)
            acc = acc + o_i.reshape((HEAD_DIM, )) * scale_i.reshape((1, ))
            
            se = se * ct.exp2(m - m_new) + scale_i
            m = m_new
            
    acc = acc / se
    acc = acc.reshape((1, 1, HEAD_DIM))
    se = ct.log2(se) + m
    se = se.reshape((1, 1))
    
    ct.store(o, (block_batch_idx, block_head_idx, 0), acc.astype(o.dtype))
    ct.store(lse, (block_batch_idx, block_head_idx), se.astype(lse.dtype))
            
def merging_attention_states(
    o_splited,
    lse_splited,
    o,
    lse,
    cu_seqlens,
):
    bs_l, num_heads, n_splits, kv_lora_rank = o_splited.shape
    
    grid = (num_heads, bs_l)
    ct.launch(torch.cuda.current_stream(), grid, merging_attention_states_kernel, 
              (
                  o_splited,
                  lse_splited,
                  o,
                  lse,
                  cu_seqlens,
                  n_splits,
                  kv_lora_rank
              ))

def mla_decode_cutile(
    q_absorb: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens: torch.Tensor,
    kv_lora_rank: int,
    n_splits: int,
    scale: Optional[float] = None,
    causal: Optional[bool] = None,
):
    """
    Argumenst:
        q_absorb: (bs, 1, num_heads_q, kv_lora_rank + q_rope)
        k_cache: (num_blocks, block_size, kv_lora_rank + q_rope)
        block_table: (bs, max_num_blocks)
        cu_seqlens: (bs, )
        kv_lora_rank: dimension of latent v
        n_splits: number of splits for kv cache
    Out:
        out: (bs, 1, num_heads_q, v_head_dim)
        softmax_lse: (bs, 1, numheads_q)
    """
    if scale is None:
        scale = q_absorb.shape[-1] ** (-0.5)
    scale = scale * 1 / math.log(2) # exp(x) = exp2(x / log2)
    
    bs, seq_len_q, num_heads, head_dim_q = q_absorb.shape
    num_blocks, block_size, _ = k_cache.shape
    
    q_nope = q_absorb[..., :kv_lora_rank].view(bs * seq_len_q, num_heads, kv_lora_rank) # [bs*seq_len_q, num_heads, kv_lora_rank]
    q_rope = q_absorb[..., kv_lora_rank:].view(bs * seq_len_q, num_heads, head_dim_q - kv_lora_rank) # [bs*seq_len_q, num_heads, q_rope_dim]
    k_nope = k_cache[..., :kv_lora_rank].view(-1, kv_lora_rank) # [num_blocks*block_size, kv_lora_rank]
    k_rope = k_cache[..., kv_lora_rank:].view(-1, head_dim_q - kv_lora_rank) # [num_blocks*block_size, q_rope_dim]
    
    o = torch.empty([bs * seq_len_q, num_heads, kv_lora_rank], dtype=q_absorb.dtype, device=q_absorb.device)
    o_splited = torch.empty([bs * seq_len_q, num_heads, n_splits, kv_lora_rank], dtype=q_absorb.dtype, device=q_absorb.device)
    lse = torch.empty([bs * seq_len_q, num_heads], dtype=torch.float32, device=q_absorb.device)
    lse_splited = torch.empty([bs * seq_len_q, num_heads, n_splits], dtype=torch.float32, device=q_absorb.device)
    
    mla_splited(
        q_nope,
        q_rope,
        k_nope,
        k_rope,
        o_splited,
        lse_splited,
        block_table,
        cu_seqlens,
        n_splits,
        scale,
        block_size,
        causal,
    )
    merging_attention_states(
        o_splited,
        lse_splited,
        o,
        lse,
        cu_seqlens,
    )
    
    return o, lse
    
    
if __name__ == "__main__":
    print("🚀 初始化 MLA Split-K 测试环境...")

    # ==========================================
    # 1. 配置模型和硬件参数 (模拟 DeepSeek MLA 配置)
    # ==========================================
    bs = 2                   # Batch size (同时处理两个句子的生成)
    seq_len_q = 1            # 解码阶段，每次 Q 只生成 1 个 Token
    num_heads = 4            # 注意力头数 (为方便测试，设小一点)
    kv_lora_rank = 512       # 潜向量维度 (c_kv 的维度)
    q_rope_dim = 64          # RoPE 旋转位置编码维度
    
    # 物理显存分配参数
    block_size = 64          # PagedAttention 每个块存放的 Token 数 (TILE_N)
    num_blocks = 128         # 显存池中总共预先分配的物理块数量
    max_num_blocks = 32      # 单个序列最多能占用多少个块 (block_table 的宽度)
    
    # 算子调度参数
    n_splits = 1             # 将超长 KV Cache 切分为 4 份并行计算
    head_dim_q = kv_lora_rank + q_rope_dim  # Q 的总维度：512 + 64 = 576

    # ==========================================
    # 2. 构造随机张量 (分配到 CUDA 并转换为 fp16)
    # ==========================================
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float16

    # Q 向量：已经过外层矩阵吸收，包含 nope 和 rope 两部分
    q_absorb = torch.randn((bs, seq_len_q, num_heads, head_dim_q), dtype=dtype, device=device)

    # 物理 KV Cache 池：模拟已经存满的历史潜向量和 RoPE
    k_cache = torch.randn((num_blocks, block_size, head_dim_q), dtype=dtype, device=device)

    # ==========================================
    # 3. 构造分页地址表 (PagedAttention 核心)
    # ==========================================
    # 随机为每个 sequence 分配一些物理块索引
    block_table = torch.randint(0, num_blocks, (bs, max_num_blocks), dtype=torch.int32, device=device)
    
    # 模拟真实序列长度：假设句子 1 长度 500，句子 2 长度 1200
    cu_seqlens = torch.tensor([500, 1200], dtype=torch.int32, device=device)

    # ==========================================
    # 4. 执行手写算子
    # ==========================================
    print(f"📦 测试参数:")
    print(f" - Batch Size: {bs}")
    print(f" - Context Lengths: {cu_seqlens.tolist()}")
    print(f" - KV_LORA_RANK (潜向量): {kv_lora_rank}")
    print(f" - ROPE_DIM (位置编码): {q_rope_dim}")
    print(f" - N_SPLITS (并行切分): {n_splits}")
    print("-" * 40)
    
    print("⏳ 正在启动 cuda.tile Kernel...")
    try:
        out, lse = mla_decode_cutile(
            q_absorb=q_absorb,
            k_cache=k_cache,
            block_table=block_table,
            cu_seqlens=cu_seqlens,
            kv_lora_rank=kv_lora_rank,
            n_splits=n_splits
        )
        
        # ==========================================
        # 5. 验证输出
        # ==========================================
        print("✅ 算子执行成功！")
        print(f"📊 输出特征 O 的形状: {out.shape}   (期望: [{bs}, {num_heads}, {kv_lora_rank}])")
        print(f"📊 输出 LSE 的形状: {lse.shape}       (期望: [{bs}, {num_heads}])")
        
        # 打印部分结果检查是否出现 NaN
        print("\n🔍 输出 O 的前五个数值 (Batch 0, Head 0):")
        print(out[0, 0, :5])
        
        if torch.isnan(out).any() or torch.isnan(lse).any():
            print("⚠️ 警告：输出包含 NaN 值，请检查空 Split 的防御逻辑或初始化的负无穷逻辑！")
        else:
            print("🎉 恭喜！结果数值稳定，无 NaN 污染！")
            
    except Exception as e:
        print(f"❌ 运行报错: {e}")