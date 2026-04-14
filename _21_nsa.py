import torch
import cuda.tile as ct
import math

INV_LOG_2 = math.log(2)

def next_power_of_2(n: int) -> int:
    """计算大于或等于 n 的最小的 2 的幂"""
    if n <= 1:
        return 1
    
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16

    return n + 1

@ct.kernel
def compress_attention_fwd_kernel(
        q: ct.Array,
        k: ct.Array,
        v: ct.Array,
        o: ct.Array,
        lse: ct.Array,
        kernel_size: int,
        kernel_stride: int,
        cu_seqlens_q: ct.Array,
        cu_seqlens_k: ct.Array,
        num_k_heads: int,
        num_share_q_heads: int,
        head_dim: int,
        sm_scale: float,
        BLOCK_SIZE_Q: ct.Constant,
        BLOCK_SIZE_K: ct.Constant,
        BLOCK_SIZE_D: ct.Constant,
):
    sm_scale *= INV_LOG_2

    batch_idx = ct.bid(0)
    q_head_idx = ct.bid(1)
    q_seq_idx = ct.bid(2)
    k_head_idx = q_head_idx // num_share_q_heads

    q_start = ct.load(cu_seqlens_q, (batch_idx,), (1,))
    q_end = ct.load(cu_seqlens_q, (batch_idx + 1,), (1,))
    q_len = q_end.item() - q_start.item()
    k_start = ct.load(cu_seqlens_k, (batch_idx,), (1,))
    k_end = ct.load(cu_seqlens_k, (batch_idx + 1,), (1,))
    k_len = k_end.item() - k_start.item()

    offs_q = q_seq_idx * BLOCK_SIZE_Q + ct.arange(0, BLOCK_SIZE_Q, dtype=ct.int32)
    offs_d = ct.arange(0, BLOCK_SIZE_D, dtype=ct.int32)
    q_tile = ct.gather(q, (q_start + offs_q, q_head_idx.item(), offs_d))
    q_tile = q_tile.reshape((BLOCK_SIZE_Q, BLOCK_SIZE_D))
    # 我们要找出当前Q能够看到的最大的k block的索引
    # 对于第i个k block，其最大索引为 i * kernel_stride + kernel_size - 1
    # 当前Q的最大索引为 q_start + BLOCK_SIZE_Q - 1
    # i * kernel_stride + kernel_size - 1 <= q_start + BLOCK_SIZE_Q - 1
    # i <= (q_start + BLOCK_SIZE_Q - kernel_size + 1) // kernel_stride
    # range的边界是开区间，所以hi需要加1
    lo = 0
    hi = min(k_len, max(0, (q_seq_idx * BLOCK_SIZE_Q + BLOCK_SIZE_Q - kernel_size + 1) // kernel_stride + 1))

    acc = ct.full((BLOCK_SIZE_Q, BLOCK_SIZE_D), 0.0, dtype=ct.float32)
    m_i = ct.full((BLOCK_SIZE_Q, 1), -float("inf"), dtype=ct.float32)
    l_i = ct.full((BLOCK_SIZE_Q, 1), 0.0, dtype=ct.float32)

    # 遍历k block
    for i in range(lo, hi, BLOCK_SIZE_K):
        k_tile = ct.gather(k, (k_start + i + ct.arange(0, BLOCK_SIZE_K, dtype=ct.int32), k_head_idx.item(), offs_d))
        k_tile = k_tile.reshape((BLOCK_SIZE_K, BLOCK_SIZE_D))
        v_tile = ct.gather(v, (k_start + i + ct.arange(0, BLOCK_SIZE_K, dtype=ct.int32), k_head_idx.item(), offs_d))
        v_tile = v_tile.reshape((BLOCK_SIZE_K, BLOCK_SIZE_D))
        
        qk = q_tile @ k_tile.transpose(-1, -2)
        qk = qk * sm_scale
        offs_k = (i + ct.arange(0, BLOCK_SIZE_K, dtype=ct.int32)) * kernel_stride + kernel_size - 1
        mask = (offs_q[:, None] >= offs_k[None, :]) & (offs_q[:, None] < q_len)
        qk = ct.where(mask, qk, -float("inf"))

        m_ij = ct.max(qk, axis=-1, keepdims=True)
        m_new = ct.max(m_i, m_ij)
        alpha = m_i - m_new

        # Softmax   
        qk = ct.exp2(qk - m_new)
        l_i = l_i * ct.exp2(alpha) + ct.sum(qk, axis=-1, keepdims=True)
        acc = ct.mma(qk, v_tile, acc * ct.exp2(alpha))

        m_i = m_new
    
    acc = acc / l_i
    ct.scatter(o, (q_start + offs_q, q_head_idx.item(), offs_d), acc)
    lse_i = m_i + ct.log2(l_i)
    lse_i = m_i + ct.log2(l_i)
    lse_i = ct.where(offs_q < q_len, lse_i, 0.0)
    ct.scatter(lse, (q_head_idx.item(), q_start + offs_q), lse_i)        


def compress_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
):
    assert k.dtype == q.dtype and v.dtype == q.dtype
    assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32

    # shape
    q_len, num_q_heads, head_dim = q.shape
    k_len, num_k_heads, head_dim = k.shape
    v_len, num_v_heads, head_dim = v.shape
    batch_size = cu_seqlens_q.shape[0] - 1

    assert k_len == v_len and q_len > k_len
    # gqa
    assert num_k_heads == num_v_heads
    assert num_q_heads % num_k_heads == 0
    num_share_q_heads = num_q_heads // num_k_heads
    o = torch.zeros_like(q)
    lse = torch.full(
        (num_q_heads, q_len),
        fill_value=-torch.inf,
        dtype=torch.float32,
        device=q.device,
    )
    BLOCK_SIZE_Q = 128
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_D = 128

    grid = (batch_size, num_q_heads, ct.cdiv(max_seqlen_q, BLOCK_SIZE_Q))
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        compress_attention_fwd_kernel,(
        q,
        k,
        v,
        o,
        lse,
        kernel_size,
        kernel_stride,
        cu_seqlens_q,
        cu_seqlens_k,
        num_k_heads,
        num_share_q_heads,
        head_dim,
        sm_scale,
        BLOCK_SIZE_Q,
        BLOCK_SIZE_K,
        BLOCK_SIZE_D
        )
    )
    return o, lse

@ct.kernel
def get_attention_score_kernel(
    q: ct.Array,
    k: ct.Array,
    lse: ct.Array,
    score: ct.Array,
    kernel_size: int,
    kernel_stride: int,
    cu_seqlens_q: ct.Array,
    cu_seqlens_k: ct.Array,
    num_k_heads: int,
    num_share_q_heads: int,
    head_dim: int,
    sm_scale: float,
    BLOCK_SIZE_Q: ct.Constant,
    BLOCK_SIZE_K: ct.Constant,
    BLOCK_SIZE_D: ct.Constant,
):
    head_idx = ct.bid(0)
    q_seq_idx = ct.bid(1)
    k_seq_idx = ct.bid(2)

    batch_idx = head_idx // num_k_heads
    k_head_idx = head_idx % num_k_heads

    sm_scale *= INV_LOG_2
    q_start = ct.load(cu_seqlens_q, (batch_idx,), (1,))
    q_end = ct.load(cu_seqlens_q, (batch_idx + 1,), (1,))
    q_len = q_end - q_start
    k_start = ct.load(cu_seqlens_k, (batch_idx,), (1,))
    k_end = ct.load(cu_seqlens_k, (batch_idx + 1,), (1,))
    k_len = k_end - k_start

    acc = ct.full((BLOCK_SIZE_Q, BLOCK_SIZE_K), 0.0, dtype=ct.float32)
    if q_seq_idx * BLOCK_SIZE_Q >= q_len or k_seq_idx * BLOCK_SIZE_K >= k_len:
        return
    offs_d = ct.arange(0, BLOCK_SIZE_D, dtype=ct.int32)
    offs_k = k_seq_idx * BLOCK_SIZE_K + ct.arange(0, BLOCK_SIZE_K, dtype=ct.int32)
    offs_q = q_seq_idx * BLOCK_SIZE_Q + ct.arange(0, BLOCK_SIZE_Q, dtype=ct.int32)
    
    # [BLOCK_SIZE_K, BLOCK_SIZE_D]
    k_tile = ct.gather(k, (offs_k + k_start, k_head_idx, offs_d))
    
    for i in range(0, num_share_q_heads):
        # [BLOCK_SIZE_Q, BLOCK_SIZE_D]
        q_tile = ct.gather(q, (offs_q + q_start, i, offs_d))
        q_tile = q_tile.reshape((BLOCK_SIZE_Q, BLOCK_SIZE_D))
        # [BLOCK_SIZE_Q]
        lse_tile = ct.gather(lse, (i, offs_q + q_start), (1, BLOCK_SIZE_Q))
        lse_tile = lse_tile.reshape((BLOCK_SIZE_Q, 1))
        qk = q_tile @ k_tile.transpose(-1, -2)
        causal_mask = offs_q[:, None] >= (offs_k * kernel_stride + kernel_size - 1)[None, :]
        qk = qk * sm_scale

        acc += ct.where(causal_mask, ct.exp2(qk - lse_tile), 0.0)

    mask = (offs_q < q_len)[:, None] & (offs_k < k_len)[None, :]
    acc = ct.where(mask, acc, 0.0)

    ct.scatter(score, (q_start + offs_q, k_head_idx, k_start + offs_k), acc.reshape(BLOCK_SIZE_Q, 1, BLOCK_SIZE_K).astype(score.dtype))



def get_attention_score(
    q: torch.Tensor,
    k: torch.Tensor,
    lse: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
):
    q_len, num_q_heads, head_dim = q.shape
    k_len, num_k_heads, _ = k.shape
    batch_size = cu_seqlens_q.shape[0] - 1

    num_share_q_heads = num_q_heads // num_k_heads

    score = torch.zeros([num_k_heads, q_len, max_seqlen_k], dtype=torch.float32, device=q.device)
    BLOCK_SIZE_Q = 128
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_D = 128
    grid = (batch_size * num_k_heads, ct.cdiv(q_len, BLOCK_SIZE_Q), ct.cdiv(max_seqlen_k, BLOCK_SIZE_K))
    
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        get_attention_score_kernel,
        (
            q,
            k,
            lse,
            score,
            kernel_size,
            kernel_stride,
            cu_seqlens_q,
            cu_seqlens_k,
            num_k_heads,
            num_share_q_heads,
            head_dim,
            sm_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_K,
            BLOCK_SIZE_D
        )
    )

@ct.kernel
def transform_score_kernel(
    score: ct.Array,
    offs: ct.Array,
    score_block: ct.Array,
    cu_seqlens_q: ct.Array,
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    num_k_heads: int,
    num_offs: int,
    pad_len: int,
    max_seqlen_k: int,
    max_blocks: int,
    init_blocks: int,
    local_blocks: int,
    BLOCK_SIZE_Q: ct.Constant,
    BLOCK_SIZE_K: ct.Constant,
    BLOCK_SIZE_O: ct.Constant,
):
    head_idx = ct.bid(0)
    q_seq_idx = ct.bid(1)
    block_idx = ct.bid(2)
    batch_idx = head_idx // num_k_heads
    k_head_idx = head_idx % num_k_heads

    q_start = ct.load(cu_seqlens_q, (batch_idx,), (1,))
    q_end = ct.load(cu_seqlens_q, (batch_idx + 1,), (1,))
    q_len = q_end - q_start

    k_start = block_idx * BLOCK_SIZE_K
    if q_seq_idx * BLOCK_SIZE_Q >= q_len:
        return

    # load weight [BLOCK_SIZE_O]
    offs_weight = ct.arange(0, BLOCK_SIZE_O, dtype=ct.int32)
    weight_tile = ct.load(offs, (0,), (BLOCK_SIZE_O), padding_mode=ct.PaddingMode.ZERO)

    # load score [BLOCK_SIZE_O, BLOCK_SIZE_K]
    offs_k = block_idx * BLOCK_SIZE_K + ct.arange(0, BLOCK_SIZE_K, dtype=ct.int32) 
    offs_k = offs_k * (block_size // kernel_stride) - pad_len # 每个block的起始压缩token
    offs_k = offs_k[None, :] + offs_weight[:, None] # 每个block包含的压缩token
    offs_q = q_seq_idx * BLOCK_SIZE_Q + ct.arange(0, BLOCK_SIZE_Q, dtype=ct.int32)


    score_tile = ct.gather(score, (k_head_idx[:, None, None], (q_start + q_start + offs_q)[:, None, None], (k_start + offs_k)[None, :, :]))
    score_tile = score_tile * weight_tile.reshape((1, BLOCK_SIZE_O, 1))
    # [BLOCK_SIZE_Q, BLOCK_SIZE_O, BLOCK_SIZE_K] - [BLOCK_SIZE_Q, BLOCK_SIZE_K]
    score_tile = ct.sum(score_tile, axis=1)
    
    offs_q_block = offs_q // block_size
    offs_k_block = k_start + ct.arange(0, BLOCK_SIZE_K, dtype=ct.int32)
    # BLOCK级别mask
    # offs_q_block [BLOCK_SIZE_Q]: [0, 1, 2, 3, 4, 5, 6, 7] -> [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    # offs_k_block [BLOCK_SIZE_K]: [0, 1, 2, 3]
    mask = ((offs_q_block[:, None] >= offs_k_block[None, :]) & (offs_q_block[:, None] < offs_k_block[None, :] + local_blocks)) | \
           (offs_k_block[None, :] < init_blocks)
    score_tile = ct.where(mask, score_tile, float("inf"))

    ct.scatter(score_block, (k_head_idx, q_seq_idx, block_idx), score_tile.astype(score_block.dtype))



def transform_score(
    score: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    init_blocks: int,
    local_blocks: int,
):
    """
    Input: 
        score: [num_k_heads, q_len, max_seqlen_k]
        offs:  [num_offs]
    Output:
        block_score: [num_k_heads, q_len, max_blocks]

    pseudo code:
        for i in range(num_offs):
            weight = offs[i]
            block_score[:, :, :full_blocks] += score[:, :, i::block_size // kernel_stride][..., :full_blocks] * weight
    """
    num_k_heads, q_len, max_seqlen_k = score.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    # kernel_size // kernel_stride代表一个kernel size中有多少个步长，减去一就是其左侧有多少个kernel可以与该kernel重叠
    pad_len = kernel_size // kernel_stride - 1

    max_blocks = math.ceil(max_seqlen_q / block_size)
    score_block = torch.zeros([num_k_heads, q_len, max_blocks], dtype=torch.float32, device=score.device)
    offs = torch.arange(kernel_size // kernel_stride, dtype=torch.int32, device=score.device)[:, None] + \
           torch.arange(block_size // kernel_stride, dtype=torch.int32, device=score.device)[None, :]
    offs = offs.view(-1)
    # 统计offs每个元素的数量
    offs = torch.bincount(offs)
    num_offs = offs.shape[0]
    BLOCK_SIZE_Q = 8
    BLOCK_SIZE_K = min(128, next_power_of_2(max_blocks))
    BLOCK_SIZE_O = next_power_of_2(num_offs)
    grid = (batch_size * num_k_heads, ct.cdiv(q_len, BLOCK_SIZE_Q), ct.cdiv(max_blocks, BLOCK_SIZE_K))
    
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        transform_score_kernel,
        (
            score,
            offs,
            score_block,
            cu_seqlens_q,
            kernel_size,
            kernel_stride,
            block_size,
            num_k_heads,
            num_offs,
            pad_len,
            max_seqlen_k,
            max_blocks,
            init_blocks,
            local_blocks,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_K,
            BLOCK_SIZE_O
        )
    )
    return score_block

def compress_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    topk: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int = None,
    max_seqlen_k: int = None,
    sm_scale: float = None,
    init_blocks: int = 1,
    local_blocks: int = 2,
):
    if max_seqlen_q is None:
        max_seqlen_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
    if max_seqlen_k is None:
        max_seqlen_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()
    
    attn_output, lse = compress_attention_fwd(
        q,
        k,
        v,
        kernel_size,
        kernel_stride,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        sm_scale,
    )

    score = get_attention_score(
        q,
        k,
        lse,
        kernel_size,
        kernel_stride,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        sm_scale,
    )

    score = transform_score(
        score,
        kernel_size,
        kernel_stride,
        block_size,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        init_blocks,
        local_blocks
    )
    # topk
    topk_idx = score.topk(topk, dim=-1).indices.sort(-1).values # 默认是升序
    q_idx = torch.cat(
        [
            torch.arange(cu_seqlens_q[i + 1] - cu_seqlens_q[i], device=q.device)
            for i in range(cu_seqlens_q.shape[0] - 1)
        ]
        , dim=0
    )
    q_idx = q_idx // block_size
    topk_idx[topk_idx > q_idx[None, :, None]] = -1
    topk_idx = topk_idx.to(torch.int32)
   
@ct.kernel
def topk_sparse_attention_kernel(
    q: ct.Array,
    k: ct.Array,
    v: ct.Array,
    topk_idx: ct.Array,
    o: ct.Array,
    lse: ct.Array,
    cu_seqlens_q: ct.Array,
    cu_seqlens_k: ct.Array,
    num_k_heads: int,
    num_share_q_heads: int,
    head_dim: int,
    topk: int,
    num_q_loop: int,
    sm_scale: float,
    BLOCK_SIZE_K: ct.Constant,
    BLOCK_SIZE_D: ct.Constant,
    BLOCK_SIZE_H: ct.Constant,
    BLOCK_SIZE_T: ct.Constant,
):
    """
    pseudo code:
        for i in range(num_q_loop):
            q: [BLOCK_SIZE_H, BLOCK_SIZE_D]
            for j in range(topk):
                topk -> k: [BLOCK_SIZE_K, BLOCK_SIZE_D]
                v: [BLOCK_SIZE_K, BLOCK_SIZE_D]
                qk: [BLOCK_SIZE_H, BLOCK_SIZE_K]
                o: [BLOCK_SIZE_H, BLOCK_SIZE_D]

    """
    batch_idx = ct.bid(0)
    k_head_idx = ct.bid(1)
    q_idx = ct.bid(2)
    q_head_idx = k_head_idx * num_share_q_heads

    q_start = ct.load(cu_seqlens_q, (batch_idx,), (1,))
    q_end = ct.load(cu_seqlens_q, (batch_idx + 1,), (1,))
    q_len = q_end - q_start
    k_start = ct.load(cu_seqlens_k, (batch_idx,), (1,))
    k_end = ct.load(cu_seqlens_k, (batch_idx + 1,), (1,))
    k_len = k_end - k_start

    if q_idx * num_q_loop >= q_len:
        return
    min_num_q_loop = min(num_q_loop, q_len - q_idx * num_q_loop)
    offs_d = ct.arange(0, BLOCK_SIZE_D, dtype=ct.int32)
    for j in range(min_num_q_loop):
        q_j_idx = q_idx * num_q_loop + j
        # load topk
        topk_idx = ct.load(topk_idx, (k_head_idx, q_start + q_j_idx, 0), (1, 1, BLOCK_SIZE_T), padding_mode=ct.PaddingMode.ZERO)
        topk_idx = topk_idx.reshape((BLOCK_SIZE_T))
        # mask
        topk_num = ct.sum(ct.where((topk_idx >= 0) & (topk_idx <= q_j_idx // BLOCK_SIZE_K), 1, 0), axis=0)
        # load q [BLOCK_SIZE_H, BLOCK_SIZE_D] BLOCK_SIZE_H = num_share_q_heads
        q_tile = ct.load(q, (q_start + q_j_idx, k_head_idx, 0), (1, BLOCK_SIZE_H, BLOCK_SIZE_D), padding_mode=ct.PaddingMode.ZERO)
        q_tile = q_tile.reshape((BLOCK_SIZE_H, BLOCK_SIZE_D))
        
        offs_h = ct.arange(0, BLOCK_SIZE_H, dtype=ct.int32) # >= num_share_q_heads
        offs_k = ct.arange(0, BLOCK_SIZE_K, dtype=ct.int32) # >= block_size
        q_tile = ct.where(offs_h[:, None] < num_share_q_heads, q_tile, 0.0)
        m_i = ct.full((BLOCK_SIZE_H), -float("inf"), dtype=ct.float32)
        l_i = ct.full((BLOCK_SIZE_H), 0.0, dtype=ct.float32)
        acc = ct.full((BLOCK_SIZE_H, BLOCK_SIZE_D), 0.0, dtype=ct.float32)

        # topk attetion
        for i in range(topk_num):
            # load first selected k block
            k_idx = ct.load(topk_idx, (k_head_idx, q_start + q_j_idx, i), (1, 1, 1))
            k_idx = ct.int32(k_idx.item())
            offs_k_i = k_idx * BLOCK_SIZE_K + ct.arange(0, BLOCK_SIZE_K, dtype=ct.int32)
            # load k
            k_i_tile = ct.gather(k, (k_start + offs_k_i, k_head_idx, offs_d)) # [BLOCK_SIZE_K, 1, BLOCK_SIZE_D]
            k_i_tile = k_i_tile.reshape((BLOCK_SIZE_K, BLOCK_SIZE_D)).transpose(-1, -2)
            qk = q_tile @ k_i_tile # [BLOCK_SIZE_H, BLOCK_SIZE_K]
            qk = qk * sm_scale

            qk = ct.where((q_j_idx >= offs_k_i)[None, :], qk, -float("inf"))
            m_ij = ct.max(qk, axis=-1) # [BLOCK_SIZE_H]
            m_i_new = ct.max(m_i, m_ij)
            alpha = m_i - m_i_new
            qk = ct.exp2(qk - m_i_new[:, None])
            l_i *= ct.exp2(alpha)
            l_i += ct.sum(qk, axis=-1)
            v_i_tile = ct.gather(v, (k_start + offs_k_i, k_head_idx, offs_d)) # [BLOCK_SIZE_K, 1, BLOCK_SIZE_D]
            v_i_tile = v_i_tile.reshape((BLOCK_SIZE_K, BLOCK_SIZE_D))
            acc = ct.mma(qk, v_i_tile, acc * ct.exp2(alpha)[:, None])
            m_i = m_i_new

        acc = acc / l_i[:, None]
        acc = ct.where(offs_h[:, None] < num_share_q_heads, acc, 0.0)
        acc = acc.reshape((1, BLOCK_SIZE_H, BLOCK_SIZE_D))
        ct.store(o, (q_start + q_j_idx, k_head_idx, 0), acc.astype(o.dtype))
        lse_i = ct.math.log2(l_i) + m_i
        lse_i = ct.where(offs_h < num_share_q_heads, lse_i, 0.0)
        lse_i = lse_i.reshape((BLOCK_SIZE_H, 1))
        ct.store(lse, (k_head_idx, q_start + q_j_idx), lse_i.astype(lse.dtype))

def topk_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idx: torch.Tensor,
    block_size: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int = None,
    max_seqlen_k: int = None,
    sm_scale: Optional[float] = None,    
):
    """Topk sparse attention varlen version implemented in triton.

    Args:
        q (torch.Tensor): shape [total_len, num_q_heads, head_dim]
        k (torch.Tensor): shape [total_len, num_kv_heads, head_dim]
        v (torch.Tensor): shape [total_len, num_kv_heads, head_dim]
        topk_idx (torch.Tensor): topk block idx for each query, shape [num_kv_heads, total_len, topk]. -1 means padding.
        block_size (int): key value block size.
        cu_seqlens (torch.Tensor): shape [batch_size + 1], similar to cu_seqlens in flash_attn_func_varlen.
        softmax_scale (Optional[float], optional): Defaults to None, means 1/sqrt(head_dim).

    Returns:
        torch.Tensor: attention output, shape [total_len, num_q_heads, head_dim]
    """
    # dtype check
    assert k.dtype == q.dtype and v.dtype == q.dtype
    assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32
    assert block_size in {32, 64, 128, 256}
    # shape
    q_len, num_q_heads, head_dim = q.shape
    k_len, num_k_heads, head_dim = k.shape
    v_len, num_v_heads, head_dim = v.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    assert q_len == k_len and k_len == v_len
    topk = topk_idx.shape[-1]
    assert topk_idx.shape[0] == num_k_heads
    assert topk_idx.shape[1] == q_len
    # gqa
    assert num_k_heads == num_v_heads
    assert num_q_heads % num_k_heads == 0
    num_share_q_heads = num_q_heads // num_k_heads
    # output tensor
    o = torch.zeros_like(q)
    lse = torch.zeros(num_q_heads, q_len, dtype=torch.float32, device=q.device)
    # launch kernel
    num_q_loop = (
        max_seqlen_q // 32768 + 1
    )  # calculate multiple querys in one kernel if seqlence length is too long
    grid = (batch_size, num_k_heads, ct.cdiv(max_seqlen_q, num_q_loop))
    BLOCK_SIZE_K = next_power_of_2(block_size)
    BLOCK_SIZE_D = next_power_of_2(head_dim)
    BLOCK_SIZE_H = max(16, next_power_of_2(num_share_q_heads))
    BLOCK_SIZE_T = next_power_of_2(topk)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        topk_sparse_attention_kernel,
        (
            q,
            k,
            v,
            topk_idx,
            o,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            num_k_heads,
            num_share_q_heads,
            head_dim,
            topk,
            num_q_loop,
            sm_scale,
            BLOCK_SIZE_K,
            BLOCK_SIZE_D,
            BLOCK_SIZE_H,
            BLOCK_SIZE_T,
        )
    )

