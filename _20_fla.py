import torch
import cuda.tile as ct

def prepare_lens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    return torch.diff(cu_seqlens)
def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    # cu_seqlens: [0, 16, 32, 64]
    # indices: [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    indices = torch.cat([torch.arange(n) for n in prepare_lens(ct.cdiv(cu_seqlens, chunk_size)).tolist()])
    # [[0, 0], [1, 0], [2, 0], [3, 0], [0, 1], [1, 1], [2, 1], [3, 1], [0, 2], [1, 2], [2, 3], [3, 3]]
    return torch.stack([(indices.eq(0).cumsum(-1) - 1), indices], dim=1).to(cu_seqlens)
def prepare_chunk_offsets(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    return torch.nn.functional.pad(ct.cdiv(prepare_lens(cu_seqlens), chunk_size), (1, 0), value=0).cumsum(dim=-1)

@ct.kernel
def chunk_fwd_h_kernel(
    k: ct.Array,
    v: ct.Array,
    h: ct.Array,
    g: ct.Array | None,
    g_gamma: ct.Array | None,
    gk: ct.Array | None,
    gv: ct.Array | None,
    h0: ct.Array | None,
    ht: ct.Array | None,
    cu_seqlens: ct.Array | None,
    cu_chunks: ct.Array | None,
    SEQ_LEN: ct.Constant,
    NUM_HEADS: ct.Constant,
    HEAD_DIM_K: ct.Constant,
    HEAD_DIM_V: ct.Constant,
    TILE_T: ct.Constant,
    TILE_S: ct.Constant,
    TILE_K: ct.Constant,
    TILE_V: ct.Constant,
    USE_G: ct.Constant,
    USE_G_GAMMA: ct.Constant,
    USE_GK: ct.Constant,
    USE_GV: ct.Constant,
    USE_INITIAL_STATE: ct.Constant,
    STORE_FINAL_STATE: ct.Constant,
    IS_VARLEN: ct.Constant,
):
    block_k_idx = ct.bid(0)
    block_v_idx = ct.bid(1)
    block_bh_idx = ct.bid(2)
    
    block_b_idx = block_bh_idx // NUM_HEADS
    block_h_idx = block_bh_idx % NUM_HEADS
    
    if IS_VARLEN:
        seq_start = ct.load(cu_seqlens, (block_b_idx, ), (1, ))
        seq_start = ct.int32(seq_start.item())
        seq_end = ct.load(cu_seqlens, (block_b_idx + 1, ), (1, ))
        seq_end = ct.int32(seq_end.item())
        
        seq_len = seq_end - seq_start
        num_seq_blocks = ct.cdiv(seq_len, TILE_T) # number of sequence blocks for this sequence
        
        num_states = ct.cdiv(seq_len, TILE_S) # number of states for this sequence
        
        num_seq_per_state = ct.cdiv(num_seq_blocks, num_states) # number of sequence blocks that contribute to each state
        
        chunk_start = ct.load(cu_chunks, (block_b_idx, ), (1, )) # the starting state index for this sequence
        chunk_start = ct.int32(chunk_start.item())
        
        states_start_idx = chunk_start // num_seq_per_state
    else:
        seq_len = SEQ_LEN
        num_seq_blocks = ct.cdiv(SEQ_LEN, TILE_T)
        num_states = ct.cdiv(SEQ_LEN, TILE_S)
        num_seq_per_state = ct.cdiv(num_seq_blocks, num_states)
        
    acc = ct.zeros((TILE_K, TILE_V), dtype=ct.float32)
    
    if USE_G_GAMMA:
        g_gamma_h = ct.load(g_gamma, (block_h_idx, ), (1, )).astype(ct.float32)
        g_gamma_chunk = g_gamma_h * (ct.arange(TILE_T, dtype=ct.int32) + 1)
    
    if USE_INITIAL_STATE:
        tile_h0 = ct.load(h0, (block_b_idx, block_h_idx, block_k_idx, block_v_idx), (1, 1, TILE_K, TILE_V), padding_mode=ct.PaddingMode.ZERO)
        acc += tile_h0.reshape((TILE_K, TILE_V))
    
    for block_seq_idx in range(num_seq_blocks):
        states_idx = block_seq_idx // num_seq_per_state
        mask_n = block_seq_idx * TILE_T + ct.arange(TILE_T, dtype=ct.int32) < seq_len
        if not IS_VARLEN:
            tileK = ct.load(k, (block_b_idx, block_seq_idx, block_h_idx, block_k_idx), (1, TILE_T, 1, TILE_K), padding_mode=ct.PaddingMode.ZERO)
            tileV = ct.load(v, (block_b_idx, block_seq_idx, block_h_idx, block_v_idx), (1, TILE_T, 1, TILE_V), padding_mode=ct.PaddingMode.ZERO)
        else:
            tileK = ct.load(k, (chunk_start + block_seq_idx, block_h_idx, block_k_idx), (TILE_T, 1, TILE_K), padding_mode=ct.PaddingMode.ZERO)
            tileV = ct.load(v, (chunk_start + block_seq_idx, block_h_idx, block_v_idx), (TILE_T, 1, TILE_V), padding_mode=ct.PaddingMode.ZERO)
            
            tileK = ct.where(mask_n.reshape((TILE_T, 1, 1)), tileK, 0)
            tileV = ct.where(mask_n.reshape((TILE_T, 1, 1)), tileV, 0)
        tileK = tileK.reshape((TILE_T, TILE_K)).astype(ct.float32)
        tileV = tileV.reshape((TILE_T, TILE_V)).astype(ct.float32)
        
        # Assume we have 8 tokens, and TILE_T = 2, TILE_S = 4, then 
        if block_seq_idx % num_seq_per_state == 0:
            if IS_VARLEN:
                ct.store(h, (states_start_idx + states_idx, block_h_idx, block_k_idx, block_v_idx), acc.reshape((1, 1, TILE_K, TILE_V)).astype(h.dtype))
            else:
                ct.store(h, (block_b_idx, states_idx, block_h_idx, block_k_idx, block_v_idx), acc.reshape((1, 1, 1, TILE_K, TILE_V)).astype(h.dtype))
        
        last_token_idx = min((block_seq_idx + 1) * TILE_T, seq_len) - 1
        
        if USE_G:
            if not IS_VARLEN:
                g_end = ct.load(g, (block_b_idx, last_token_idx, block_h_idx), (1, 1, 1), padding_mode=ct.PaddingMode.ZERO)
                g_end = g_end.reshape((1, 1)).astype(ct.float32)
                
                g_chunk = ct.load(g, (block_b_idx, block_seq_idx, block_h_idx), (1, TILE_T, 1), padding_mode=ct.PaddingMode.ZERO)
                g_chunk = g_chunk.reshape((TILE_T, 1)).astype(ct.float32)
            else:
                g_end = ct.load(g, (chunk_start + last_token_idx, block_h_idx), (1, 1), padding_mode=ct.PaddingMode.ZERO)
                g_end =g_end.astype(ct.float32)
                
                g_chunk = ct.load(g, (chunk_start + block_seq_idx, block_h_idx), (TILE_T, 1), padding_mode=ct.PaddingMode.ZERO)
                g_chunk = ct.where(mask_n.reshape((TILE_T, 1)), g_chunk, 0).astype(ct.float32)
                
            tileV = tileV * ct.exp(g_end - g_chunk) # [TILE_T, TILE_V]
            acc *= ct.exp(g_end) # [TILE_K, TILE_V]
        
        if USE_G_GAMMA:
            g_gamma_end = g_gamma_h * min((seq_len - TILE_T * block_seq_idx), TILE_T)
            acc *= ct.exp(g_gamma_end)
            tileV = tileV * ct.exp((g_gamma_end - g_gamma_chunk).reshape((TILE_T, 1)))
            
        if USE_GK:
            if not IS_VARLEN:
                gk_end = ct.load(gk, (block_b_idx, last_token_idx, block_h_idx, block_k_idx), (1, 1, 1, TILE_K), padding_mode=ct.PaddingMode.ZERO)
                gk_end = gk_end.reshape((1, TILE_K)).astype(ct.float32)
                
                gk_chunk = ct.load(gk, (block_b_idx, block_seq_idx, block_h_idx, block_k_idx), (1, TILE_T, 1, TILE_K), padding_mode=ct.PaddingMode.ZERO)
                gk_chunk = gk_chunk.reshape((TILE_T, TILE_K)).astype(ct.float32)
            else:
                gk_end = ct.load(gk, (seq_start + last_token_idx, block_h_idx, block_k_idx), (1, 1, TILE_K), padding_mode=ct.PaddingMode.ZERO)
                gk_end = gk_end.reshape((1, TILE_K)).astype(ct.float32)
                
                gk_chunk = ct.load(gk, (chunk_start + block_seq_idx, block_h_idx, block_k_idx), (TILE_T, 1, TILE_K), padding_mode=ct.PaddingMode.ZERO)
                gk_chunk = gk_chunk.reshape((TILE_T, TILE_K)).astype(ct.float32)
                gk_chunk = ct.where(mask_n.reshape((TILE_T, 1)), gk_chunk, 0)
                
            acc *= ct.exp(gk_end) # [TILE_K, TILE_V]
            tileK = tileK * ct.exp(gk_end - gk_chunk) # [TILE_T, TILE_K]
        
        if  USE_GV:
            if not IS_VARLEN:
                gv_end = ct.load(gv, (block_b_idx, last_token_idx, block_h_idx, block_v_idx), (1, 1, 1, TILE_V), padding_mode=ct.PaddingMode.ZERO)
                gv_end = gv_end.reshape((1, TILE_V)).astype(ct.float32)
                
                gv_chunk = ct.load(gv, (block_b_idx, block_seq_idx, block_h_idx, block_v_idx), (1, TILE_T, 1, TILE_V), padding_mode=ct.PaddingMode.ZERO)
                gv_chunk = gv_chunk.reshape((TILE_T, TILE_V)).astype(ct.float32)
            else:
                gv_end = ct.load(gv, (seq_start + last_token_idx, block_h_idx, block_v_idx), (1, 1, TILE_V), padding_mode=ct.PaddingMode.ZERO)
                gv_end = gv_end.reshape((1, TILE_V)).astype(ct.float32)
                
                gv_chunk = ct.load(gv, (chunk_start + block_seq_idx, block_h_idx, block_v_idx), (TILE_T, 1, TILE_V), padding_mode=ct.PaddingMode.ZERO)
                gv_chunk = gv_chunk.reshape((TILE_T, TILE_V)).astype(ct.float32)
                gv_chunk = ct.where(mask_n.reshape((TILE_T, 1)), gv_chunk, 0)
            acc *= ct.exp(gv_end) # [TILE_K, TILE_V]
            tileV = tileV * ct.exp(gv_end - gv_chunk) # [TILE_T, TILE_V]
        
        acc = ct.mma(tileK.transpose(-1, -2), tileV, acc)
        
    if STORE_FINAL_STATE:
        ct.store(ht, (block_b_idx, block_h_idx, block_k_idx, block_v_idx), acc.reshape((1, 1, TILE_K, TILE_V)).astype(ht.dtype))

@ct.kernel
def chunk_fwd_o_kernel(
    q : ct.Array,
    k : ct.Array,
    v : ct.Array,
    h : ct.Array,
    g : ct.Array,
    g_gamma : ct.Array,
    o : ct.Array,   
    cu_seqlens: ct.Array,
    chunk_indices: ct.Array,
    scale: float,
    SEQ_LEN: ct.Constant,
    NUM_HEADS: ct.Constant,
    HEAD_DIM_K: ct.Constant,
    HEAD_DIM_V: ct.Constant,
    TILE_T: ct.Constant,
    TILE_K: ct.Constant,
    TILE_V: ct.Constant,
    USE_G: ct.Constant,
    USE_G_GAMMA: ct.Constant,
    USE_EXP2: ct.Constant,
    IS_VARLEN: ct.Constant,
):
    block_bh_idx = ct.bid(2)
    block_v_idx = ct.bid(0)
    block_chunk_idx = ct.bid(1)

    block_b_idx = block_bh_idx // NUM_HEADS
    block_h_idx = block_bh_idx % NUM_HEADS

    if IS_VARLEN:
        # chunk_indices: [total_nchunks, 2]
        # chunk_indices[:, 0]: the sequence index [0, 0, 0, 1, 1]
        # chunk_indices[:, 1]: the chunk index of this sequence [0, 1, 2, 0, 1]
        seq_idx = ct.load(chunk_indices, (block_chunk_idx, 0), (1, 1))
        chunk_seq_idx = ct.load(chunk_indices, (block_chunk_idx, 1), (1, 1))
        seq_start = ct.load(cu_seqlens, (seq_idx.item(), ), (1, ))
        seq_end = ct.load(cu_seqlens, (seq_idx.item() + 1, ), (1, ))
        seq_len = seq_end - seq_start
    else:
        seq_len = SEQ_LEN
    
    tile_o = ct.zeros((TILE_T, TILE_V), dtype=ct.float32)
    tile_A = ct.zeros((TILE_T, TILE_T), dtype=ct.float32)

    for k_idx in range(ct.cdiv(HEAD_DIM_K, TILE_K)):
            
        # We do Q[TILE_T, TILE_K] @ S[TILE_K, TILE_V] + Q[TILE_T, TILE_K] @ K^T[TILE_K, TILE_T]
        if not IS_VARLEN:
            tile_Q = ct.load(q, (block_b_idx, block_chunk_idx, block_h_idx, k_idx), (1, TILE_T, 1, TILE_K), padding_mode=ct.PaddingMode.ZERO)
            tile_S = ct.load(h, (block_b_idx, block_chunk_idx, block_h_idx, k_idx, block_v_idx), (1, 1, 1, TILE_K, TILE_V), padding_mode=ct.PaddingMode.ZERO)
            tile_K = ct.load(k, (block_b_idx, block_chunk_idx, block_h_idx, k_idx), (1, TILE_T, 1, TILE_K), padding_mode=ct.PaddingMode.ZERO)
        else:
            tile_Q = ct.load(q, (block_chunk_idx, block_h_idx, k_idx), (TILE_T, 1, TILE_K), padding_mode=ct.PaddingMode.ZERO)
            tile_S = ct.load(h, (block_chunk_idx, block_h_idx, k_idx, block_v_idx), (1, 1, TILE_K, TILE_V), padding_mode=ct.PaddingMode.ZERO)
            tile_K = ct.load(k, (block_chunk_idx, block_h_idx, k_idx), (TILE_T, 1, TILE_K), padding_mode=ct.PaddingMode.ZERO)
            mask_n = chunk_seq_idx * TILE_T + ct.arange(TILE_T, dtype=ct.int32) < seq_len
            tile_Q = ct.where(mask_n.reshape((TILE_T, 1, 1)), tile_Q, 0)
            tile_K = ct.where(mask_n.reshape((TILE_T, 1, 1)), tile_K, 0)
            
        tile_Q = tile_Q.reshape((TILE_T, TILE_K))
        tile_S = tile_S.reshape((TILE_K, TILE_V))
        tile_K = tile_K.reshape((TILE_T, TILE_K))

        tile_o += tile_Q @ tile_S # [TILE_T, TILE_V]
        tile_A += tile_Q @ tile_K.transpose(-1, -2) # [TILE_T, TILE_T]

    if USE_G:
        if not IS_VARLEN:
            g = ct.load(g, (block_b_idx, block_chunk_idx, block_h_idx), (1, TILE_T, 1), padding_mode=ct.PaddingMode.ZERO)
            g = g.reshape((TILE_T, 1)).astype(ct.float32) # [TILE_T, TILE_V]
        else:
            g = ct.load(g, (block_chunk_idx, block_h_idx), (TILE_T, 1), padding_mode=ct.PaddingMode.ZERO)
            g = ct.where(mask_n.reshape((TILE_T, 1)), g, 0).astype(ct.float32)
        if USE_EXP2:
            tile_o *= ct.exp2(g)
            tile_A *= ct.exp2(g - g.transpose(-1, -2))
        else:
            tile_o *= ct.exp(g)
            tile_A *= ct.exp(g - g.transpose(-1, -2))
    
    if USE_G_GAMMA:
        g_gamma = ct.load(g_gamma, (block_h_idx, ), (1, )).astype(ct.float32)
        g_gamma_cumsum = g_gamma * (ct.arange(TILE_T, dtype=ct.int32) + 1)
        g_gamma_cumsum = g_gamma_cumsum.reshape((TILE_T, 1))
        if USE_EXP2:
            tile_o *= ct.exp2(g_gamma_cumsum)
            tile_A *= ct.exp2(g_gamma_cumsum - g_gamma_cumsum.transpose(-1, -2))
        else:
            tile_o *= ct.exp(g_gamma_cumsum)
            tile_A *= ct.exp(g_gamma_cumsum - g_gamma_cumsum.transpose(-1, -2))
    if not IS_VARLEN:
        offset = block_chunk_idx * TILE_T + ct.arange(TILE_T, dtype=ct.int32)
    else:
        offset = chunk_seq_idx * TILE_T + ct.arange(TILE_T, dtype=ct.int32)
    mask = (offset.reshape((TILE_T, 1)) < seq_len) & (offset.reshape((TILE_T, 1)) >= offset.reshape((1, TILE_T)))
    tile_A = ct.where(mask, tile_A, 0)
    if not IS_VARLEN:
        tile_V = ct.load(v, (block_b_idx, block_chunk_idx, block_h_idx, block_v_idx), (1, TILE_T, 1, TILE_V), padding_mode=ct.PaddingMode.ZERO)
        tile_V = tile_V.reshape((TILE_T, TILE_V)).astype(ct.float32)
    else:
        tile_V = ct.load(v, (block_chunk_idx, block_h_idx, block_v_idx), (TILE_T, 1, TILE_V), padding_mode=ct.PaddingMode.ZERO)
        mask_n = chunk_seq_idx * TILE_T + ct.arange(TILE_T, dtype=ct.int32) < seq_len
        tile_V = ct.where(mask_n.reshape((TILE_T, 1)), tile_V.reshape((TILE_T, TILE_V)), 0).astype(ct.float32)

    tile_o = tile_o * scale + tile_A @ tile_V * scale
    if not IS_VARLEN:
        tile_o = tile_o.reshape((1, TILE_T, 1, TILE_V))
        ct.store(o, (block_b_idx, block_chunk_idx, block_h_idx, block_v_idx), tile_o.astype(o.dtype))
    else:
        tile_o = tile_o.reshape((TILE_T, 1, TILE_V))
        ct.store(o, (block_chunk_idx, block_h_idx, block_v_idx), tile_o.astype(o.dtype))
def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = False,
    max_seq_len: int | None = None,
):
    """
    q: [bs, seq_len, num_heads, head_dim_q] or [total_tokens, num_heads, head_dim_q] if variable sequence length
    k: [bs, seq_len, num_heads, head_dim_k] or [total_tokens, num_heads, head_dim_k] if variable sequence length
    v: [bs, seq_len, num_heads, head_dim_v] or [total_tokens, num_heads, head_dim_v] if variable sequence length
    h: [bs, num_states, num_heads, head_dim_k, head_dim_v] or [num_states, num_heads, head_dim_k, head_dim_v]
    g: [bs, seq_len, num_heads] or [total_tokens, num_heads] if variable sequence length or None
    g_gamma: [num_heads] or None
    cu_seqlens: [bs + 1] or None
    """
    if cu_seqlens is None:
        bs, seq_len, num_heads, head_dim,  head_dim_v= *q.shape, v.shape[-1]
    else:
        bs, num_heads, head_dim, head_dim_v= q.shape[0], q.shape[1], q.shape[2], v.shape[-1]
    tileT = chunk_size
    tileK = 16
    tileV = 16
    if chunk_indices is None and cu_seqlens is not None:
        # [total_chunks, 2]
        chunk_indices = prepare_chunk_indices(cu_seqlens, tileT)
    num_chunks = ct.cdiv(seq_len, tileT) if cu_seqlens is None else len(chunk_indices)

    if scale is None:
        scale = k.shape[-1] ** -0.5
    
    empty = q.new_empty(0)
    
    g = g if g is not None else empty
    g_gamma = g_gamma if g_gamma is not None else empty
    cu_seqlens = cu_seqlens if cu_seqlens is not None else empty
    chunk_indices = chunk_indices if chunk_indices is not None else empty
    
    o = torch.empty_like(v)
    grid = (ct.cdiv(head_dim_v, tileV), num_chunks, bs * num_heads)
    ct.launch(torch.cuda.current_stream(), grid, chunk_fwd_o_kernel, 
              (
                  q, k, v, h, g, g_gamma, o, cu_seqlens, chunk_indices, scale, seq_len if cu_seqlens is empty else max_seq_len, num_heads, head_dim, head_dim_v, tileT, tileK, tileV,
                  g is not empty, g_gamma is not empty, use_exp2, cu_seqlens is not empty
              )
              )
    return o
        
def chunk_fwd_h(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    split_size: int | None = None,
    states_in_fp32: bool = False,
    max_seq_len: int | None = None,
):
    """
    k: [bs, seq_len, num_heads, head_dim_k] or [total_tokens, num_heads, head_dim_k] if variable sequence length
    v: [bs, seq_len, num_heads, head_dim_v] or [total_tokens, num_heads, head_dim_v] if variable sequence length
    g: [bs, seq_len, num_heads] or None or [total_tokens, num_heads] if variable sequence length
    g_gamma: [num_heads] or None
    gk: [bs, seq_len, num_heads, head_dim_k] or [total_tokens, num_heads, head_dim_k] or None
    gv: [bs, seq_len, num_heads, head_dim_v] or [total_tokens, num_heads, head_dim_v] or None
    h0: [bs, num_heads, head_dim_k, head_dim_v] or None
    cu_seqlens: [bs + 1] or None
    """
    if cu_seqlens is None:
        bs, seq_len, num_heads, head_dim_k, head_dim_v = *k.shape, v.shape[-1]
    else:
        bs, num_heads, head_dim_k, head_dim_v = k.shape[0], k.shape[1], k.shape[2], v.shape[-1]
    
    tileT = chunk_size
    tileS = tileT if split_size is None else split_size # We set tileS = tileT !
    assert tileS % tileT == 0, f"The `split_size` (got {tileS}) must be a multiple of `chunk_size` {tileT}"
    
    if cu_seqlens is not None:
        cu_chunks = prepare_chunk_offsets(cu_seqlens, tileS) # [bs + 1], cumsum of chunks
        num_seqs, num_states = len(cu_seqlens) - 1, cu_chunks[-1].item() # we have 3 sequences, and 4 states (2 for the last sequence)
        h = k.new_empty([num_states, num_heads, head_dim_k, head_dim_v], dtype=k.dtype if not states_in_fp32 else torch.float32)
    else:
        num_seqs = bs
        num_states_per_seq = ct.cdiv(seq_len, tileS)
        cu_chunks = None
        h = k.new_empty([bs, num_states_per_seq, num_heads, head_dim_k, head_dim_v], dtype=k.dtype if not states_in_fp32 else torch.float32)
    
    ht = k.new_empty([num_seqs, num_heads, head_dim_k, head_dim_v], dtype=k.dtype if not states_in_fp32 else torch.float32) if output_final_state else None
    
    tileK = 16
    tileV = 16
    
    grid = (ct.cdiv(head_dim_k, tileK), ct.cdiv(head_dim_v, tileV), num_seqs * num_heads)
    
    # cuda tile is not supported to pass None to kernel function.
    empty_tensor = k.new_empty(0) 
    g = g if g is not None else empty_tensor
    g_gamma = g_gamma if g_gamma is not None else empty_tensor
    gk = gk if gk is not None else empty_tensor
    gv = gv if gv is not None else empty_tensor
    h0 = h0 if h0 is not None else empty_tensor
    ht = ht if ht is not None else empty_tensor
    cu_seqlens = cu_seqlens if cu_seqlens is not None else empty_tensor
    cu_chunks = cu_chunks if cu_chunks is not None else empty_tensor

    ct.launch(torch.cuda.current_stream(), grid, chunk_fwd_h_kernel, 
              (
                  k, v, h, g, g_gamma, gk, gv, h0, ht, cu_seqlens, cu_chunks,
                  max_seq_len if cu_seqlens is not empty_tensor else seq_len, num_heads, head_dim_k, head_dim_v, tileT, tileS, tileK, tileV,
                  g is not empty_tensor, g_gamma is not empty_tensor, gk is not empty_tensor, gv is not empty_tensor, h0 is not empty_tensor, ht is not empty_tensor, cu_seqlens is not empty_tensor
              )
              )
    return h, ht
    

def FlashLinearAttention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    max_seq_len: int | None = None,
):
    # h: [bs, num_states, num_heads, head_dim_k, head_dim_v]
    # ht: [num_seqs, num_heads, head_dim_k, head_dim_v]
    if cu_seqlens is None:
        g_k = torch.randn(B, T, H, K_DIM, device=device) * 0.1
        g_v = torch.randn(B, T, H, V_DIM, device=device) * 0.1
    else:
        g_k = torch.randn([128, H, K_DIM], device=device) * 0.1
        g_v = torch.randn([128, H, V_DIM], device=device) * 0.1
    h, ht = chunk_fwd_h(
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        gk=g_k,
        gv=g_v,
        h0=initial_state,
        output_final_state=output_final_state,
        states_in_fp32=False,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        max_seq_len=max_seq_len,
    )
    
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        h=h,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        max_seq_len=max_seq_len,
    )
    
    return o, ht

if __name__ == "__main__":
    torch.manual_seed(42)
    
    # 设定测试维度
    B = 2
    T = 128
    H = 4
    K_DIM = 64
    V_DIM = 64
    CHUNK_SIZE = 16
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化张量
    q = torch.randn(B, T, H, K_DIM, device=device)
    k = torch.randn(B, T, H, K_DIM, device=device)
    v = torch.randn(B, T, H, V_DIM, device=device)
    g = torch.randn(B, T, H, device=device) * 0.1
    g_gamma = torch.randn(H, device=device) * 0.1
    # 初始化对数空间衰减门 (稍微给点负值，模拟真实的遗忘率)
    g = -torch.rand(B, T, H, device=device) * 0.1 
    
    # 注意：我们的标杆测试需要先把形状 transpose 成 [B, T, H, D]
    out_ref, _ = FlashLinearAttention(q, k, v, g=g, g_gamma=g_gamma, max_seq_len=T)
    
    print("Passed basic functionality test!")
    
    cu_seqlens = torch.tensor([0, 64, 128], device=device)  # 两个序列，长度分别为64和64
    max_seq_len = torch.max(torch.diff(cu_seqlens))
    q = torch.randn([128, H, K_DIM], device=device)  # 注意：这里的形状是 [total_seq_len, H, D]
    k = torch.randn([128, H, K_DIM], device=device)
    v = torch.randn([128, H, V_DIM], device=device)
    
    out_ref, _ = FlashLinearAttention(q, k, v, g=None, cu_seqlens=cu_seqlens, chunk_size=CHUNK_SIZE, max_seq_len=64)
    
    print("Passed variable sequence length test!")