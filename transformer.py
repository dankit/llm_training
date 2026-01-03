import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SwigluFFN(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or int((2/3) * 4 * dim)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_value = nn.Linear(dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, dim, bias=False)
    
    def forward(self, x):
        return self.out(F.silu(self.w_gate(x)) * self.w_value(x))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len=2048):
        super().__init__()
        self.head_dim = head_dim
        # Pre-compute and cache cos/sin for all positions up to max_seq_len
        idx = torch.arange(0, head_dim // 2)
        thetas = torch.pow(10000, -2 * idx / head_dim)
        positions = torch.arange(max_seq_len)
        m_theta = positions.unsqueeze(-1) * thetas  # (max_seq_len, head_dim//2)
        self.register_buffer("cos_cached", torch.cos(m_theta))
        self.register_buffer("sin_cached", torch.sin(m_theta))

    def forward(self, x, start_pos=0):
        seq_len = x.shape[2]
        cos = self.cos_cached[start_pos:start_pos + seq_len]  # (seq_len, head_dim//2)
        sin = self.sin_cached[start_pos:start_pos + seq_len]
        
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((x_even * cos - x_odd * sin, x_odd * cos + x_even * sin), dim=-1).flatten(-2)


class GQAAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, rotary_pe):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.kv_repeat = n_heads // n_kv_heads
        self.rotary_pe = rotary_pe
        
        self.w_q = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(dim, dim, bias=False)

    def forward(self, x, start_pos=0, kv_cache=None):
        B, T, _ = x.shape
        Q = self.w_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.w_k(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        V = self.w_v(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        Q = self.rotary_pe(Q, start_pos)
        K = self.rotary_pe(K, start_pos)
        
        # KV cache: write new K,V into pre-allocated cache, slice valid portion
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            # Write new K, V at position start_pos
            cache_k[:, :, start_pos:start_pos + T] = K
            cache_v[:, :, start_pos:start_pos + T] = V
            # Use all cached K, V up to current position
            K = cache_k[:, :, :start_pos + T]
            V = cache_v[:, :, :start_pos + T]
        
        # Fast-path: skip repeat_interleave when using MHA (n_heads == n_kv_heads)
        if self.kv_repeat > 1:
            K_expanded = K.repeat_interleave(self.kv_repeat, dim=1)
            V_expanded = V.repeat_interleave(self.kv_repeat, dim=1)
        else:
            K_expanded, V_expanded = K, V
        
        # Use causal mask when processing multiple tokens at once (training/prefill)
        # Single token generation (T=1) doesn't need masking - it attends to all previous
        is_causal = T > 1
        out = F.scaled_dot_product_attention(Q, K_expanded, V_expanded, is_causal=is_causal)
        return self.w_o(out.transpose(1, 2).contiguous().view(B, T, -1)), (cache_k, cache_v) if kv_cache is not None else (K, V)


class DecoderBlock(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, ffn_hidden_dim, rotary_pe):
        super().__init__()
        self.attn_norm = nn.RMSNorm(dim, eps=1e-6)
        self.attention = GQAAttention(dim, n_heads, n_kv_heads, rotary_pe)
        self.ffn_norm = nn.RMSNorm(dim, eps=1e-6)
        self.ffn = SwigluFFN(dim, ffn_hidden_dim)

    def forward(self, x, start_pos=0, kv_cache=None):
        attn_out, new_kv_cache = self.attention(self.attn_norm(x), start_pos, kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_kv_cache
    
    def forward_checkpoint(self, x):
        """Forward with gradient checkpointing - recomputes activations during backward."""
        def attn_fn(t):
            out, _ = self.attention(self.attn_norm(t))
            return out
        x = x + torch.utils.checkpoint.checkpoint(attn_fn, x, use_reentrant=False)
        x = x + torch.utils.checkpoint.checkpoint(
            lambda t: self.ffn(self.ffn_norm(t)), x, use_reentrant=False
        )
        return x


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = config.gradient_checkpointing
        self.embeddings = nn.Embedding(config.vocab_size, config.dim)
        ffn_hidden_dim = int(config.dim * config.ffn_hidden_mult)
        max_seq_len = config.seq_len
        
        # Share RoPE across all layers (saves memory - one buffer instead of n_layers)
        head_dim = config.dim // config.n_heads
        self.shared_rope = RotaryPositionalEmbedding(head_dim, max_seq_len)
        
        self.layers = nn.ModuleList([
            DecoderBlock(config.dim, config.n_heads, config.n_kv_heads, ffn_hidden_dim, self.shared_rope) 
            for _ in range(config.n_layers)
        ])
        self.norm = nn.RMSNorm(config.dim, eps=1e-6)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Weight tying - saves ~38M params and improves quality
        self.lm_head.weight = self.embeddings.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """GPT-style initialization: Normal(0, 0.02), scaled residual projections."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        # Scale residual projections by 1/sqrt(2*n_layers)
        scale = 0.02 / math.sqrt(2 * self.config.n_layers)
        for layer in self.layers:
            torch.nn.init.normal_(layer.attention.w_o.weight, std=scale)
            torch.nn.init.normal_(layer.ffn.out.weight, std=scale)

    def allocate_kv_cache(self, batch_size, max_seq_len, device, dtype=torch.bfloat16):
        """Pre-allocate KV cache for efficient generation."""
        head_dim = self.config.dim // self.config.n_heads
        cache_shape = (batch_size, self.config.n_kv_heads, max_seq_len, head_dim)
        return [
            (torch.zeros(cache_shape, device=device, dtype=dtype),
             torch.zeros(cache_shape, device=device, dtype=dtype))
            for _ in range(self.config.n_layers)
        ]

    def forward(self, x, start_pos=0, kv_caches=None):
        """
        Forward pass with optional KV cache for efficient generation.
        
        Args:
            x: Input token ids (B, T)
            start_pos: Position offset for RoPE (used during cached generation)
            kv_caches: List of (K, V) tuples, one per layer. None for training/prefill.
        
        Returns:
            logits: Output logits (B, T, vocab_size)
            new_kv_caches: Updated KV caches (only when kv_caches is not None or during first call)
        """
        x = self.embeddings(x)
        new_kv_caches = []
        
        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                x = layer.forward_checkpoint(x)
                new_kv_caches = None  # No cache during training
            else:
                layer_cache = kv_caches[i] if kv_caches is not None else None
                x, new_cache = layer(x, start_pos, layer_cache)
                new_kv_caches.append(new_cache)
        
        logits = self.lm_head(self.norm(x))
        return logits, new_kv_caches
